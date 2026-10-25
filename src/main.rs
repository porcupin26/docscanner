use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::Parser;
use rayon::prelude::*;

use docscanner::enhance::{Filter, PageImage};
use docscanner::ocr::{self, markdown_document, OcrEngine, OcrMode, OcrPage};
use docscanner::pdf::{encode_jpeg, write_pdf, PageSize};
use docscanner::scanner::{debug_overlay, load_image, parse_corners, scan_image, Deskew, ScanOptions};

const IMAGE_EXTS: &[&str] = &["jpg", "jpeg", "png", "tif", "tiff", "webp", "bmp"];

/// Scan documents from photos: automatic page detection, perspective correction,
/// shadow removal and enhancement, exported as a multi-page PDF or images.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Cli {
    /// Input images, or directories of images (pages are ordered by file name)
    #[arg(required = true)]
    inputs: Vec<PathBuf>,

    /// Output: a .pdf file, an image file (.jpg/.png), or a directory for per-page images
    #[arg(short, long)]
    output: PathBuf,

    /// Enhancement filter
    #[arg(short, long, value_enum, default_value_t = Filter::Auto)]
    filter: Filter,

    /// Disable automatic page detection and cropping
    #[arg(long)]
    no_crop: bool,

    /// Manual page corners in source pixels, e.g. "120,80 1900,95 1950,2600 90,2580" (single input only)
    #[arg(long, conflicts_with = "no_crop")]
    corners: Option<String>,

    /// Rotate pages clockwise after flattening
    #[arg(long, default_value_t = 0, value_parser = clap::builder::TypedValueParser::map(
        clap::builder::PossibleValuesParser::new(["0", "90", "180", "270"]),
        |s: String| s.parse::<u32>().unwrap()
    ))]
    rotate: u32,

    /// Straighten tilted text lines
    #[arg(long, value_enum, default_value_t = Deskew::Auto)]
    deskew: Deskew,

    /// PDF page size
    #[arg(long, value_enum, default_value_t = PageSize::Fit)]
    page_size: PageSize,

    /// Maximum output resolution in dots per inch
    #[arg(long, default_value_t = 300.0)]
    dpi: f64,

    /// JPEG quality for colour/grey pages (1-100)
    #[arg(short, long, default_value_t = 85, value_parser = clap::value_parser!(u8).range(1..=100))]
    quality: u8,

    /// Recognise text locally and make the PDF searchable: `best` adds a local vision model (Ollama) to Tesseract
    #[arg(long, value_enum, num_args = 0..=1, default_missing_value = "best")]
    ocr: Option<OcrMode>,

    /// Write the recognised text as Markdown to this file (implies --ocr)
    #[arg(long)]
    text: Option<PathBuf>,

    /// Ollama vision model for --ocr best (default: saved setting, else first installed)
    #[arg(long)]
    ocr_model: Option<String>,

    /// Tesseract languages, e.g. "pol+eng" (default: saved setting, else all installed)
    #[arg(long)]
    ocr_lang: Option<String>,

    /// Write detection overlays into this directory
    #[arg(long)]
    debug: Option<PathBuf>,
}

fn has_image_ext(p: &Path) -> bool {
    p.extension().and_then(|e| e.to_str()).is_some_and(|e| IMAGE_EXTS.contains(&e.to_ascii_lowercase().as_str()))
}

fn collect_inputs(inputs: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for input in inputs {
        if input.is_dir() {
            let mut entries: Vec<PathBuf> = std::fs::read_dir(input)?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.is_file() && has_image_ext(p))
                .collect();
            entries.sort();
            files.extend(entries);
        } else if input.is_file() {
            files.push(input.clone());
        } else {
            bail!("input not found: {}", input.display());
        }
    }
    if files.is_empty() {
        bail!("no input images found");
    }
    Ok(files)
}

fn save_page(page: &PageImage, path: &Path, quality: u8) -> Result<()> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    if ext == "jpg" || ext == "jpeg" {
        std::fs::write(path, encode_jpeg(page, quality)?)?;
    } else {
        match page {
            PageImage::Color(i) => i.save(path)?,
            PageImage::Gray(i) | PageImage::Bilevel(i) => i.save(path)?,
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let files = collect_inputs(&cli.inputs)?;
    let corners = cli.corners.as_deref().map(parse_corners).transpose()?;
    if corners.is_some() && files.len() != 1 {
        bail!("--corners can only be used with a single input image");
    }
    if let Some(d) = &cli.debug {
        std::fs::create_dir_all(d)?;
    }
    let opts = ScanOptions {
        filter: cli.filter,
        auto_crop: !cli.no_crop,
        corners,
        rotate: cli.rotate,
        deskew: cli.deskew,
        page_size: cli.page_size,
        dpi: cli.dpi,
    };

    let started = Instant::now();
    let pages: Vec<PageImage> = files
        .par_iter()
        .enumerate()
        .map(|(i, path)| -> Result<PageImage> {
            let t = Instant::now();
            let img = load_image(path).with_context(|| format!("reading {}", path.display()))?;
            let (sw, sh) = img.dimensions();
            let overlay_src = cli.debug.as_ref().map(|_| img.clone());
            let page = scan_image(img, &opts);

            let found = match &page.detection {
                Some(d) => format!("page found via {} (score {:.2})", d.method, d.score),
                None => "no page edges found, using full frame".into(),
            };
            let (w, h) = page.image.dimensions();
            let skew = if page.skew != 0.0 { format!(", deskewed {:.2}°", page.skew) } else { String::new() };
            eprintln!(
                "[{}/{}] {}: {sw}x{sh} → {w}x{h} {} — {found}{skew} ({:.2}s)",
                i + 1,
                files.len(),
                path.display(),
                page.image.kind(),
                t.elapsed().as_secs_f64()
            );

            if let (Some(dir), Some(src), Some(d)) = (&cli.debug, overlay_src, &page.detection) {
                let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("page");
                debug_overlay(&src, &d.quad).save(dir.join(format!("{stem}-detect.jpg")))?;
            }
            Ok(page.image)
        })
        .collect::<Result<_>>()?;

    let ocr: Vec<Option<OcrPage>> = if cli.ocr.is_some() || cli.text.is_some() {
        ocr::tesseract_version()?;
        let mut cfg = ocr::load_config();
        cfg.mode = cli.ocr.unwrap_or(cfg.mode);
        cfg.model = cli.ocr_model.clone().or(cfg.model);
        cfg.languages = cli.ocr_lang.clone().unwrap_or(cfg.languages);
        let client = OcrEngine::new(cfg);
        let t = Instant::now();
        let results: Vec<Option<OcrPage>> = pages
            .par_iter()
            .enumerate()
            .map(|(i, page)| match client.recognize(page) {
                Ok(o) => {
                    eprintln!("[{}/{}] OCR: {} lines, text by {}", i + 1, pages.len(), o.lines.len(), o.engine);
                    if let Some(w) = &o.warning {
                        eprintln!("[{}/{}] warning: {w}", i + 1, pages.len());
                    }
                    Some(o)
                }
                Err(e) => {
                    eprintln!("[{}/{}] OCR failed: {e:#}", i + 1, pages.len());
                    None
                }
            })
            .collect();
        eprintln!("OCR done in {:.1}s", t.elapsed().as_secs_f64());
        results
    } else {
        Vec::new()
    };
    if let Some(path) = &cli.text {
        std::fs::write(path, markdown_document(&ocr))?;
        eprintln!("wrote {}", path.display());
    }

    let out = &cli.output;
    let ext = out.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    if ext == "pdf" {
        let bytes = write_pdf(&pages, &ocr, cli.page_size, cli.quality)?;
        std::fs::write(out, &bytes)?;
        eprintln!("wrote {} ({} pages, {:.1} KiB)", out.display(), pages.len(), bytes.len() as f64 / 1024.0);
    } else if has_image_ext(out) {
        if pages.len() == 1 {
            save_page(&pages[0], out, cli.quality)?;
            eprintln!("wrote {}", out.display());
        } else {
            let stem = out.file_stem().and_then(|s| s.to_str()).unwrap_or("page");
            for (i, page) in pages.iter().enumerate() {
                let p = out.with_file_name(format!("{stem}-{:03}.{ext}", i + 1));
                save_page(page, &p, cli.quality)?;
                eprintln!("wrote {}", p.display());
            }
        }
    } else {
        std::fs::create_dir_all(out)?;
        for (page, src) in pages.iter().zip(&files) {
            let stem = src.file_stem().and_then(|s| s.to_str()).unwrap_or("page");
            let ext = if matches!(page, PageImage::Bilevel(_)) { "png" } else { "jpg" };
            let p = out.join(format!("{stem}.{ext}"));
            save_page(page, &p, cli.quality)?;
            eprintln!("wrote {}", p.display());
        }
    }
    eprintln!("done in {:.2}s", started.elapsed().as_secs_f64());
    Ok(())
}
