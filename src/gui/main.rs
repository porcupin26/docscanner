//! Desktop GUI: import photos, adjust the detected page corners, pick a filter
//! with a live preview, reorder pages and export a PDF or images.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod sane;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};

use eframe::egui::{
    self, pos2, Align2, Color32, ColorImage, FontId, Key, Modifiers, Pos2, Rect, Sense, Shape, Stroke, TextureHandle,
    TextureOptions, Vec2,
};
use image::{imageops, RgbImage};
use rayon::prelude::*;

use docscanner::detect::detect_document;
use docscanner::enhance::{Filter, PageImage};
use docscanner::geometry::{order_corners, Point, Quad};
use docscanner::ocr::{self, markdown_document, OcrConfig, OcrEngine, OcrMode, OcrPage};
use docscanner::pdf::{encode_jpeg, write_pdf, PageSize};
use docscanner::scanner::{load_image, scan_image, Deskew, ScanOptions};
use sane::{Device, DeviceOptions, ScanHandle, ScanSettings};

/// Long side of the working copy used for on-screen display and live previews.
const WORK_SIZE: u32 = 1600;
const IMAGE_EXTS: &[&str] = &["jpg", "jpeg", "png", "tif", "tiff", "webp", "bmp"];
const ACCENT: Color32 = Color32::from_rgb(0, 170, 255);

struct Page {
    id: u64,
    name: String,
    source: Arc<RgbImage>,
    work: Arc<RgbImage>,
    source_tex: TextureHandle,
    /// Page corners in source pixel coordinates.
    quad: Quad,
    detected: Option<Quad>,
    filter: Filter,
    rotate: u32,
    straighten: bool,
    preview: Option<TextureHandle>,
    dirty: bool,
    in_flight: bool,
    /// Recognised text, tagged with the page geometry it was made for.
    ocr: Option<(GeomKey, OcrPage)>,
    ocr_busy: bool,
    ocr_error: Option<String>,
}

/// Crop/rotation state; OCR bounding boxes stay valid while it is unchanged.
type GeomKey = (Quad, u32, bool);

impl Page {
    fn full_frame(&self) -> Quad {
        let (w, h) = (self.source.width() as f64 - 1.0, self.source.height() as f64 - 1.0);
        [Point::new(0.0, 0.0), Point::new(w, 0.0), Point::new(w, h), Point::new(0.0, h)]
    }

    fn geom_key(&self) -> GeomKey {
        (self.quad, self.rotate, self.straighten)
    }

    fn current_ocr(&self) -> Option<&OcrPage> {
        self.ocr.as_ref().filter(|(k, _)| *k == self.geom_key()).map(|(_, o)| o)
    }

    fn options(&self, corners: Quad, page_size: PageSize, dpi: f64) -> ScanOptions {
        ScanOptions {
            filter: self.filter,
            auto_crop: false,
            corners: Some(corners),
            rotate: self.rotate,
            deskew: if self.straighten { Deskew::Always } else { Deskew::Never },
            page_size,
            dpi,
        }
    }
}

enum Msg {
    Loaded { path: PathBuf, source: RgbImage, work: RgbImage, detected: Option<Quad> },
    LoadFailed(PathBuf, String),
    Preview { id: u64, image: ColorImage },
    Exported(Result<String, String>),
    Devices(Result<Vec<Device>, String>),
    Options { device: String, result: Result<DeviceOptions, String> },
    ScanFinished(Result<u32, String>),
    Ocr { id: u64, key: GeomKey, result: Result<OcrPage, String> },
    OcrModels(Result<Vec<String>, String>),
}

#[derive(Clone, Copy, PartialEq)]
enum ExportKind {
    Pdf,
    Images,
    Text,
}

#[derive(Clone, Copy, PartialEq)]
enum RightTab {
    Result,
    Text,
}

/// State of the "Scan" dialog.
struct ScanUi {
    open: bool,
    devices: Vec<Device>,
    listing: bool,
    listed: bool,
    settings: ScanSettings,
    options: Option<DeviceOptions>,
    querying: bool,
    running: Option<ScanHandle>,
    next_page: u32,
    message: Option<(bool, String)>,
}

impl ScanUi {
    fn new() -> Self {
        Self {
            open: false,
            devices: Vec::new(),
            listing: false,
            listed: false,
            settings: sane::load_settings().unwrap_or_default(),
            options: None,
            querying: false,
            running: None,
            next_page: 1,
            message: None,
        }
    }

    /// Keeps the remembered choice when the device offers it, otherwise picks a sensible default.
    fn adopt_options(&mut self, opts: DeviceOptions) {
        let pick = |cur: &Option<String>, list: &[String], prefer: &str| -> Option<String> {
            cur.clone()
                .filter(|c| list.contains(c))
                .or_else(|| list.iter().find(|v| v.eq_ignore_ascii_case(prefer)).cloned())
                .or_else(|| list.first().cloned())
        };
        let s = &mut self.settings;
        s.source = pick(&s.source, &opts.sources, "Flatbed");
        s.mode = pick(&s.mode, &opts.modes, "Color");
        s.resolution = s
            .resolution
            .filter(|r| opts.resolutions.contains(r))
            .or_else(|| opts.resolutions.iter().copied().filter(|&r| r <= 300).max())
            .or_else(|| opts.resolutions.first().copied());
        s.paper = s
            .paper
            .clone()
            .filter(|p| p == sane::FULL_AREA || sane::PAPER_SIZES.iter().any(|(n, _, _)| n == p))
            .or_else(|| Some("Letter".into()));
        self.options = Some(opts);
    }
}

struct Export {
    done: Arc<AtomicUsize>,
    total: usize,
}

#[derive(Clone, Copy)]
enum ListAction {
    Select(usize),
    MoveUp(usize),
    MoveDown(usize),
    Remove(usize),
}

struct App {
    pages: Vec<Page>,
    selected: usize,
    next_id: u64,
    tx: mpsc::Sender<Msg>,
    rx: mpsc::Receiver<Msg>,
    loading: usize,
    export: Option<Export>,
    status: String,
    page_size: PageSize,
    dpi: f64,
    quality: u8,
    scan: ScanUi,
    right_tab: RightTab,
    ocr_on_export: bool,
    ocr_cfg: OcrConfig,
    /// Installed Ollama vision models; `None` until fetched.
    ocr_models: Option<Result<Vec<String>, String>>,
    ocr_models_loading: bool,
    tesseract: Result<String, String>,
    tess_langs: Vec<String>,
    settings_open: bool,
    settings_note: Option<String>,
}

/// Loads a photo or scan, detects the page and prepares the on-screen working copy.
fn load_page(path: PathBuf) -> Msg {
    match load_image(&path) {
        Ok(source) => {
            let detected = detect_document(&source).map(|d| d.quad);
            let (w, h) = source.dimensions();
            let s = (WORK_SIZE as f64 / w.max(h) as f64).min(1.0);
            let work = imageops::resize(
                &source,
                ((w as f64 * s).round() as u32).max(1),
                ((h as f64 * s).round() as u32).max(1),
                imageops::FilterType::Triangle,
            );
            Msg::Loaded { path, source, work, detected }
        }
        Err(e) => Msg::LoadFailed(path, format!("{e:#}")),
    }
}

fn to_color_image(p: &PageImage) -> ColorImage {
    let (w, h) = p.dimensions();
    match p {
        PageImage::Color(i) => ColorImage::from_rgb([w as usize, h as usize], i.as_raw()),
        PageImage::Gray(i) | PageImage::Bilevel(i) => ColorImage::from_gray([w as usize, h as usize], i.as_raw()),
    }
}

fn is_image(p: &Path) -> bool {
    p.extension().and_then(|e| e.to_str()).is_some_and(|e| IMAGE_EXTS.contains(&e.to_ascii_lowercase().as_str()))
}

fn filter_label(f: Filter) -> &'static str {
    match f {
        Filter::Auto => "✨ Auto",
        Filter::Color => "🎨 Color",
        Filter::Gray => "◐ Gray",
        Filter::Bw => "◼ B&W",
        Filter::Original => "📷 Original",
    }
}

/// Largest rect of the image's aspect ratio centred inside `area`.
fn fit_rect(area: Rect, w: f32, h: f32) -> (Rect, f32) {
    let scale = (area.width() / w).min(area.height() / h);
    (Rect::from_center_size(area.center(), Vec2::new(w * scale, h * scale)), scale)
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>, files: Vec<PathBuf>) -> Self {
        let (tx, rx) = mpsc::channel();
        let mut app = Self {
            pages: Vec::new(),
            selected: 0,
            next_id: 1,
            tx,
            rx,
            loading: 0,
            export: None,
            status: "Drop photos of documents onto the window, or click Open.".into(),
            page_size: PageSize::Fit,
            dpi: 300.0,
            quality: 85,
            scan: ScanUi::new(),
            right_tab: RightTab::Result,
            ocr_on_export: false,
            ocr_cfg: ocr::load_config(),
            ocr_models: None,
            ocr_models_loading: false,
            tesseract: ocr::tesseract_version().map_err(|e| e.to_string()),
            tess_langs: ocr::tesseract_languages().unwrap_or_default(),
            settings_open: false,
            settings_note: None,
        };
        app.refresh_ocr_models(&cc.egui_ctx);
        if !files.is_empty() {
            app.load_files(&cc.egui_ctx, files);
        }
        app
    }

    /// Loads and auto-detects pages in the background, one batch thread so page order is kept.
    fn load_files(&mut self, ctx: &egui::Context, paths: Vec<PathBuf>) {
        let mut files = Vec::new();
        for p in paths {
            if p.is_dir() {
                let mut entries: Vec<PathBuf> = std::fs::read_dir(&p)
                    .map(|rd| rd.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| is_image(p)).collect())
                    .unwrap_or_default();
                entries.sort();
                files.extend(entries);
            } else {
                files.push(p);
            }
        }
        if files.is_empty() {
            return;
        }
        self.loading += files.len();
        let (tx, ctx) = (self.tx.clone(), ctx.clone());
        std::thread::spawn(move || {
            for path in files {
                let _ = tx.send(load_page(path));
                ctx.request_repaint();
            }
        });
    }

    fn poll(&mut self, ctx: &egui::Context) {
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                Msg::Loaded { path, source, work, detected } => {
                    self.loading = self.loading.saturating_sub(1);
                    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                    let tex = ctx.load_texture(
                        format!("src-{}", self.next_id),
                        ColorImage::from_rgb([work.width() as usize, work.height() as usize], work.as_raw()),
                        TextureOptions::LINEAR,
                    );
                    let mut page = Page {
                        id: self.next_id,
                        name,
                        source: Arc::new(source),
                        work: Arc::new(work),
                        source_tex: tex,
                        quad: [Point::default(); 4],
                        detected,
                        filter: Filter::Auto,
                        rotate: 0,
                        straighten: detected.is_none(),
                        preview: None,
                        dirty: true,
                        in_flight: false,
                        ocr: None,
                        ocr_busy: false,
                        ocr_error: None,
                    };
                    page.quad = detected.unwrap_or_else(|| page.full_frame());
                    self.status = match detected {
                        Some(_) => format!("Loaded {} — page detected", page.name),
                        None => format!("Loaded {} — no page edges found, drag the corners to crop", page.name),
                    };
                    self.next_id += 1;
                    self.pages.push(page);
                    if self.scan.running.is_some() || self.pages.len() == 1 {
                        self.selected = self.pages.len() - 1;
                    }
                }
                Msg::LoadFailed(path, err) => {
                    self.loading = self.loading.saturating_sub(1);
                    self.status = format!("Could not open {}: {err}", path.display());
                }
                Msg::Preview { id, image } => {
                    if let Some(p) = self.pages.iter_mut().find(|p| p.id == id) {
                        p.in_flight = false;
                        match &mut p.preview {
                            Some(t) => t.set(image, TextureOptions::LINEAR),
                            None => p.preview = Some(ctx.load_texture(format!("prev-{id}"), image, TextureOptions::LINEAR)),
                        }
                    }
                }
                Msg::Exported(res) => {
                    self.export = None;
                    self.status = res.unwrap_or_else(|e| format!("Export failed: {e}"));
                }
                Msg::Devices(res) => {
                    self.scan.listing = false;
                    self.scan.listed = true;
                    match res {
                        Ok(devs) => {
                            if devs.is_empty() {
                                self.scan.message = Some((true, "No scanners found. Is it switched on and on the network?".into()));
                            }
                            let known = devs.iter().any(|d| d.name == self.scan.settings.device);
                            self.scan.devices = devs;
                            if !known {
                                if let Some(first) = self.scan.devices.first().map(|d| d.name.clone()) {
                                    self.select_device(ctx, first);
                                }
                            }
                        }
                        Err(e) => self.scan.message = Some((true, e)),
                    }
                }
                Msg::Options { device, result } => {
                    if device == self.scan.settings.device {
                        self.scan.querying = false;
                        match result {
                            Ok(opts) => {
                                self.scan.adopt_options(opts);
                                self.scan.message = None;
                            }
                            Err(e) => self.scan.message = Some((true, format!("Scanner not responding: {e}"))),
                        }
                    }
                }
                Msg::OcrModels(res) => {
                    self.ocr_models_loading = false;
                    self.ocr_models = Some(res);
                }
                Msg::Ocr { id, key, result } => {
                    if let Some(p) = self.pages.iter_mut().find(|p| p.id == id) {
                        p.ocr_busy = false;
                        match result {
                            Ok(o) => {
                                p.ocr = Some((key, o));
                                p.ocr_error = None;
                            }
                            Err(e) => p.ocr_error = Some(e),
                        }
                    }
                }
                Msg::ScanFinished(res) => {
                    self.scan.running = None;
                    match res {
                        Ok(n) => {
                            let more = if self.scan.settings.uses_feeder() { "Load more pages and scan again, or export." } else { "Place the next page and scan again, or export." };
                            self.scan.message = Some((false, format!("Scanned {n} page{}. {more}", if n == 1 { "" } else { "s" })));
                            self.status = format!("Scanned {n} page(s)");
                        }
                        Err(e) => self.scan.message = Some((true, e)),
                    }
                }
            }
        }
    }

    /// Re-renders previews of edited pages, at most one job per page at a time.
    fn schedule_previews(&mut self, ctx: &egui::Context) {
        for p in self.pages.iter_mut().filter(|p| p.dirty && !p.in_flight) {
            p.dirty = false;
            p.in_flight = true;
            let s = p.work.width() as f64 / p.source.width() as f64;
            let opts = p.options(p.quad.map(|c| c * s), PageSize::Fit, 1e6);
            let (work, id, tx, ctx) = (p.work.clone(), p.id, self.tx.clone(), ctx.clone());
            rayon::spawn(move || {
                let page = scan_image((*work).clone(), &opts);
                let _ = tx.send(Msg::Preview { id, image: to_color_image(&page.image) });
                ctx.request_repaint();
            });
        }
    }

    fn open_dialog(&mut self, ctx: &egui::Context) {
        if let Some(files) = rfd::FileDialog::new().add_filter("Images", IMAGE_EXTS).set_title("Add document photos").pick_files() {
            self.load_files(ctx, files);
        }
    }

    /// Opens the OCR settings when Tesseract is missing; returns whether OCR can run.
    fn ensure_ocr_ready(&mut self) -> bool {
        if self.tesseract.is_ok() {
            return true;
        }
        self.settings_open = true;
        self.settings_note = Some("Text recognition needs Tesseract — see below.".into());
        false
    }

    fn refresh_ocr_models(&mut self, ctx: &egui::Context) {
        self.ocr_models_loading = true;
        let (url, tx, ctx) = (self.ocr_cfg.ollama_url.clone(), self.tx.clone(), ctx.clone());
        std::thread::spawn(move || {
            let _ = tx.send(Msg::OcrModels(ocr::vision_models(&url).map_err(|e| format!("{e:#}"))));
            ctx.request_repaint();
        });
    }

    fn run_ocr(&mut self, ctx: &egui::Context, idx: usize) {
        if !self.ensure_ocr_ready() {
            return;
        }
        let p = &mut self.pages[idx];
        p.ocr_busy = true;
        p.ocr_error = None;
        let (source, opts, id, key) = (p.source.clone(), p.options(p.quad, self.page_size, self.dpi), p.id, p.geom_key());
        let (engine, tx, ctx) = (OcrEngine::new(self.ocr_cfg.clone()), self.tx.clone(), ctx.clone());
        std::thread::spawn(move || {
            let page = scan_image((*source).clone(), &opts);
            let result = engine.recognize(&page.image).map_err(|e| format!("{e:#}"));
            let _ = tx.send(Msg::Ocr { id, key, result });
            ctx.request_repaint();
        });
    }

    fn start_export(&mut self, ctx: &egui::Context, kind: ExportKind) {
        if self.pages.is_empty() || self.export.is_some() {
            return;
        }
        let needs_ocr = kind == ExportKind::Text || (kind == ExportKind::Pdf && self.ocr_on_export);
        if needs_ocr && !self.ensure_ocr_ready() {
            return;
        }
        let target = match kind {
            ExportKind::Pdf => rfd::FileDialog::new().add_filter("PDF", &["pdf"]).set_file_name("scan.pdf").save_file(),
            ExportKind::Text => rfd::FileDialog::new().add_filter("Markdown", &["md", "txt"]).set_file_name("scan.md").save_file(),
            ExportKind::Images => rfd::FileDialog::new().set_title("Folder for page images").pick_folder(),
        };
        let Some(target) = target else { return };

        struct Job {
            source: Arc<RgbImage>,
            opts: ScanOptions,
            id: u64,
            key: GeomKey,
            cached: Option<OcrPage>,
        }
        let (page_size, dpi, quality) = (self.page_size, self.dpi, self.quality);
        let jobs: Vec<Job> = self
            .pages
            .iter()
            .map(|p| Job {
                source: p.source.clone(),
                opts: p.options(p.quad, page_size, dpi),
                id: p.id,
                key: p.geom_key(),
                cached: p.current_ocr().cloned(),
            })
            .collect();
        let client = needs_ocr.then(|| OcrEngine::new(self.ocr_cfg.clone()));
        let done = Arc::new(AtomicUsize::new(0));
        self.export = Some(Export { done: done.clone(), total: jobs.len() });
        self.status = if needs_ocr { "Exporting and recognising text…".into() } else { "Exporting…".into() };
        let (tx, ctx) = (self.tx.clone(), ctx.clone());
        std::thread::spawn(move || {
            let failures = std::sync::Mutex::new(Vec::<String>::new());
            let run = || -> anyhow::Result<String> {
                let results: Vec<(PageImage, Option<OcrPage>)> = jobs
                    .par_iter()
                    .enumerate()
                    .map(|(i, job)| {
                        let page = scan_image((*job.source).clone(), &job.opts).image;
                        let text = match (&client, &job.cached) {
                            (None, _) => None,
                            (Some(_), Some(cached)) => Some(cached.clone()),
                            (Some(c), None) => match c.recognize(&page) {
                                Ok(o) => {
                                    let _ = tx.send(Msg::Ocr { id: job.id, key: job.key, result: Ok(o.clone()) });
                                    Some(o)
                                }
                                Err(e) => {
                                    failures.lock().unwrap().push(format!("page {}: {e:#}", i + 1));
                                    None
                                }
                            },
                        };
                        done.fetch_add(1, Ordering::Relaxed);
                        ctx.request_repaint();
                        (page, text)
                    })
                    .collect();
                let (pages, texts): (Vec<PageImage>, Vec<Option<OcrPage>>) = results.into_iter().unzip();
                let mut msg = match kind {
                    ExportKind::Pdf => {
                        let bytes = write_pdf(&pages, &texts, page_size, quality)?;
                        std::fs::write(&target, &bytes)?;
                        let searchable = if client.is_some() { "searchable, " } else { "" };
                        format!("Saved {} ({searchable}{} pages, {:.1} MB)", target.display(), pages.len(), bytes.len() as f64 / 1e6)
                    }
                    ExportKind::Text => {
                        std::fs::write(&target, markdown_document(&texts))?;
                        format!("Saved text of {} pages to {}", pages.len(), target.display())
                    }
                    ExportKind::Images => {
                        for (i, page) in pages.iter().enumerate() {
                            let path = target.join(format!(
                                "page-{:03}.{}",
                                i + 1,
                                if matches!(page, PageImage::Bilevel(_)) { "png" } else { "jpg" }
                            ));
                            match page {
                                PageImage::Bilevel(img) => img.save(&path)?,
                                _ => std::fs::write(&path, encode_jpeg(page, quality)?)?,
                            }
                        }
                        format!("Saved {} page images to {}", pages.len(), target.display())
                    }
                };
                let failures = failures.lock().unwrap();
                if let Some(first) = failures.first() {
                    msg += &format!(" — OCR failed on {} page(s): {first}", failures.len());
                }
                Ok(msg)
            };
            let _ = tx.send(Msg::Exported(run().map_err(|e| format!("{e:#}"))));
            ctx.request_repaint();
        });
    }

    fn settings_window(&mut self, ctx: &egui::Context) {
        let mut open = self.settings_open;
        let before = self.ocr_cfg.clone();
        let mut refresh = false;
        egui::Window::new("⚙ Text recognition")
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .default_width(460.0)
            .anchor(Align2::CENTER_TOP, Vec2::new(0.0, 80.0))
            .show(ctx, |ui| {
                ui.label("OCR runs entirely on this computer with open-source tools — nothing is uploaded.");
                if let Some(note) = &self.settings_note {
                    ui.colored_label(Color32::from_rgb(255, 200, 90), note);
                }
                ui.add_space(6.0);
                let cfg = &mut self.ocr_cfg;
                ui.radio_value(&mut cfg.mode, OcrMode::Best, "Best — Tesseract + local AI model (tables, handwriting; ~10 s/page on GPU)");
                ui.radio_value(&mut cfg.mode, OcrMode::Fast, "Fast — Tesseract only (~1 s/page)");
                ui.add_space(6.0);

                egui::Grid::new("ocr-grid").num_columns(2).spacing([12.0, 8.0]).show(ui, |ui| {
                    ui.label("Tesseract");
                    match &self.tesseract {
                        Ok(v) => ui.label(format!("✔ {v}")),
                        Err(e) => ui.colored_label(Color32::from_rgb(255, 110, 100), format!("✖ {e}")),
                    };
                    ui.end_row();

                    ui.label("Languages");
                    ui.horizontal_wrapped(|ui| {
                        let mut chosen: Vec<String> = cfg.languages.split('+').filter(|l| !l.is_empty()).map(String::from).collect();
                        for lang in &self.tess_langs {
                            let mut on = chosen.contains(lang);
                            if ui.checkbox(&mut on, lang).changed() {
                                if on {
                                    chosen.push(lang.clone());
                                } else if chosen.len() > 1 {
                                    chosen.retain(|l| l != lang);
                                }
                            }
                        }
                        cfg.languages = chosen.join("+");
                    });
                    ui.end_row();

                    ui.label("AI model");
                    ui.add_enabled_ui(cfg.mode == OcrMode::Best, |ui| {
                        ui.horizontal(|ui| match &self.ocr_models {
                            Some(Ok(models)) if !models.is_empty() => {
                                let current = cfg.model.clone().filter(|m| models.contains(m)).unwrap_or_else(|| models[0].clone());
                                egui::ComboBox::from_id_salt("ocr-model").selected_text(&current).width(220.0).show_ui(ui, |ui| {
                                    for m in models {
                                        if ui.selectable_label(*m == current, m).clicked() {
                                            cfg.model = Some(m.clone());
                                        }
                                    }
                                });
                                if self.ocr_models_loading {
                                    ui.spinner();
                                } else if ui.button("🔄").on_hover_text("Reload models from Ollama").clicked() {
                                    refresh = true;
                                }
                            }
                            Some(Ok(_)) => {
                                ui.colored_label(Color32::from_rgb(255, 200, 90), "No vision model installed — run: ollama pull gemma4:12b");
                                if ui.button("🔄").clicked() {
                                    refresh = true;
                                }
                            }
                            Some(Err(e)) => {
                                ui.colored_label(Color32::from_rgb(255, 110, 100), format!("✖ {e}"));
                                if ui.button("🔄").clicked() {
                                    refresh = true;
                                }
                            }
                            None => {
                                ui.spinner();
                                ui.label("Asking Ollama…");
                            }
                        });
                    });
                    ui.end_row();
                });
                ui.add_space(4.0);
                ui.label(
                    egui::RichText::new(
                        "The AI model writes the Markdown text; Tesseract supplies word positions for the searchable PDF layer. \
                         If the model is unavailable, Tesseract's text is used.",
                    )
                    .small()
                    .weak(),
                );
            });
        if self.ocr_cfg != before {
            ocr::save_config(&self.ocr_cfg);
        }
        if refresh {
            self.refresh_ocr_models(ctx);
        }
        self.settings_open = open;
    }

    fn open_scan_dialog(&mut self, ctx: &egui::Context) {
        self.scan.open = true;
        if !self.scan.listed && !self.scan.listing {
            self.refresh_devices(ctx);
            // Device discovery is slow; talk to the remembered scanner straight away.
            if !self.scan.settings.device.is_empty() && self.scan.options.is_none() {
                let dev = self.scan.settings.device.clone();
                self.select_device(ctx, dev);
            }
        }
    }

    fn refresh_devices(&mut self, ctx: &egui::Context) {
        self.scan.listing = true;
        let (tx, ctx) = (self.tx.clone(), ctx.clone());
        std::thread::spawn(move || {
            let _ = tx.send(Msg::Devices(sane::list_devices()));
            ctx.request_repaint();
        });
    }

    fn select_device(&mut self, ctx: &egui::Context, device: String) {
        self.scan.settings.device = device.clone();
        self.scan.options = None;
        self.scan.querying = true;
        self.scan.message = None;
        let source = self.scan.settings.source.clone();
        let (tx, ctx) = (self.tx.clone(), ctx.clone());
        std::thread::spawn(move || {
            // A source remembered from another device may not exist here; fall back to the default.
            let result = sane::query_options(&device, source.as_deref())
                .or_else(|_| sane::query_options(&device, None));
            let _ = tx.send(Msg::Options { device, result });
            ctx.request_repaint();
        });
    }

    fn start_scan(&mut self, ctx: &egui::Context) {
        let settings = self.scan.settings.clone();
        sane::save_settings(&settings);
        let area = self.scan.options.as_ref().and_then(|o| sane::paper_area(&settings, o));
        let handle: ScanHandle = Default::default();
        self.scan.running = Some(handle.clone());
        self.scan.message = None;
        let first = self.scan.next_page;
        // Reserve page numbers so file names stay unique across scans in this session.
        self.scan.next_page += 1000;
        let dir = std::env::temp_dir().join(format!("docscanner-{}", std::process::id()));
        let (tx, ctx) = (self.tx.clone(), ctx.clone());
        std::thread::spawn(move || {
            let result = sane::scan(&settings, area, &dir, first, &handle, |path| {
                let (tx, ctx2) = (tx.clone(), ctx.clone());
                // Detection runs off the scanner thread so the next page can feed meanwhile.
                rayon::spawn(move || {
                    let _ = tx.send(load_page(path));
                    ctx2.request_repaint();
                });
            });
            let _ = tx.send(Msg::ScanFinished(result));
            ctx.request_repaint();
        });
    }

    fn scan_window(&mut self, ctx: &egui::Context) {
        let mut open = self.scan.open;
        let mut select = None;
        let (mut do_scan, mut do_refresh) = (false, false);
        let source_before = self.scan.settings.source.clone();
        egui::Window::new("🖶 Scan")
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .default_width(420.0)
            .anchor(Align2::CENTER_TOP, Vec2::new(0.0, 80.0))
            .show(ctx, |ui| {
                let sc = &mut self.scan;
                let busy = sc.running.is_some();
                egui::Grid::new("scan-grid").num_columns(2).spacing([12.0, 8.0]).show(ui, |ui| {
                    ui.label("Scanner");
                    ui.horizontal(|ui| {
                        let current = sc
                            .devices
                            .iter()
                            .find(|d| d.name == sc.settings.device)
                            .map(|d| d.label.clone())
                            .unwrap_or_else(|| if sc.settings.device.is_empty() { "—".into() } else { sc.settings.device.clone() });
                        ui.add_enabled_ui(!busy, |ui| {
                            egui::ComboBox::from_id_salt("scan-device").width(300.0).truncate().selected_text(current).show_ui(ui, |ui| {
                                for d in &sc.devices {
                                    if ui.selectable_label(d.name == sc.settings.device, &d.label).on_hover_text(&d.name).clicked() {
                                        select = Some(d.name.clone());
                                    }
                                }
                            });
                        });
                        if sc.listing {
                            ui.spinner().on_hover_text("Searching for scanners…");
                        } else if ui.add_enabled(!busy, egui::Button::new("🔄")).on_hover_text("Search for scanners again").clicked() {
                            do_refresh = true;
                        }
                    });
                    ui.end_row();

                    if let Some(opts) = sc.options.clone() {
                        let combo = |ui: &mut egui::Ui, label: &str, id: &str, value: &mut Option<String>, list: &[String]| {
                            if list.is_empty() {
                                return;
                            }
                            ui.label(label);
                            ui.add_enabled_ui(!busy, |ui| {
                                egui::ComboBox::from_id_salt(id).selected_text(value.clone().unwrap_or_default()).show_ui(ui, |ui| {
                                    for v in list {
                                        ui.selectable_value(value, Some(v.clone()), v);
                                    }
                                });
                            });
                            ui.end_row();
                        };
                        combo(ui, "Source", "scan-source", &mut sc.settings.source, &opts.sources);
                        combo(ui, "Mode", "scan-mode", &mut sc.settings.mode, &opts.modes);
                        if opts.max_area.is_some() {
                            let papers: Vec<String> = sane::PAPER_SIZES
                                .iter()
                                .map(|(n, _, _)| n.to_string())
                                .chain([sane::FULL_AREA.to_string()])
                                .collect();
                            combo(ui, "Paper", "scan-paper", &mut sc.settings.paper, &papers);
                        }
                        if !opts.resolutions.is_empty() {
                            ui.label("Resolution");
                            ui.add_enabled_ui(!busy, |ui| {
                                egui::ComboBox::from_id_salt("scan-dpi")
                                    .selected_text(sc.settings.resolution.map(|r| format!("{r} dpi")).unwrap_or_default())
                                    .show_ui(ui, |ui| {
                                        for &r in &opts.resolutions {
                                            ui.selectable_value(&mut sc.settings.resolution, Some(r), format!("{r} dpi"));
                                        }
                                    });
                            });
                            ui.end_row();
                        }
                    }
                });

                if sc.querying {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label("Connecting to scanner…");
                    });
                } else if sc.listing && sc.devices.is_empty() && sc.settings.device.is_empty() {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label("Searching for scanners (this can take ~15 s)…");
                    });
                }

                ui.add_space(8.0);
                let feeder = sc.settings.uses_feeder();
                ui.horizontal(|ui| {
                    if busy {
                        ui.spinner();
                        ui.label(if feeder { "Scanning pages from the feeder…" } else { "Scanning…" });
                        if ui.button("✖ Cancel").clicked() {
                            if let Some(h) = &sc.running {
                                sane::cancel(h);
                            }
                        }
                    } else {
                        let ready = sc.options.is_some() && !sc.settings.device.is_empty();
                        let label = if feeder { "🖶 Scan all pages in feeder" } else { "🖶 Scan page" };
                        if ui.add_enabled(ready, egui::Button::new(egui::RichText::new(label).strong()).min_size(Vec2::new(200.0, 32.0))).clicked() {
                            do_scan = true;
                        }
                    }
                });
                if let Some((is_err, msg)) = &sc.message {
                    ui.add_space(4.0);
                    ui.colored_label(if *is_err { Color32::from_rgb(255, 110, 100) } else { Color32::from_rgb(120, 220, 140) }, msg);
                }
                ui.add_space(4.0);
                ui.label(
                    egui::RichText::new("Scanned pages are added to the page list and cleaned up with the same filters as photos.")
                        .small()
                        .weak(),
                );
            });
        self.scan.open = open;
        if let Some(d) = select {
            self.select_device(ctx, d);
        } else if self.scan.settings.source != source_before {
            // Resolutions and scan area differ between flatbed and feeder.
            let dev = self.scan.settings.device.clone();
            self.select_device(ctx, dev);
        }
        if do_refresh {
            self.refresh_devices(ctx);
        }
        if do_scan {
            self.start_scan(ctx);
        }
    }

    fn apply_list_action(&mut self, a: ListAction) {
        match a {
            ListAction::Select(i) => self.selected = i,
            ListAction::MoveUp(i) if i > 0 => {
                self.pages.swap(i, i - 1);
                self.selected = i - 1;
            }
            ListAction::MoveDown(i) if i + 1 < self.pages.len() => {
                self.pages.swap(i, i + 1);
                self.selected = i + 1;
            }
            ListAction::Remove(i) => {
                self.pages.remove(i);
                self.selected = self.selected.min(self.pages.len().saturating_sub(1));
            }
            _ => {}
        }
    }

    fn shortcuts(&mut self, ctx: &egui::Context) {
        let dropped: Vec<PathBuf> = ctx.input(|i| i.raw.dropped_files.iter().filter_map(|f| f.path.clone()).collect());
        if !dropped.is_empty() {
            self.load_files(ctx, dropped);
        }
        if ctx.input_mut(|i| i.consume_key(Modifiers::COMMAND, Key::O)) {
            self.open_dialog(ctx);
        }
        if ctx.input_mut(|i| i.consume_key(Modifiers::COMMAND, Key::N)) {
            self.open_scan_dialog(ctx);
        }
        if ctx.input_mut(|i| i.consume_key(Modifiers::COMMAND, Key::E)) {
            self.start_export(ctx, ExportKind::Pdf);
        }
        if ctx.egui_wants_keyboard_input() || self.pages.is_empty() {
            return;
        }
        let (up, down, del) = ctx.input(|i| {
            (i.key_pressed(Key::ArrowUp), i.key_pressed(Key::ArrowDown), i.key_pressed(Key::Delete))
        });
        if up {
            self.selected = self.selected.saturating_sub(1);
        }
        if down {
            self.selected = (self.selected + 1).min(self.pages.len() - 1);
        }
        if del {
            self.apply_list_action(ListAction::Remove(self.selected));
        }
    }

    fn toolbar(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        ui.horizontal(|ui| {
            ui.heading("📄 DocScanner");
            ui.separator();
            if ui.button("🖶 Scan…").on_hover_text("Scan pages from a scanner (Ctrl+N)").clicked() {
                self.open_scan_dialog(&ctx);
            }
            if ui.button("📂 Open…").on_hover_text("Add photos (Ctrl+O)").clicked() {
                self.open_dialog(&ctx);
            }
            let can_export = !self.pages.is_empty() && self.export.is_none() && self.loading == 0;
            if ui.add_enabled(can_export, egui::Button::new("💾 Export PDF…")).on_hover_text("Ctrl+E").clicked() {
                self.start_export(&ctx, ExportKind::Pdf);
            }
            ui.checkbox(&mut self.ocr_on_export, "Searchable (OCR)")
                .on_hover_text("Recognise text locally and embed it, so the PDF can be searched and copied");
            if ui.add_enabled(can_export, egui::Button::new("🖼 Images…")).on_hover_text("Export page images").clicked() {
                self.start_export(&ctx, ExportKind::Images);
            }
            if ui.add_enabled(can_export, egui::Button::new("📝 Text…")).on_hover_text("Export recognised text as Markdown").clicked() {
                self.start_export(&ctx, ExportKind::Text);
            }
            ui.separator();
            ui.label("Page:");
            egui::ComboBox::from_id_salt("page-size")
                .selected_text(format!("{:?}", self.page_size))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.page_size, PageSize::Fit, "Fit");
                    ui.selectable_value(&mut self.page_size, PageSize::A4, "A4");
                    ui.selectable_value(&mut self.page_size, PageSize::Letter, "Letter");
                });
            ui.label("DPI:");
            ui.add(egui::DragValue::new(&mut self.dpi).range(72.0..=600.0).speed(5.0));
            ui.label("JPEG quality:");
            ui.add(egui::Slider::new(&mut self.quality, 30..=100));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("⚙").on_hover_text("Text recognition settings").clicked() {
                    self.settings_note = None;
                    self.settings_open = true;
                }
            });
        });
    }

    fn page_list(&mut self, ui: &mut egui::Ui) {
        ui.add_space(4.0);
        ui.strong(format!("Pages ({})", self.pages.len()));
        ui.separator();
        let mut action = None;
        let n = self.pages.len();
        egui::ScrollArea::vertical().show(ui, |ui| {
            for (i, p) in self.pages.iter().enumerate() {
                let tex = p.preview.as_ref().unwrap_or(&p.source_tex);
                let [tw, th] = tex.size().map(|v| v as f32);
                let width = ui.available_width() - 8.0;
                let size = Vec2::new(width, (width * th / tw).min(width * 1.6));
                let selected = i == self.selected;
                let frame = egui::Frame::new()
                    .inner_margin(4.0)
                    .corner_radius(6.0)
                    .stroke(if selected { Stroke::new(2.0, ACCENT) } else { Stroke::new(1.0, Color32::from_gray(60)) });
                let resp = frame
                    .show(ui, |ui| {
                        let (rect, resp) = ui.allocate_exact_size(Vec2::new(width - 8.0, size.y), Sense::click());
                        let (img_rect, _) = fit_rect(rect, tw, th);
                        ui.painter().image(tex.id(), img_rect, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
                        ui.painter().text(
                            rect.left_top() + Vec2::new(4.0, 4.0),
                            Align2::LEFT_TOP,
                            format!("{}", i + 1),
                            FontId::proportional(16.0),
                            ACCENT,
                        );
                        if p.current_ocr().is_some() {
                            ui.painter().text(rect.left_bottom() + Vec2::new(4.0, -4.0), Align2::LEFT_BOTTOM, "🔤", FontId::proportional(14.0), ACCENT);
                        }
                        if p.in_flight || p.ocr_busy {
                            ui.painter().text(rect.right_top() + Vec2::new(-4.0, 4.0), Align2::RIGHT_TOP, "⏳", FontId::proportional(14.0), Color32::WHITE);
                        }
                        ui.horizontal(|ui| {
                            ui.add(egui::Label::new(egui::RichText::new(&p.name).small()).truncate());
                        });
                        ui.horizontal(|ui| {
                            if ui.add_enabled(i > 0, egui::Button::new("⬆").small()).on_hover_text("Move up").clicked() {
                                action = Some(ListAction::MoveUp(i));
                            }
                            if ui.add_enabled(i + 1 < n, egui::Button::new("⬇").small()).on_hover_text("Move down").clicked() {
                                action = Some(ListAction::MoveDown(i));
                            }
                            if ui.add(egui::Button::new("🗑").small()).on_hover_text("Remove page (Del)").clicked() {
                                action = Some(ListAction::Remove(i));
                            }
                        });
                        resp
                    })
                    .inner;
                if resp.clicked() {
                    action = action.or(Some(ListAction::Select(i)));
                }
                ui.add_space(6.0);
            }
            if self.loading > 0 {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(format!("Loading {}…", self.loading));
                });
            }
        });
        if let Some(a) = action {
            self.apply_list_action(a);
        }
    }

    fn page_controls(&mut self, ui: &mut egui::Ui) {
        let idx = self.selected;
        let mut apply_all = None;
        let p = &mut self.pages[idx];
        ui.horizontal_wrapped(|ui| {
            for f in [Filter::Auto, Filter::Color, Filter::Gray, Filter::Bw, Filter::Original] {
                if ui.selectable_label(p.filter == f, filter_label(f)).clicked() && p.filter != f {
                    p.filter = f;
                    p.dirty = true;
                }
            }
            if ui.small_button("apply to all").on_hover_text("Use this filter on every page").clicked() {
                apply_all = Some(p.filter);
            }
            ui.separator();
            if ui.button("⟲").on_hover_text("Rotate left").clicked() {
                p.rotate = (p.rotate + 270) % 360;
                p.dirty = true;
            }
            if ui.button("⟳").on_hover_text("Rotate right").clicked() {
                p.rotate = (p.rotate + 90) % 360;
                p.dirty = true;
            }
            ui.separator();
            if ui.add_enabled(p.detected.is_some(), egui::Button::new("🔍 Auto-detect")).on_hover_text("Restore the detected page corners").clicked() {
                p.quad = p.detected.unwrap();
                p.dirty = true;
            }
            if ui.button("⛶ Full image").on_hover_text("Don't crop").clicked() {
                p.quad = p.full_frame();
                p.dirty = true;
            }
            if ui.checkbox(&mut p.straighten, "Straighten text").changed() {
                p.dirty = true;
            }
        });
        if let Some(f) = apply_all {
            for q in self.pages.iter_mut().filter(|q| q.filter != f) {
                q.filter = f;
                q.dirty = true;
            }
        }
    }

    /// Source photo with draggable corner handles and a magnifier while dragging.
    fn corner_editor(&mut self, ui: &mut egui::Ui) {
        let p = &mut self.pages[self.selected];
        let (resp, painter) = ui.allocate_painter(ui.available_size(), Sense::hover());
        let (sw, sh) = (p.source.width() as f32, p.source.height() as f32);
        let (img_rect, scale) = fit_rect(resp.rect.shrink(12.0), sw, sh);
        let full_uv = Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0));
        painter.image(p.source_tex.id(), img_rect, full_uv, Color32::WHITE);
        let to_screen = |q: Point| img_rect.min + Vec2::new(q.x as f32 * scale, q.y as f32 * scale);

        let mut active = None;
        for i in 0..4 {
            let pos = to_screen(p.quad[i]);
            let r = ui.interact(Rect::from_center_size(pos, Vec2::splat(30.0)), ui.id().with(("corner", p.id, i)), Sense::drag());
            if r.dragged() {
                let d = r.drag_delta() / scale;
                p.quad[i].x = (p.quad[i].x + d.x as f64).clamp(0.0, sw as f64 - 1.0);
                p.quad[i].y = (p.quad[i].y + d.y as f64).clamp(0.0, sh as f64 - 1.0);
                p.dirty = true;
                active = Some(i);
            }
            if r.drag_stopped() {
                p.quad = order_corners(p.quad);
            }
            if r.hovered() || r.dragged() {
                ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
            }
        }

        // Dim everything outside the page outline.
        let pts: Vec<Pos2> = p.quad.iter().map(|&q| to_screen(q)).collect();
        let shade = Color32::from_black_alpha(110);
        let ring: Vec<Pos2> = vec![img_rect.left_top(), img_rect.right_top(), img_rect.right_bottom(), img_rect.left_bottom()];
        for i in 0..4 {
            let (a, b) = (pts[i], pts[(i + 1) % 4]);
            let (ra, rb) = (ring[i], ring[(i + 1) % 4]);
            painter.add(Shape::convex_polygon(vec![ra, rb, b, a], shade, Stroke::NONE));
        }
        painter.add(Shape::closed_line(pts.clone(), Stroke::new(2.0, ACCENT)));
        for (i, &c) in pts.iter().enumerate() {
            let big = active == Some(i);
            painter.circle(c, if big { 11.0 } else { 8.0 }, ACCENT.gamma_multiply(0.35), Stroke::new(2.0, ACCENT));
        }

        if let Some(i) = active {
            let c = p.quad[i];
            let half = Vec2::new(40.0 / sw, 40.0 / sh) * (sw.max(sh) / 1000.0).max(1.0);
            let center = pos2(c.x as f32 / sw, c.y as f32 / sh);
            let uv = Rect::from_min_max(center - half, center + half);
            let size = 150.0;
            let at_left = pts[i].x > resp.rect.center().x;
            let anchor = if at_left { resp.rect.left_top() } else { resp.rect.right_top() - Vec2::new(size, 0.0) };
            let loupe = Rect::from_min_size(anchor + Vec2::new(if at_left { 8.0 } else { -8.0 }, 8.0), Vec2::splat(size));
            painter.rect_filled(loupe, 4.0, Color32::BLACK);
            painter.image(p.source_tex.id(), loupe, uv, Color32::WHITE);
            painter.line_segment([loupe.center_top(), loupe.center_bottom()], Stroke::new(1.0, ACCENT));
            painter.line_segment([loupe.left_center(), loupe.right_center()], Stroke::new(1.0, ACCENT));
            painter.rect_stroke(loupe, 4.0, Stroke::new(2.0, ACCENT), egui::StrokeKind::Outside);
        }
    }

    fn preview_pane(&mut self, ui: &mut egui::Ui) {
        let p = &self.pages[self.selected];
        let (resp, painter) = ui.allocate_painter(ui.available_size(), Sense::hover());
        let area = resp.rect.shrink(12.0);
        match &p.preview {
            Some(tex) => {
                let [w, h] = tex.size().map(|v| v as f32);
                let (r, _) = fit_rect(area, w, h);
                painter.rect_filled(r.expand(1.0), 0.0, Color32::from_gray(40));
                painter.image(tex.id(), r, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
            }
            None => {
                painter.text(area.center(), Align2::CENTER_CENTER, "Rendering…", FontId::proportional(18.0), Color32::GRAY);
            }
        }
        if p.in_flight {
            let spinner = Rect::from_min_size(area.right_top() - Vec2::new(24.0, 0.0), Vec2::splat(20.0));
            ui.put(spinner, egui::Spinner::new());
        }
    }

    fn text_pane(&mut self, ui: &mut egui::Ui) {
        let idx = self.selected;
        let mut run = false;
        let p = &self.pages[idx];
        let current = p.current_ocr().cloned();
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            if p.ocr_busy {
                ui.spinner();
                ui.label(if self.ocr_cfg.mode == OcrMode::Best { "Recognising text (local AI model)…" } else { "Recognising text…" });
            } else {
                let label = if current.is_some() { "↻ Recognise again" } else { "🔤 Recognise text" };
                run = ui.button(label).clicked();
            }
            if let Some(o) = &current {
                if ui.button("📋 Copy").clicked() {
                    ui.ctx().copy_text(o.markdown.clone());
                }
                ui.label(egui::RichText::new(format!("by {} · {} lines", o.engine, o.lines.len())).weak());
            }
        });
        if let Some(w) = current.as_ref().and_then(|o| o.warning.as_ref()) {
            ui.colored_label(Color32::from_rgb(255, 200, 90), w);
        }
        if let Some(e) = &p.ocr_error {
            ui.colored_label(Color32::from_rgb(255, 110, 100), e);
        }
        ui.separator();
        match &current {
            Some(o) => {
                egui::ScrollArea::vertical().auto_shrink(false).show(ui, |ui| {
                    let mut text = o.markdown.as_str();
                    ui.add(egui::TextEdit::multiline(&mut text).code_editor().desired_width(f32::INFINITY));
                });
            }
            None if !p.ocr_busy => {
                ui.label(
                    egui::RichText::new(
                        "Transcribe this page on your computer (Tesseract + a local AI model via Ollama). \
                         The text appears here as Markdown and can be embedded in the PDF to make it searchable. \
                         ⚙ chooses the mode, model and languages.",
                    )
                    .weak(),
                );
            }
            None => {}
        }
        if run {
            let ctx = ui.ctx().clone();
            self.run_ocr(&ctx, idx);
        }
    }

    fn workspace(&mut self, ui: &mut egui::Ui) {
        if self.pages.is_empty() {
            let rect = ui.available_rect_before_wrap();
            let drop_hover = ui.ctx().input(|i| !i.raw.hovered_files.is_empty());
            ui.painter().rect_stroke(
                rect.shrink(30.0),
                12.0,
                Stroke::new(2.0, if drop_hover { ACCENT } else { Color32::from_gray(80) }),
                egui::StrokeKind::Inside,
            );
            ui.painter().text(
                rect.center() - Vec2::new(0.0, 14.0),
                Align2::CENTER_CENTER,
                "📷  Drop document photos here",
                FontId::proportional(26.0),
                Color32::from_gray(200),
            );
            ui.painter().text(
                rect.center() + Vec2::new(0.0, 22.0),
                Align2::CENTER_CENTER,
                "or click 🖶 Scan… / 📂 Open… · pages are detected, flattened and cleaned automatically",
                FontId::proportional(14.0),
                Color32::GRAY,
            );
            return;
        }
        self.page_controls(ui);
        ui.separator();
        let avail = ui.available_rect_before_wrap();
        let half = avail.width() / 2.0;
        let left = Rect::from_min_size(avail.min, Vec2::new(half, avail.height()));
        let right = Rect::from_min_size(avail.min + Vec2::new(half, 0.0), Vec2::new(half, avail.height()));
        ui.scope_builder(egui::UiBuilder::new().max_rect(left), |ui| {
            ui.label(egui::RichText::new("Original — drag the corners to adjust").weak());
            self.corner_editor(ui);
        });
        ui.scope_builder(egui::UiBuilder::new().max_rect(right), |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.right_tab, RightTab::Result, "Result");
                ui.selectable_value(&mut self.right_tab, RightTab::Text, "🔤 Text");
            });
            match self.right_tab {
                RightTab::Result => self.preview_pane(ui),
                RightTab::Text => self.text_pane(ui),
            }
        });
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.poll(&ctx);
        self.shortcuts(&ctx);
        self.schedule_previews(&ctx);

        egui::Panel::top("toolbar").show(ui, |ui| {
            ui.add_space(4.0);
            self.toolbar(ui);
            ui.add_space(2.0);
        });
        egui::Panel::bottom("status").show(ui, |ui| {
            ui.horizontal(|ui| {
                if let Some(e) = &self.export {
                    let done = e.done.load(Ordering::Relaxed);
                    ui.add(egui::ProgressBar::new(done as f32 / e.total as f32).desired_width(200.0).text(format!("{done}/{}", e.total)));
                }
                ui.label(&self.status);
            });
        });
        egui::Panel::left("pages").resizable(true).default_size(190.0).min_size(130.0).show(ui, |ui| self.page_list(ui));
        egui::CentralPanel::default().show(ui, |ui| self.workspace(ui));
        if self.scan.open {
            self.scan_window(&ctx);
        }
        if self.settings_open {
            self.settings_window(&ctx);
        }
    }
}

fn main() -> eframe::Result {
    let files: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("DocScanner")
            // Matches docscanner.desktop, so Wayland compositors find the icon and group windows.
            .with_app_id("docscanner")
            .with_icon(eframe::icon_data::from_png_bytes(include_bytes!("../../packaging/docscanner.png")).unwrap_or_default())
            .with_inner_size([1280.0, 820.0])
            .with_min_inner_size([800.0, 500.0])
            .with_drag_and_drop(true),
        ..Default::default()
    };
    eframe::run_native("DocScanner", options, Box::new(move |cc| Ok(Box::new(App::new(cc, files)))))
}
