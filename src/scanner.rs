//! End-to-end page pipeline: load → detect → flatten → resize → enhance → deskew.

use std::path::Path;

use anyhow::{Context, Result};
use image::{imageops, DynamicImage, ImageDecoder, ImageReader, Rgb, RgbImage};

use crate::detect::{detect_document, Detection};
use crate::enhance::{apply_filter, estimate_skew, normalize_illumination, Filter, PageImage};
use crate::geometry::{Point, Quad};
use crate::pdf::PageSize;
use crate::warp::{output_size, rotate_small, warp_perspective};

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Deskew {
    /// Only when no page edges were found (flatbed scans, screenshots)
    Auto,
    Always,
    Never,
}

#[derive(Clone, Debug)]
pub struct ScanOptions {
    pub filter: Filter,
    pub auto_crop: bool,
    pub corners: Option<Quad>,
    pub rotate: u32,
    pub deskew: Deskew,
    pub page_size: PageSize,
    pub dpi: f64,
}

pub struct ScannedPage {
    pub image: PageImage,
    pub detection: Option<Detection>,
    pub skew: f64,
}

/// Loads an image and applies its EXIF orientation (phone photos are often stored sideways).
pub fn load_image(path: &Path) -> Result<RgbImage> {
    let mut decoder = ImageReader::open(path)?.with_guessed_format()?.into_decoder()?;
    let orientation = decoder.orientation()?;
    let mut img = DynamicImage::from_decoder(decoder).with_context(|| format!("decoding {}", path.display()))?;
    img.apply_orientation(orientation);
    Ok(img.to_rgb8())
}

pub fn scan_image(img: RgbImage, opts: &ScanOptions) -> ScannedPage {
    let detection = match opts.corners {
        Some(quad) => Some(Detection { quad, score: 1.0, method: "manual" }),
        None if opts.auto_crop => detect_document(&img),
        None => None,
    };

    let mut page = match &detection {
        Some(d) => {
            let (ow, oh) = output_size(&d.quad, img.width(), img.height());
            warp_perspective(&img, &d.quad, ow, oh)
        }
        None => img,
    };
    page = match opts.rotate % 360 {
        90 => imageops::rotate90(&page),
        180 => imageops::rotate180(&page),
        270 => imageops::rotate270(&page),
        _ => page,
    };

    let f = opts.page_size.downscale_factor(page.width(), page.height(), opts.dpi);
    if f < 0.999 {
        let (w, h) = ((page.width() as f64 * f).round() as u32, (page.height() as f64 * f).round() as u32);
        page = imageops::resize(&page, w.max(1), h.max(1), imageops::FilterType::CatmullRom);
    }

    let mut norm = normalize_illumination(&page);
    let want_deskew = match opts.deskew {
        Deskew::Always => true,
        Deskew::Never => false,
        Deskew::Auto => detection.is_none(),
    };
    let skew = if want_deskew { estimate_skew(&norm) } else { 0.0 };
    if skew != 0.0 {
        let white = Rgb([255, 255, 255]);
        norm = rotate_small(&norm, skew, white);
        page = rotate_small(&page, skew, white);
    }

    ScannedPage { image: apply_filter(opts.filter, &norm, page), detection, skew }
}

/// Parses `"x1,y1 x2,y2 x3,y3 x4,y4"` (any order; separators: spaces or semicolons).
pub fn parse_corners(s: &str) -> Result<Quad> {
    let pts: Vec<Point> = s
        .split(|c: char| c == ';' || c.is_whitespace())
        .filter(|t| !t.is_empty())
        .map(|t| {
            let (x, y) = t.split_once(',').context("corner must look like x,y")?;
            Ok(Point::new(x.trim().parse()?, y.trim().parse()?))
        })
        .collect::<Result<_>>()?;
    anyhow::ensure!(pts.len() == 4, "expected 4 corners, got {}", pts.len());
    Ok(crate::geometry::order_corners([pts[0], pts[1], pts[2], pts[3]]))
}

/// Copy of the photo with the detected outline drawn on it, for troubleshooting.
pub fn debug_overlay(img: &RgbImage, quad: &Quad) -> RgbImage {
    let mut out = img.clone();
    let r = (img.width().max(img.height()) / 300).max(2) as i64;
    let (w, h) = (out.width() as i64, out.height() as i64);
    let mut dot = |c: Point, radius: i64, colour: Rgb<u8>| {
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let (x, y) = (c.x as i64 + dx, c.y as i64 + dy);
                if dx * dx + dy * dy <= radius * radius && x >= 0 && y >= 0 && x < w && y < h {
                    out.put_pixel(x as u32, y as u32, colour);
                }
            }
        }
    };
    for i in 0..4 {
        let (a, b) = (quad[i], quad[(i + 1) % 4]);
        let n = a.dist(b).ceil() as usize;
        for k in 0..=n {
            dot(a + (b - a) * (k as f64 / n.max(1) as f64), r, Rgb([0, 200, 255]));
        }
    }
    for c in quad {
        dot(*c, r * 4, Rgb([255, 60, 0]));
    }
    out
}
