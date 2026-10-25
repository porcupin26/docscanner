//! Page enhancement: shadow/illumination removal, scan filters and deskew.

use clap::ValueEnum;
use image::{imageops, GrayImage, Luma, RgbImage};
use rayon::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Filter {
    /// Pick colour or greyscale depending on page content
    Auto,
    /// "Magic colour": white paper, punchy ink, preserved colours
    Color,
    /// Clean greyscale
    Gray,
    /// Black & white (adaptive threshold), smallest files
    Bw,
    /// Only crop/flatten, no enhancement
    Original,
}

pub enum PageImage {
    Color(RgbImage),
    Gray(GrayImage),
    /// Pixels are strictly 0 or 255.
    Bilevel(GrayImage),
}

impl PageImage {
    pub fn dimensions(&self) -> (u32, u32) {
        match self {
            PageImage::Color(i) => i.dimensions(),
            PageImage::Gray(i) | PageImage::Bilevel(i) => i.dimensions(),
        }
    }

    pub fn kind(&self) -> &'static str {
        match self {
            PageImage::Color(_) => "color",
            PageImage::Gray(_) => "gray",
            PageImage::Bilevel(_) => "b&w",
        }
    }
}

#[inline]
fn luma(p: &[u8]) -> f32 {
    0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32
}

/// Estimates the local paper colour on a coarse grid and divides it out, removing
/// shadows, uneven lighting and colour casts so blank paper becomes pure white.
pub fn normalize_illumination(img: &RgbImage) -> RgbImage {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let block = ((w.max(h) as f32 / 40.0).round() as usize).max(8);
    let (gw, gh) = (w.div_ceil(block), h.div_ceil(block));
    let raw = img.as_raw();

    // Paper colour per block: mean of its brightest 12% pixels.
    let mut grid: Vec<[f32; 3]> = (0..gw * gh)
        .into_par_iter()
        .map(|i| {
            let (x0, y0) = ((i % gw) * block, (i / gw) * block);
            let (x1, y1) = ((x0 + block).min(w), (y0 + block).min(h));
            let mut hist = [0u32; 256];
            for y in y0..y1 {
                for x in x0..x1 {
                    hist[luma(&raw[(y * w + x) * 3..]) as usize] += 1;
                }
            }
            let target = (((x1 - x0) * (y1 - y0)) as f32 * 0.12).max(1.0) as u32;
            let (mut acc, mut t) = (0, 255usize);
            while t > 0 {
                acc += hist[t];
                if acc >= target {
                    break;
                }
                t -= 1;
            }
            let (mut sum, mut n) = ([0f32; 3], 0f32);
            for y in y0..y1 {
                for x in x0..x1 {
                    let p = &raw[(y * w + x) * 3..(y * w + x) * 3 + 3];
                    if luma(p) as usize >= t {
                        (0..3).for_each(|c| sum[c] += p[c] as f32);
                        n += 1.0;
                    }
                }
            }
            sum.map(|s| s / n.max(1.0))
        })
        .collect();

    // Blocks dominated by ink or pictures look much darker than a neighbour: borrow the neighbour's
    // paper colour. Repeating lets it creep into large dark areas; soft shadows never trip the 25% step.
    let gl = |c: &[f32; 3]| 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2];
    for _ in 0..gw.max(gh) {
        let snapshot = grid.clone();
        let mut changed = false;
        for gy in 0..gh {
            for gx in 0..gw {
                let mut brightest = snapshot[gy * gw + gx];
                for ny in gy.saturating_sub(1)..(gy + 2).min(gh) {
                    for nx in gx.saturating_sub(1)..(gx + 2).min(gw) {
                        if gl(&snapshot[ny * gw + nx]) > gl(&brightest) {
                            brightest = snapshot[ny * gw + nx];
                        }
                    }
                }
                if gl(&snapshot[gy * gw + gx]) < 0.75 * gl(&brightest) {
                    grid[gy * gw + gx] = brightest;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    for _ in 0..2 {
        let snapshot = grid.clone();
        for gy in 0..gh {
            for gx in 0..gw {
                let (mut s, mut n) = ([0f32; 3], 0f32);
                for ny in gy.saturating_sub(1)..(gy + 2).min(gh) {
                    for nx in gx.saturating_sub(1)..(gx + 2).min(gw) {
                        (0..3).for_each(|c| s[c] += snapshot[ny * gw + nx][c]);
                        n += 1.0;
                    }
                }
                grid[gy * gw + gx] = s.map(|v| (v / n).max(30.0));
            }
        }
    }

    let mut out = vec![0u8; w * h * 3];
    out.par_chunks_mut(w * 3).enumerate().for_each(|(y, row)| {
        let fy = ((y as f32 + 0.5) / block as f32 - 0.5).clamp(0.0, (gh - 1) as f32);
        let (y0, ty) = (fy.floor() as usize, fy.fract());
        let y1 = (y0 + 1).min(gh - 1);
        for x in 0..w {
            let fx = ((x as f32 + 0.5) / block as f32 - 0.5).clamp(0.0, (gw - 1) as f32);
            let (x0, tx) = (fx.floor() as usize, fx.fract());
            let x1 = (x0 + 1).min(gw - 1);
            for c in 0..3 {
                let top = grid[y0 * gw + x0][c] * (1.0 - tx) + grid[y0 * gw + x1][c] * tx;
                let bot = grid[y1 * gw + x0][c] * (1.0 - tx) + grid[y1 * gw + x1][c] * tx;
                let bg = top * (1.0 - ty) + bot * ty;
                row[x * 3 + c] = (raw[(y * w + x) * 3 + c] as f32 / bg * 255.0).min(255.0) as u8;
            }
        }
    });
    RgbImage::from_raw(w as u32, h as u32, out).expect("buffer size matches")
}

/// Levels + gamma lookup table.
fn tone_lut(black: f32, white: f32, gamma: f32) -> [u8; 256] {
    std::array::from_fn(|v| {
        let x = ((v as f32 - black) / (white - black)).clamp(0.0, 1.0);
        (x.powf(gamma) * 255.0).round() as u8
    })
}

/// Black point: 1st-percentile luminance, capped so pages without dark content are not over-stretched.
fn black_point(img: &RgbImage) -> f32 {
    let mut hist = [0u64; 256];
    img.pixels().step_by(3).for_each(|p| hist[luma(&p.0) as usize] += 1);
    let total: u64 = hist.iter().sum();
    let mut acc = 0;
    for (v, &c) in hist.iter().enumerate() {
        acc += c;
        if acc * 100 >= total {
            return (v as f32).min(60.0);
        }
    }
    0.0
}

fn is_colourful(norm: &RgbImage) -> bool {
    let (mut colourful, mut total) = (0usize, 0usize);
    for p in norm.pixels().step_by(4) {
        let (mx, mn) = (p.0.iter().max().unwrap(), p.0.iter().min().unwrap());
        let l = luma(&p.0);
        total += 1;
        if mx - mn > 45 && (30.0..235.0).contains(&l) {
            colourful += 1;
        }
    }
    colourful * 100 > total
}

fn magic_color(norm: &RgbImage) -> RgbImage {
    let lut = tone_lut(black_point(norm), 235.0, 1.15);
    let mut out = norm.clone();
    out.pixels_mut().for_each(|p| {
        let t = p.0.map(|v| lut[v as usize] as f32);
        let l = luma(&p.0.map(|v| lut[v as usize]));
        p.0 = t.map(|v| (l + (v - l) * 1.15).clamp(0.0, 255.0) as u8);
    });
    out
}

fn to_luma(img: &RgbImage) -> GrayImage {
    GrayImage::from_fn(img.width(), img.height(), |x, y| Luma([luma(&img.get_pixel(x, y).0).round() as u8]))
}

fn enhanced_gray(norm: &RgbImage) -> GrayImage {
    let lut = tone_lut(black_point(norm), 230.0, 1.3);
    let mut g = to_luma(norm);
    g.pixels_mut().for_each(|p| p.0[0] = lut[p.0[0] as usize]);
    g
}

/// Sauvola adaptive binarisation using integral images.
fn sauvola(g: &GrayImage, k: f64) -> GrayImage {
    let (w, h) = (g.width() as usize, g.height() as usize);
    let r = ((w.max(h) / 120).clamp(7, 50)) as isize;
    let stride = w + 1;
    let mut sum = vec![0f64; stride * (h + 1)];
    let mut sq = vec![0f64; stride * (h + 1)];
    for y in 0..h {
        let (mut rs, mut rq) = (0f64, 0f64);
        for x in 0..w {
            let v = g.get_pixel(x as u32, y as u32)[0] as f64;
            rs += v;
            rq += v * v;
            sum[(y + 1) * stride + x + 1] = sum[y * stride + x + 1] + rs;
            sq[(y + 1) * stride + x + 1] = sq[y * stride + x + 1] + rq;
        }
    }
    let raw = g.as_raw();
    let mut out = vec![0u8; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let (y0, y1) = ((y as isize - r).max(0) as usize, ((y as isize + r + 1) as usize).min(h));
        for (x, o) in row.iter_mut().enumerate() {
            let (x0, x1) = ((x as isize - r).max(0) as usize, ((x as isize + r + 1) as usize).min(w));
            let n = ((x1 - x0) * (y1 - y0)) as f64;
            let rect = |t: &[f64]| t[y1 * stride + x1] - t[y0 * stride + x1] - t[y1 * stride + x0] + t[y0 * stride + x0];
            let mean = rect(&sum) / n;
            let std = (rect(&sq) / n - mean * mean).max(0.0).sqrt();
            let t = mean * (1.0 + k * (std / 128.0 - 1.0));
            // Anything this dark on a normalised page is ink, which keeps large solid areas filled.
            let v = raw[y * w + x] as f64;
            *o = if v > t && v > 110.0 { 255 } else { 0 };
        }
    });
    GrayImage::from_raw(w as u32, h as u32, out).expect("buffer size matches")
}

/// Applies the chosen scan filter. `norm` is the illumination-normalised page,
/// `original` the plain flattened page.
pub fn apply_filter(filter: Filter, norm: &RgbImage, original: RgbImage) -> PageImage {
    let filter = match filter {
        Filter::Auto if is_colourful(norm) => Filter::Color,
        Filter::Auto => Filter::Gray,
        f => f,
    };
    match filter {
        Filter::Original => PageImage::Color(original),
        Filter::Color => PageImage::Color(magic_color(norm)),
        Filter::Gray => PageImage::Gray(enhanced_gray(norm)),
        Filter::Bw | Filter::Auto => PageImage::Bilevel(sauvola(&to_luma(norm), 0.2)),
    }
}

/// Text skew angle in degrees (projection-profile method); 0 when there is no clear text structure.
pub fn estimate_skew(norm: &RgbImage) -> f64 {
    let (w, h) = norm.dimensions();
    let scale = (1000.0 / w.max(h) as f64).min(1.0);
    let small = imageops::resize(
        norm,
        ((w as f64 * scale) as u32).max(1),
        ((h as f64 * scale) as u32).max(1),
        imageops::FilterType::Triangle,
    );
    let mut pts: Vec<(f64, f64)> = small
        .enumerate_pixels()
        .filter(|(_, _, p)| luma(&p.0) < 140.0)
        .map(|(x, y, _)| (x as f64, y as f64))
        .collect();
    if pts.len() < 200 {
        return 0.0;
    }
    if pts.len() > 60_000 {
        let step = pts.len() / 60_000 + 1;
        pts = pts.into_iter().step_by(step).collect();
    }
    let span = (small.width() + small.height()) as usize;
    let profile_score = |deg: f64| -> f64 {
        let (s, c) = deg.to_radians().sin_cos();
        let mut bins = vec![0u32; span * 2 + 1];
        for &(x, y) in &pts {
            bins[(-x * s + y * c + span as f64) as usize] += 1;
        }
        bins.iter().map(|&b| (b as f64).powi(2)).sum()
    };
    let search = |from: f64, to: f64, step: f64| {
        let n = ((to - from) / step).round() as i32;
        (0..=n).map(|i| from + i as f64 * step).map(|a| (a, profile_score(a))).max_by(|a, b| a.1.total_cmp(&b.1)).unwrap()
    };
    let (coarse, _) = search(-10.0, 10.0, 0.25);
    let (angle, best) = search(coarse - 0.25, coarse + 0.25, 0.05);
    if best < profile_score(0.0) * 1.05 || angle.abs() < 0.2 {
        0.0
    } else {
        angle
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::warp::rotate_small;
    use image::Rgb;

    fn lined_page() -> RgbImage {
        RgbImage::from_fn(800, 1000, |x, y| {
            if (100..700).contains(&x) && (100..900).contains(&y) && y % 30 < 8 { Rgb([20, 20, 20]) } else { Rgb([250, 250, 250]) }
        })
    }

    #[test]
    fn deskew_recovers_rotation() {
        let page = lined_page();
        let tilted = rotate_small(&page, -3.0, Rgb([255, 255, 255]));
        let est = estimate_skew(&tilted);
        assert!((est - 3.0).abs() < 0.3, "estimated {est}");
        let fixed = rotate_small(&tilted, est, Rgb([255, 255, 255]));
        assert!(estimate_skew(&fixed).abs() < 0.3);
    }

    #[test]
    fn normalisation_removes_shadow() {
        let img = RgbImage::from_fn(600, 600, |x, _| {
            let v = (250 - x / 4) as u8; // strong left-to-right shadow
            Rgb([v, v, v])
        });
        let n = normalize_illumination(&img);
        let (l, r) = (n.get_pixel(20, 300)[0], n.get_pixel(580, 300)[0]);
        assert!(l > 235 && r > 235, "{l} {r}");
    }

    #[test]
    fn bw_is_bilevel() {
        let page = lined_page();
        let PageImage::Bilevel(g) = apply_filter(Filter::Bw, &normalize_illumination(&page), page.clone()) else { panic!() };
        assert!(g.pixels().all(|p| p[0] == 0 || p[0] == 255));
        assert_eq!(g.get_pixel(400, 122)[0], 0);
        assert_eq!(g.get_pixel(400, 110)[0], 255);
    }
}
