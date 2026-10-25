//! Automatic document boundary detection.
//!
//! Several independent segmentation strategies each propose a page outline;
//! every outline is reduced to its largest inscribed quadrilateral, snapped to
//! the real paper edges with sub-pixel line fitting, and scored by how much of
//! its perimeter is backed by detected edges. The best candidate wins.

use image::{imageops, RgbImage};

use crate::geometry::*;
use crate::imgops::*;

/// Long side (px) of the working copy used for detection.
const DETECT_SIZE: f64 = 640.0;

#[derive(Clone, Debug)]
pub struct Detection {
    pub quad: Quad,
    pub score: f64,
    pub method: &'static str,
}

pub fn detect_document(img: &RgbImage) -> Option<Detection> {
    let (w0, h0) = img.dimensions();
    let scale = (DETECT_SIZE / w0.max(h0) as f64).min(1.0);
    let small = if scale < 1.0 {
        let (sw, sh) = ((w0 as f64 * scale).round().max(1.0), (h0 as f64 * scale).round().max(1.0));
        imageops::resize(img, sw as u32, sh as u32, imageops::FilterType::Triangle)
    } else {
        img.clone()
    };
    let (w, h) = (small.width() as usize, small.height() as usize);
    if w < 32 || h < 32 {
        return None;
    }

    let mut rgb = [Plane::new(w, h), Plane::new(w, h), Plane::new(w, h)];
    for (i, px) in small.pixels().enumerate() {
        for c in 0..3 {
            rgb[c].data[i] = px[c] as f32;
        }
    }
    let rgb = rgb.map(|p| gaussian_blur(&p, 1.4));
    let edges = canny(&rgb);
    let support = dilate(&edges, 2);
    let edge_points = edges.points();

    let luma = Plane {
        w,
        h,
        data: (0..w * h).map(|i| 0.299 * rgb[0].data[i] + 0.587 * rgb[1].data[i] + 0.114 * rgb[2].data[i]).collect(),
    };
    // Paper is bright *and* neutral: its weakest channel stays high, unlike wood or coloured desks.
    let whiteness = Plane {
        w,
        h,
        data: (0..w * h).map(|i| rgb[0].data[i].min(rgb[1].data[i]).min(rgb[2].data[i])).collect(),
    };

    let mut candidates: Vec<(Vec<Point>, &'static str)> = Vec::new();
    for (r, name) in [(1, "edge-contour"), (3, "edge-contour-closed")] {
        let walls = dilate(&edges, r);
        let outside = flood_from_border(&walls.invert());
        if let Some(outline) = largest_component_outline(&outside.invert()) {
            candidates.push((outline, name));
        }
    }
    for (plane, name) in [(&luma, "brightness"), (&whiteness, "whiteness")] {
        let t = otsu(plane.data.iter().map(|v| v.clamp(0.0, 255.0) as u8)) as f32;
        let mask = Mask::from_fn(w, h, |i| plane.data[i] > t);
        let opened = dilate(&erode(&mask, 2), 2);
        if let Some(outline) = largest_component_outline(&opened) {
            candidates.push((outline, name));
        }
    }

    let diag = (w as f64).hypot(h as f64);
    let mut best: Option<Detection> = None;
    for (outline, method) in candidates {
        let hull = simplify_hull(convex_hull(outline), 1.5, 48);
        let Some(quad) = max_area_quad(&hull) else { continue };
        let quad = refine_quad(quad, &edge_points, diag);
        let Some(score) = score_quad(&quad, &support, w, h) else { continue };
        if best.as_ref().is_none_or(|b| score > b.score) {
            best = Some(Detection { quad, score, method });
        }
    }

    best.map(|mut d| {
        d.quad = d.quad.map(|p| Point::new((p.x + 0.5) / scale - 0.5, (p.y + 0.5) / scale - 0.5));
        d
    })
}

/// Snaps each side of the quad to the edge pixels running along it, then
/// recomputes the corners as intersections of the fitted lines.
fn refine_quad(quad: Quad, edge_points: &[Point], diag: f64) -> Quad {
    let mut cur = quad;
    for band in [6.0, 3.0] {
        let mut lines = [None; 4];
        for (i, line) in lines.iter_mut().enumerate() {
            let (a, b) = (cur[i], cur[(i + 1) % 4]);
            let side = Line::through(a, b);
            let len = a.dist(b);
            let near: Vec<Point> = edge_points
                .iter()
                .copied()
                .filter(|&p| {
                    let t = (p - a).dot(side.dir) / len;
                    (0.08..0.92).contains(&t) && side.distance(p) < band
                })
                .collect();
            *line = Some(side);
            if near.len() < (len * 0.25).max(12.0) as usize {
                continue;
            }
            // Fit, drop outliers, refit.
            let Some(first) = fit_line(&near) else { continue };
            let inliers: Vec<Point> = near.into_iter().filter(|&p| first.distance(p) < 1.5).collect();
            if let Some(fit) = fit_line(&inliers).filter(|_| inliers.len() >= 10) {
                if fit.dir.dot(side.dir).abs() > 6f64.to_radians().cos() {
                    *line = Some(fit);
                }
            }
        }
        let refined: Option<Vec<Point>> =
            (0..4).map(|i| lines[(i + 3) % 4].unwrap().intersect(&lines[i].unwrap())).collect();
        match refined {
            Some(r) if r.iter().zip(cur.iter()).all(|(n, o)| n.dist(*o) < 0.04 * diag) => {
                cur = order_corners([r[0], r[1], r[2], r[3]]);
            }
            _ => break,
        }
    }
    cur
}

/// Plausibility and edge-support score of a candidate page quad; `None` if implausible.
fn score_quad(q: &Quad, support: &Mask, w: usize, h: usize) -> Option<f64> {
    if !is_convex(q) {
        return None;
    }
    let area = polygon_area(q) / (w * h) as f64;
    if !(0.08..=1.05).contains(&area) || interior_angles(q).iter().any(|a| !(35.0..=145.0).contains(a)) {
        return None;
    }
    let (wf, hf) = (w as f64, h as f64);
    let on_border = |a: Point, b: Point| {
        let near = |v: f64, lim: f64| v <= 3.0 || v >= lim - 4.0;
        (near(a.x, wf) && near(b.x, wf) && (a.x - b.x).abs() < 4.0) || (near(a.y, hf) && near(b.y, hf) && (a.y - b.y).abs() < 4.0)
    };
    let mut sides = [0.0; 4];
    for (i, s) in sides.iter_mut().enumerate() {
        let (a, b) = (q[i], q[(i + 1) % 4]);
        if a.dist(b) < 0.1 * wf.min(hf) {
            return None;
        }
        if on_border(a, b) {
            // A page cut by the frame has no visible edge there; stay neutral.
            *s = 0.6;
            continue;
        }
        let n = a.dist(b).ceil() as usize;
        let hits = (0..n)
            .filter(|&k| {
                let p = a + (b - a) * (0.05 + 0.9 * k as f64 / n as f64);
                support.get_f(p.x, p.y)
            })
            .count();
        *s = hits as f64 / n as f64;
    }
    let mean = sides.iter().sum::<f64>() / 4.0;
    let min = sides.iter().copied().fold(f64::MAX, f64::min);
    if mean < 0.45 || min < 0.25 {
        return None;
    }
    Some((0.6 * mean + 0.4 * min) * (0.4 + 0.6 * area.sqrt()))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use image::Rgb;

    /// Renders a synthetic photo: textured brown desk, a white page with "text" lines in perspective.
    pub fn synthetic_photo(quad: &Quad, w: u32, h: u32) -> RgbImage {
        let rect = [Point::new(0.0, 0.0), Point::new(1.0, 0.0), Point::new(1.0, 1.414), Point::new(0.0, 1.414)];
        let to_page = Homography::from_points(quad, &rect).unwrap();
        RgbImage::from_fn(w, h, |x, y| {
            let p = to_page.apply(x as f64, y as f64);
            let inside = (0.0..1.0).contains(&p.x) && (0.0..1.414).contains(&p.y);
            if inside {
                let text = (0.1..0.9).contains(&p.x) && p.y > 0.1 && p.y < 1.3 && (p.y * 40.0).fract() < 0.35 && (p.x * 23.0).fract() < 0.8;
                // gentle shadow gradient across the page
                let shade = (235.0 - 50.0 * p.x) as u8;
                if text { Rgb([30, 30, 35]) } else { Rgb([shade, shade, shade.saturating_sub(5)]) }
            } else {
                let grain = ((x as f64 * 0.21).sin() * 8.0 + (y as f64 * 0.05).cos() * 6.0) as i32;
                Rgb([(120 + grain) as u8, (80 + grain) as u8, (50 + grain / 2) as u8])
            }
        })
    }

    #[test]
    fn finds_page_in_perspective() {
        let truth = [Point::new(310.0, 140.0), Point::new(1180.0, 260.0), Point::new(1100.0, 1450.0), Point::new(170.0, 1320.0)];
        let img = synthetic_photo(&truth, 1400, 1600);
        let d = detect_document(&img).expect("page should be detected");
        for (found, want) in d.quad.iter().zip(truth.iter()) {
            assert!(found.dist(*want) < 8.0, "{found:?} vs {want:?} ({})", d.method);
        }
    }

    #[test]
    fn rejects_blank_scene() {
        let img = RgbImage::from_pixel(800, 600, Rgb([128, 100, 90]));
        assert!(detect_document(&img).is_none());
    }
}
