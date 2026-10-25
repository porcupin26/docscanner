//! Perspective correction and resampling.

use image::{Rgb, RgbImage};
use rayon::prelude::*;

use crate::geometry::{Homography, Point, Quad};

type V3 = [f64; 3];

fn cross3(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn dot3(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// True width/height ratio of a photographed rectangle, recovered with the
/// pinhole-camera method of Zhang & He ("Whiteboard scanning and image enhancement").
/// Assumes the principal point at the image centre and square pixels.
fn camera_aspect_ratio(q: &Quad, img_w: f64, img_h: f64) -> Option<f64> {
    let (u0, v0) = (img_w / 2.0, img_h / 2.0);
    let m = |p: Point| [p.x - u0, p.y - v0, 1.0];
    let (m1, m2, m3, m4) = (m(q[0]), m(q[1]), m(q[3]), m(q[2]));
    let k2 = dot3(cross3(m1, m4), m3) / dot3(cross3(m2, m4), m3);
    let k3 = dot3(cross3(m1, m4), m2) / dot3(cross3(m3, m4), m2);
    let n2: V3 = std::array::from_fn(|i| k2 * m2[i] - m1[i]);
    let n3: V3 = std::array::from_fn(|i| k3 * m3[i] - m1[i]);
    let affine = ((n2[0].powi(2) + n2[1].powi(2)) / (n3[0].powi(2) + n3[1].powi(2))).sqrt();

    let denom = n2[2] * n3[2];
    if denom.abs() < 1e-12 {
        return affine.is_finite().then_some(affine);
    }
    let f2 = -(n2[0] * n3[0] + n2[1] * n3[1]) / denom;
    let long = img_w.max(img_h);
    if !(f2 > (0.3 * long).powi(2) && f2 < (5.0 * long).powi(2)) {
        // Focal length unrecoverable (near fronto-parallel shot): affine estimate is accurate there.
        return affine.is_finite().then_some(affine);
    }
    let r2 = (n2[0].powi(2) / f2 + n2[1].powi(2) / f2 + n2[2].powi(2)) / (n3[0].powi(2) / f2 + n3[1].powi(2) / f2 + n3[2].powi(2));
    let r = r2.sqrt();
    r.is_finite().then_some(r)
}

/// Output dimensions of the flattened page, preserving the source resolution.
pub fn output_size(q: &Quad, img_w: u32, img_h: u32) -> (u32, u32) {
    let w_naive = q[0].dist(q[1]).max(q[3].dist(q[2]));
    let h_naive = q[0].dist(q[3]).max(q[1].dist(q[2]));
    let naive = w_naive / h_naive;
    let ratio = camera_aspect_ratio(q, img_w as f64, img_h as f64)
        .filter(|r| (0.6..1.67).contains(&(r / naive)))
        .unwrap_or(naive);
    let (mut w, mut h) = (w_naive, w_naive / ratio);
    if h < h_naive {
        (w, h) = (h_naive * ratio, h_naive);
    }
    (w.round().max(1.0) as u32, h.round().max(1.0) as u32)
}

#[inline]
fn sample_bilinear(src: &RgbImage, x: f64, y: f64, fill: Option<[u8; 3]>) -> [u8; 3] {
    let (w, h) = (src.width() as i64, src.height() as i64);
    if let Some(f) = fill {
        if x < -0.5 || y < -0.5 || x > w as f64 - 0.5 || y > h as f64 - 0.5 {
            return f;
        }
    }
    let (x, y) = (x.clamp(0.0, (w - 1) as f64), y.clamp(0.0, (h - 1) as f64));
    let (x0, y0) = (x.floor() as i64, y.floor() as i64);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (fx, fy) = (x - x0 as f64, y - y0 as f64);
    let p = |xx: i64, yy: i64| src.get_pixel(xx as u32, yy as u32).0;
    let (a, b, c, d) = (p(x0, y0), p(x1, y0), p(x0, y1), p(x1, y1));
    std::array::from_fn(|i| {
        let top = a[i] as f64 * (1.0 - fx) + b[i] as f64 * fx;
        let bot = c[i] as f64 * (1.0 - fx) + d[i] as f64 * fx;
        (top * (1.0 - fy) + bot * fy).round() as u8
    })
}

/// Renders `out_w x out_h` image by mapping each output pixel through `map` into `src`.
fn remap(src: &RgbImage, out_w: u32, out_h: u32, fill: Option<[u8; 3]>, map: impl Fn(f64, f64) -> Point + Sync) -> RgbImage {
    let mut buf = vec![0u8; out_w as usize * out_h as usize * 3];
    buf.par_chunks_mut(out_w as usize * 3).enumerate().for_each(|(v, row)| {
        for u in 0..out_w as usize {
            let p = map(u as f64 + 0.5, v as f64 + 0.5);
            row[u * 3..u * 3 + 3].copy_from_slice(&sample_bilinear(src, p.x - 0.5, p.y - 0.5, fill));
        }
    });
    RgbImage::from_raw(out_w, out_h, buf).expect("buffer size matches")
}

/// Flattens the quadrilateral region `q` of `src` into an upright `out_w x out_h` image.
pub fn warp_perspective(src: &RgbImage, q: &Quad, out_w: u32, out_h: u32) -> RgbImage {
    let (fw, fh) = (out_w as f64, out_h as f64);
    let rect = [Point::new(0.0, 0.0), Point::new(fw, 0.0), Point::new(fw, fh), Point::new(0.0, fh)];
    // Quad corners are pixel-centre coordinates; shift to the continuous frame used by `remap`.
    let src_q = q.map(|p| Point::new(p.x + 0.5, p.y + 0.5));
    let h = Homography::from_points(&rect, &src_q).expect("non-degenerate quad");
    remap(src, out_w, out_h, None, |x, y| h.apply(x, y))
}

/// Rotates by `degrees` about the centre, keeping the canvas size. Positive values
/// straighten lines running down to the right at that angle (as `estimate_skew` reports).
pub fn rotate_small(src: &RgbImage, degrees: f64, fill: Rgb<u8>) -> RgbImage {
    let (s, c) = degrees.to_radians().sin_cos();
    let (cx, cy) = (src.width() as f64 / 2.0, src.height() as f64 / 2.0);
    remap(src, src.width(), src.height(), Some(fill.0), |x, y| {
        let (dx, dy) = (x - cx, y - cy);
        Point::new(cx + dx * c - dy * s, cy + dx * s + dy * c)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aspect_ratio_of_tilted_a4() {
        // A4 page (1:1.414) photographed with a strong top-to-bottom tilt.
        let q = [Point::new(400.0, 300.0), Point::new(1100.0, 300.0), Point::new(1300.0, 1300.0), Point::new(200.0, 1300.0)];
        let r = camera_aspect_ratio(&q, 1500.0, 1600.0).unwrap();
        assert!(r > 0.3 && r < 3.0);
        let (w, h) = output_size(&q, 1500, 1600);
        assert!(w > 0 && h > 0);
    }

    #[test]
    fn warp_identity_quad() {
        let src = RgbImage::from_fn(40, 30, |x, y| Rgb([(x * 6) as u8, (y * 8) as u8, 7]));
        let q = [Point::new(0.0, 0.0), Point::new(39.0, 0.0), Point::new(39.0, 29.0), Point::new(0.0, 29.0)];
        let out = warp_perspective(&src, &q, 40, 30);
        let diff = |a: u8, b: u8| (a as i32 - b as i32).abs();
        assert!(diff(out.get_pixel(20, 15)[0], src.get_pixel(20, 15)[0]) <= 6);
    }
}
