//! Low-level raster operations on single-channel float planes and binary masks.

use std::collections::VecDeque;

use crate::geometry::Point;

#[derive(Clone)]
pub struct Plane {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f32>,
}

impl Plane {
    pub fn new(w: usize, h: usize) -> Self {
        Self { w, h, data: vec![0.0; w * h] }
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.w + x]
    }

    pub fn map(&self, f: impl Fn(f32) -> f32) -> Plane {
        Plane { w: self.w, h: self.h, data: self.data.iter().map(|&v| f(v)).collect() }
    }
}

/// Separable Gaussian blur with clamped borders.
pub fn gaussian_blur(p: &Plane, sigma: f32) -> Plane {
    let r = (sigma * 3.0).ceil() as isize;
    let mut k: Vec<f32> = (-r..=r).map(|i| (-(i * i) as f32 / (2.0 * sigma * sigma)).exp()).collect();
    let sum: f32 = k.iter().sum();
    k.iter_mut().for_each(|v| *v /= sum);

    let (w, h) = (p.w as isize, p.h as isize);
    let mut tmp = Plane::new(p.w, p.h);
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for (i, kv) in k.iter().enumerate() {
                let sx = (x + i as isize - r).clamp(0, w - 1);
                acc += kv * p.data[(y * w + sx) as usize];
            }
            tmp.data[(y * w + x) as usize] = acc;
        }
    }
    let mut out = Plane::new(p.w, p.h);
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for (i, kv) in k.iter().enumerate() {
                let sy = (y + i as isize - r).clamp(0, h - 1);
                acc += kv * tmp.data[(sy * w + x) as usize];
            }
            out.data[(y * w + x) as usize] = acc;
        }
    }
    out
}

#[derive(Clone)]
pub struct Mask {
    pub w: usize,
    pub h: usize,
    pub data: Vec<bool>,
}

impl Mask {
    pub fn new(w: usize, h: usize) -> Self {
        Self { w, h, data: vec![false; w * h] }
    }

    pub fn from_fn(w: usize, h: usize, f: impl Fn(usize) -> bool) -> Self {
        Self { w, h, data: (0..w * h).map(f).collect() }
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> bool {
        self.data[y * self.w + x]
    }

    /// Like `get`, but out-of-bounds coordinates read as `false`.
    pub fn get_f(&self, x: f64, y: f64) -> bool {
        let (xi, yi) = (x.round(), y.round());
        xi >= 0.0 && yi >= 0.0 && (xi as usize) < self.w && (yi as usize) < self.h && self.get(xi as usize, yi as usize)
    }

    pub fn invert(&self) -> Mask {
        Mask { w: self.w, h: self.h, data: self.data.iter().map(|v| !v).collect() }
    }

    pub fn points(&self) -> Vec<Point> {
        (0..self.data.len())
            .filter(|&i| self.data[i])
            .map(|i| Point::new((i % self.w) as f64, (i / self.w) as f64))
            .collect()
    }
}

/// Multi-channel Canny edge detector: at each pixel the channel with the strongest
/// gradient wins, so paper/background boundaries are found even when they only
/// differ in hue. Thresholds adapt to the image's gradient statistics.
pub fn canny(channels: &[Plane]) -> Mask {
    let (w, h) = (channels[0].w, channels[0].h);
    let mut mag = vec![0f32; w * h];
    let mut dir = vec![0u8; w * h];
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let (mut bm, mut bgx, mut bgy) = (0f32, 0f32, 0f32);
            for p in channels {
                let g = |dx: isize, dy: isize| p.get((x as isize + dx) as usize, (y as isize + dy) as usize);
                let gx = g(1, -1) + 2.0 * g(1, 0) + g(1, 1) - g(-1, -1) - 2.0 * g(-1, 0) - g(-1, 1);
                let gy = g(-1, 1) + 2.0 * g(0, 1) + g(1, 1) - g(-1, -1) - 2.0 * g(0, -1) - g(1, -1);
                let m = gx.hypot(gy);
                if m > bm {
                    (bm, bgx, bgy) = (m, gx, gy);
                }
            }
            let i = y * w + x;
            mag[i] = bm;
            let a = bgy.atan2(bgx).to_degrees().rem_euclid(180.0);
            dir[i] = match a {
                a if !(22.5..157.5).contains(&a) => 0,
                a if a < 67.5 => 1,
                a if a < 112.5 => 2,
                _ => 3,
            };
        }
    }

    // Non-maximum suppression along the gradient direction.
    let mut nms = vec![0f32; w * h];
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let i = y * w + x;
            let (a, b) = match dir[i] {
                0 => (i - 1, i + 1),
                1 => (i - w - 1, i + w + 1),
                2 => (i - w, i + w),
                _ => (i - w + 1, i + w - 1),
            };
            if mag[i] >= mag[a] && mag[i] >= mag[b] {
                nms[i] = mag[i];
            }
        }
    }

    let mut ridge: Vec<f32> = nms.iter().copied().filter(|&v| v > 0.0).collect();
    let high = if ridge.is_empty() {
        f32::MAX
    } else {
        let k = (ridge.len() * 80 / 100).min(ridge.len() - 1);
        *ridge.select_nth_unstable_by(k, |a, b| a.total_cmp(b)).1
    }
    .max(30.0);
    let low = high * 0.4;

    // Hysteresis: grow strong edges through weak ones.
    let mut out = Mask::new(w, h);
    let mut queue: VecDeque<usize> = (0..w * h).filter(|&i| nms[i] >= high).collect();
    queue.iter().for_each(|&i| out.data[i] = true);
    while let Some(i) = queue.pop_front() {
        let (x, y) = ((i % w) as isize, (i / w) as isize);
        for dy in -1..=1 {
            for dx in -1..=1 {
                let (nx, ny) = (x + dx, y + dy);
                if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize {
                    continue;
                }
                let j = ny as usize * w + nx as usize;
                if !out.data[j] && nms[j] >= low {
                    out.data[j] = true;
                    queue.push_back(j);
                }
            }
        }
    }
    out
}

/// Otsu's threshold for 8-bit samples.
pub fn otsu(values: impl Iterator<Item = u8>) -> u8 {
    let mut hist = [0u64; 256];
    values.for_each(|v| hist[v as usize] += 1);
    let total: u64 = hist.iter().sum();
    let sum_all: f64 = hist.iter().enumerate().map(|(i, &c)| i as f64 * c as f64).sum();
    let (mut w0, mut sum0, mut best, mut best_t) = (0u64, 0f64, -1f64, 128u8);
    for t in 0..256 {
        w0 += hist[t];
        if w0 == 0 || w0 == total {
            continue;
        }
        sum0 += t as f64 * hist[t] as f64;
        let w1 = total - w0;
        let (m0, m1) = (sum0 / w0 as f64, (sum_all - sum0) / w1 as f64);
        let between = w0 as f64 * w1 as f64 * (m0 - m1).powi(2);
        if between > best {
            best = between;
            best_t = t as u8;
        }
    }
    best_t
}

/// Square-window morphology. `all == false` dilates, `all == true` erodes.
fn morph(m: &Mask, r: usize, all: bool) -> Mask {
    let pass = |src: &[bool], len: usize, stride: usize, lines: usize, line_stride: usize| -> Vec<bool> {
        let mut out = vec![false; src.len()];
        let mut prefix = vec![0u32; len + 1];
        for l in 0..lines {
            let base = l * line_stride;
            for i in 0..len {
                prefix[i + 1] = prefix[i] + src[base + i * stride] as u32;
            }
            for i in 0..len {
                let (a, b) = (i.saturating_sub(r), (i + r + 1).min(len));
                let n = prefix[b] - prefix[a];
                out[base + i * stride] = if all { n as usize == b - a } else { n > 0 };
            }
        }
        out
    };
    let h_pass = pass(&m.data, m.w, 1, m.h, m.w);
    let v_pass = pass(&h_pass, m.h, m.w, m.w, 1);
    Mask { w: m.w, h: m.h, data: v_pass }
}

pub fn dilate(m: &Mask, r: usize) -> Mask {
    morph(m, r, false)
}

pub fn erode(m: &Mask, r: usize) -> Mask {
    morph(m, r, true)
}

/// Pixels reachable from the image border through `passable` pixels (4-connected).
pub fn flood_from_border(passable: &Mask) -> Mask {
    let (w, h) = (passable.w, passable.h);
    let mut reached = Mask::new(w, h);
    let mut queue = VecDeque::new();
    let seed = |i: usize, reached: &mut Mask, queue: &mut VecDeque<usize>| {
        if passable.data[i] && !reached.data[i] {
            reached.data[i] = true;
            queue.push_back(i);
        }
    };
    for x in 0..w {
        seed(x, &mut reached, &mut queue);
        seed((h - 1) * w + x, &mut reached, &mut queue);
    }
    for y in 0..h {
        seed(y * w, &mut reached, &mut queue);
        seed(y * w + w - 1, &mut reached, &mut queue);
    }
    while let Some(i) = queue.pop_front() {
        let (x, y) = (i % w, i / w);
        let mut push = |j: usize| seed(j, &mut reached, &mut queue);
        if x > 0 {
            push(i - 1);
        }
        if x + 1 < w {
            push(i + 1);
        }
        if y > 0 {
            push(i - w);
        }
        if y + 1 < h {
            push(i + w);
        }
    }
    reached
}

/// Outline (left/right extreme of each row) of the largest 4-connected component.
pub fn largest_component_outline(m: &Mask) -> Option<Vec<Point>> {
    let (w, h) = (m.w, m.h);
    let mut labels = vec![0u32; w * h];
    let (mut best_label, mut best_size, mut next) = (0u32, 0usize, 1u32);
    let mut queue = VecDeque::new();
    for start in 0..w * h {
        if !m.data[start] || labels[start] != 0 {
            continue;
        }
        labels[start] = next;
        queue.push_back(start);
        let mut size = 0;
        while let Some(i) = queue.pop_front() {
            size += 1;
            let (x, y) = (i % w, i / w);
            let neighbours = [
                (x > 0).then(|| i - 1),
                (x + 1 < w).then(|| i + 1),
                (y > 0).then(|| i - w),
                (y + 1 < h).then(|| i + w),
            ];
            for j in neighbours.into_iter().flatten() {
                if m.data[j] && labels[j] == 0 {
                    labels[j] = next;
                    queue.push_back(j);
                }
            }
        }
        if size > best_size {
            (best_size, best_label) = (size, next);
        }
        next += 1;
    }
    if best_size == 0 {
        return None;
    }
    let mut outline = Vec::new();
    for y in 0..h {
        let row = &labels[y * w..(y + 1) * w];
        if let (Some(a), Some(b)) = (row.iter().position(|&l| l == best_label), row.iter().rposition(|&l| l == best_label)) {
            outline.push(Point::new(a as f64, y as f64));
            outline.push(Point::new(b as f64, y as f64));
        }
    }
    Some(outline)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn otsu_splits_bimodal() {
        let vals = (0..1000).map(|i| if i % 2 == 0 { 40u8 } else { 200u8 });
        let t = otsu(vals);
        assert!((40..200).contains(&t));
    }

    #[test]
    fn morphology_opening_removes_specks() {
        let mut m = Mask::new(20, 20);
        m.data[5 * 20 + 5] = true;
        for y in 10..18 {
            for x in 10..18 {
                m.data[y * 20 + x] = true;
            }
        }
        let opened = dilate(&erode(&m, 1), 1);
        assert!(!opened.get(5, 5));
        assert!(opened.get(14, 14));
    }
}
