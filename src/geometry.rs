//! Small 2-D geometry toolkit: points, quads, convex hulls, line fitting and homographies.

use std::ops::{Add, Mul, Sub};

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
    pub fn dot(self, o: Point) -> f64 {
        self.x * o.x + self.y * o.y
    }
    pub fn cross(self, o: Point) -> f64 {
        self.x * o.y - self.y * o.x
    }
    pub fn len(self) -> f64 {
        self.x.hypot(self.y)
    }
    pub fn dist(self, o: Point) -> f64 {
        (self - o).len()
    }
}

impl Add for Point {
    type Output = Point;
    fn add(self, o: Point) -> Point {
        Point::new(self.x + o.x, self.y + o.y)
    }
}

impl Sub for Point {
    type Output = Point;
    fn sub(self, o: Point) -> Point {
        Point::new(self.x - o.x, self.y - o.y)
    }
}

impl Mul<f64> for Point {
    type Output = Point;
    fn mul(self, s: f64) -> Point {
        Point::new(self.x * s, self.y * s)
    }
}

/// Corners ordered top-left, top-right, bottom-right, bottom-left (image coordinates, y down).
pub type Quad = [Point; 4];

/// Orders four arbitrary corners clockwise starting from the top-left one.
pub fn order_corners(mut p: [Point; 4]) -> Quad {
    let c = p.iter().fold(Point::default(), |a, &b| a + b) * 0.25;
    let angle = |q: &Point| (q.y - c.y).atan2(q.x - c.x);
    p.sort_by(|a, b| angle(a).total_cmp(&angle(b)));
    let start = (0..4)
        .min_by(|&i, &j| (p[i].x + p[i].y).total_cmp(&(p[j].x + p[j].y)))
        .unwrap();
    p.rotate_left(start);
    p
}

pub fn polygon_area(pts: &[Point]) -> f64 {
    let n = pts.len();
    (0..n).map(|i| pts[i].cross(pts[(i + 1) % n])).sum::<f64>().abs() * 0.5
}

pub fn is_convex(q: &Quad) -> bool {
    let signs: Vec<f64> = (0..4)
        .map(|i| (q[(i + 1) % 4] - q[i]).cross(q[(i + 2) % 4] - q[(i + 1) % 4]))
        .collect();
    signs.iter().all(|&s| s > 0.0) || signs.iter().all(|&s| s < 0.0)
}

/// Interior angle at each corner, in degrees.
pub fn interior_angles(q: &Quad) -> [f64; 4] {
    std::array::from_fn(|i| {
        let a = q[(i + 3) % 4] - q[i];
        let b = q[(i + 1) % 4] - q[i];
        (a.dot(b) / (a.len() * b.len())).clamp(-1.0, 1.0).acos().to_degrees()
    })
}

fn turn(o: Point, a: Point, b: Point) -> f64 {
    (a - o).cross(b - o)
}

/// Andrew's monotone chain convex hull.
pub fn convex_hull(mut pts: Vec<Point>) -> Vec<Point> {
    pts.sort_by(|a, b| a.x.total_cmp(&b.x).then(a.y.total_cmp(&b.y)));
    pts.dedup();
    if pts.len() < 3 {
        return pts;
    }
    let mut hull: Vec<Point> = Vec::with_capacity(pts.len() * 2);
    for pass in 0..2 {
        let start = hull.len();
        let iter: Box<dyn Iterator<Item = &Point>> =
            if pass == 0 { Box::new(pts.iter()) } else { Box::new(pts.iter().rev()) };
        for &p in iter {
            while hull.len() >= start + 2 && turn(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0 {
                hull.pop();
            }
            hull.push(p);
        }
        hull.pop();
    }
    hull
}

/// Removes near-collinear hull vertices (pixel staircase) until every remaining
/// vertex contributes more than `min_area` or only `max_vertices` remain.
pub fn simplify_hull(mut hull: Vec<Point>, min_area: f64, max_vertices: usize) -> Vec<Point> {
    while hull.len() > 4 {
        let n = hull.len();
        let (idx, area) = (0..n)
            .map(|i| (i, turn(hull[(i + n - 1) % n], hull[i], hull[(i + 1) % n]).abs() * 0.5))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        if area >= min_area && n <= max_vertices {
            break;
        }
        hull.remove(idx);
    }
    hull
}

/// Largest-area quadrilateral whose corners are vertices of the convex polygon `hull`.
pub fn max_area_quad(hull: &[Point]) -> Option<Quad> {
    let n = hull.len();
    if n < 4 {
        return None;
    }
    let tri = |a: usize, b: usize, c: usize| turn(hull[a], hull[b], hull[c]).abs() * 0.5;
    let mut best: Option<(f64, [usize; 4])> = None;
    for i in 0..n {
        for k in i + 2..n {
            if i == 0 && k == n - 1 {
                continue;
            }
            let j = (i + 1..k).max_by(|&a, &b| tri(i, a, k).total_cmp(&tri(i, b, k)));
            let l = (k + 1..n + i).map(|v| v % n).max_by(|&a, &b| tri(k, a, i).total_cmp(&tri(k, b, i)));
            if let (Some(j), Some(l)) = (j, l) {
                let area = tri(i, j, k) + tri(k, l, i);
                if best.is_none_or(|(b, _)| area > b) {
                    best = Some((area, [i, j, k, l]));
                }
            }
        }
    }
    best.map(|(_, idx)| order_corners(idx.map(|i| hull[i])))
}

/// A line through `origin` with unit direction `dir`.
#[derive(Clone, Copy, Debug)]
pub struct Line {
    pub origin: Point,
    pub dir: Point,
}

impl Line {
    pub fn through(a: Point, b: Point) -> Line {
        let d = b - a;
        Line { origin: a, dir: d * (1.0 / d.len()) }
    }

    pub fn distance(&self, p: Point) -> f64 {
        self.dir.cross(p - self.origin).abs()
    }

    pub fn intersect(&self, o: &Line) -> Option<Point> {
        let denom = self.dir.cross(o.dir);
        if denom.abs() < 1e-9 {
            return None;
        }
        let t = (o.origin - self.origin).cross(o.dir) / denom;
        Some(self.origin + self.dir * t)
    }
}

/// Total-least-squares line fit (principal axis of the point cloud).
pub fn fit_line(pts: &[Point]) -> Option<Line> {
    if pts.len() < 2 {
        return None;
    }
    let n = pts.len() as f64;
    let c = pts.iter().fold(Point::default(), |a, &b| a + b) * (1.0 / n);
    let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
    for p in pts {
        let d = *p - c;
        sxx += d.x * d.x;
        sxy += d.x * d.y;
        syy += d.y * d.y;
    }
    let theta = 0.5 * (2.0 * sxy).atan2(sxx - syy);
    Some(Line { origin: c, dir: Point::new(theta.cos(), theta.sin()) })
}

/// Projective transform stored row-major with h[8] == 1.
#[derive(Clone, Copy, Debug)]
pub struct Homography([f64; 9]);

impl Homography {
    /// Computes the homography mapping each `src[i]` onto `dst[i]`.
    pub fn from_points(src: &[Point; 4], dst: &[Point; 4]) -> Option<Self> {
        let mut a = [[0.0f64; 9]; 8];
        for i in 0..4 {
            let (x, y, u, v) = (src[i].x, src[i].y, dst[i].x, dst[i].y);
            a[2 * i] = [x, y, 1.0, 0.0, 0.0, 0.0, -x * u, -y * u, u];
            a[2 * i + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -x * v, -y * v, v];
        }
        // Gaussian elimination with partial pivoting on the augmented 8x9 system.
        for col in 0..8 {
            let piv = (col..8).max_by(|&r, &s| a[r][col].abs().total_cmp(&a[s][col].abs()))?;
            if a[piv][col].abs() < 1e-12 {
                return None;
            }
            a.swap(col, piv);
            for r in 0..8 {
                if r != col {
                    let f = a[r][col] / a[col][col];
                    for c in col..9 {
                        a[r][c] -= f * a[col][c];
                    }
                }
            }
        }
        let mut h = [1.0; 9];
        for i in 0..8 {
            h[i] = a[i][8] / a[i][i];
        }
        Some(Homography(h))
    }

    #[inline]
    pub fn apply(&self, x: f64, y: f64) -> Point {
        let h = &self.0;
        let w = h[6] * x + h[7] * y + h[8];
        Point::new((h[0] * x + h[1] * y + h[2]) / w, (h[3] * x + h[4] * y + h[5]) / w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orders_corners() {
        let q = order_corners([
            Point::new(10.0, 90.0),
            Point::new(90.0, 10.0),
            Point::new(5.0, 5.0),
            Point::new(95.0, 95.0),
        ]);
        assert_eq!(q[0], Point::new(5.0, 5.0));
        assert_eq!(q[1], Point::new(90.0, 10.0));
        assert_eq!(q[2], Point::new(95.0, 95.0));
        assert_eq!(q[3], Point::new(10.0, 90.0));
    }

    #[test]
    fn homography_round_trip() {
        let src = [Point::new(0.0, 0.0), Point::new(100.0, 0.0), Point::new(100.0, 50.0), Point::new(0.0, 50.0)];
        let dst = [Point::new(12.0, 7.0), Point::new(130.0, 20.0), Point::new(120.0, 90.0), Point::new(3.0, 70.0)];
        let h = Homography::from_points(&src, &dst).unwrap();
        for i in 0..4 {
            assert!(h.apply(src[i].x, src[i].y).dist(dst[i]) < 1e-6);
        }
    }

    #[test]
    fn max_quad_of_octagon_like_hull() {
        let mut pts = vec![];
        for i in 0..=100 {
            let t = i as f64;
            pts.extend([Point::new(t, 0.0), Point::new(t, 100.0), Point::new(0.0, t), Point::new(100.0, t)]);
        }
        let hull = simplify_hull(convex_hull(pts), 0.5, 48);
        let q = max_area_quad(&hull).unwrap();
        assert!((polygon_area(&q) - 10_000.0).abs() < 1e-6);
    }
}
