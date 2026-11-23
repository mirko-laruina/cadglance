//! Hatch boundaries to loops, solid fill fans, and pattern lines clipped by even-odd parity.

use crate::curves::{arc_points, bulged, ellipse_points, nurbs_points};
use crate::entities::{HatchPath, HatchSeg, P2, PatternLine};

/// Each boundary path as a closed loop of points (first point not repeated at the end).
pub fn loops(paths: &[HatchPath], tol: f64) -> Vec<Vec<P2>> {
    let mut out = Vec::with_capacity(paths.len());
    for path in paths {
        let mut pts: Vec<P2> = match path {
            HatchPath::Polyline { pts, bulges, .. } => bulged(pts, bulges, true, tol),
            HatchPath::Segments(segs) => {
                let mut v: Vec<P2> = Vec::new();
                for seg in segs {
                    let piece = match seg {
                        HatchSeg::Line(a, b) => vec![*a, *b],
                        HatchSeg::Arc { center, radius, start, end, ccw } => {
                            if *ccw {
                                arc_points(*center, *radius, *start, *end, tol)
                            } else {
                                // clockwise edge: runs from -start to -end; same points as the
                                // counterclockwise arc from -end to -start, reversed
                                let mut p = arc_points(*center, *radius, -*end, -*start, tol);
                                p.reverse();
                                p
                            }
                        }
                        HatchSeg::Ellipse { center, major, ratio, start, end, ccw } => {
                            let mut p = ellipse_points(*center, *major, *ratio, *start, *end, tol);
                            if !*ccw {
                                p.reverse();
                            }
                            p
                        }
                        HatchSeg::Spline { degree, knots, ctrl, weights } => {
                            nurbs_points(*degree as usize, knots, ctrl, weights).unwrap_or_else(|| ctrl.clone())
                        }
                    };
                    v.extend(piece);
                }
                v
            }
        };
        while pts.len() > 1 && pts.first() == pts.last() {
            pts.pop();
        }
        if pts.len() >= 3 {
            out.push(pts);
        }
    }
    out
}

/// Triangle fans of all loops, for even-odd stencil filling (fans may overlap; parity decides).
pub fn fill_fans(loops: &[Vec<P2>], out: &mut Vec<P2>) {
    for l in loops {
        for i in 1..l.len() - 1 {
            out.push(l[0]);
            out.push(l[i]);
            out.push(l[i + 1]);
        }
    }
}

/// Pattern line segments inside the loops (even-odd). Stops at `budget` segments.
pub fn pattern_segments(loops: &[Vec<P2>], pattern: &[PatternLine], budget: usize, out: &mut Vec<P2>) {
    let start_len = out.len();
    for pl in pattern {
        let (s, c) = pl.angle.sin_cos();
        // to line frame: u along the lines, v across
        let to = |p: P2| [p[0] * c + p[1] * s, -p[0] * s + p[1] * c];
        let back = |u: f64, v: f64| [u * c - v * s, u * s + v * c];
        let base = to(pl.base);
        let off = [pl.offset[0] * c + pl.offset[1] * s, -pl.offset[0] * s + pl.offset[1] * c];
        if off[1].abs() < 1e-12 {
            continue;
        }
        let edges: Vec<(P2, P2)> = loops
            .iter()
            .flat_map(|l| {
                let lt: Vec<P2> = l.iter().map(|p| to(*p)).collect();
                (0..lt.len()).map(move |i| (lt[i], lt[(i + 1) % lt.len()])).collect::<Vec<_>>()
            })
            .collect();
        let (vmin, vmax) = edges.iter().fold((f64::MAX, f64::MIN), |(a, b), (p, _)| (a.min(p[1]), b.max(p[1])));
        let (k0, k1) = {
            let a = (vmin - base[1]) / off[1];
            let b = (vmax - base[1]) / off[1];
            (a.min(b).ceil() as i64, a.max(b).floor() as i64)
        };
        if k1 < k0 || (k1 - k0) as usize > budget {
            continue; // too dense for the budget
        }
        let period: f64 = pl.dashes.iter().map(|d| d.abs()).sum();
        let mut xs: Vec<f64> = Vec::new();
        for k in k0..=k1 {
            let v = base[1] + k as f64 * off[1];
            let origin = base[0] + k as f64 * off[0];
            xs.clear();
            for (a, b) in &edges {
                if (a[1] <= v) != (b[1] <= v) {
                    xs.push(a[0] + (v - a[1]) * (b[0] - a[0]) / (b[1] - a[1]));
                }
            }
            xs.sort_by(|a, b| a.total_cmp(b));
            for pair in xs.chunks_exact(2) {
                let (x0, x1) = (pair[0], pair[1]);
                if pl.dashes.is_empty() || period <= 1e-12 {
                    out.push(back(x0, v));
                    out.push(back(x1, v));
                } else {
                    let mut t = origin + ((x0 - origin) / period).floor() * period;
                    'dash: while t < x1 {
                        for d in &pl.dashes {
                            let len = d.abs();
                            if *d >= 0.0 {
                                // dots (0) get a tiny length so they stay visible
                                let e = if len == 0.0 { t + period * 1e-3 } else { t + len };
                                let (a, b) = (t.max(x0), e.min(x1));
                                if b > a {
                                    out.push(back(a, v));
                                    out.push(back(b, v));
                                }
                            }
                            t += len;
                            if t >= x1 || out.len() - start_len >= budget * 2 {
                                break 'dash;
                            }
                        }
                    }
                }
                if out.len() - start_len >= budget * 2 {
                    return;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(a: f64, b: f64) -> Vec<P2> {
        vec![[a, a], [b, a], [b, b], [a, b]]
    }

    #[test]
    fn horizontal_lines_respect_hole() {
        // outer 0..10, hole 4..6; lines every 1 at y = 0.5, 1.5, ...
        let l = vec![square(0.0, 10.0), square(4.0, 6.0)];
        let pat = [PatternLine { angle: 0.0, base: [0.0, 0.5], offset: [0.0, 1.0], dashes: vec![] }];
        let mut out = Vec::new();
        pattern_segments(&l, &pat, 10_000, &mut out);
        // y = 4.5 and 5.5 cross the hole: 2 segments each; the other 8 lines: 1 segment
        assert_eq!(out.len() / 2, 8 + 2 * 2);
        let total: f64 = out.chunks(2).map(|s| (s[1][0] - s[0][0]).abs()).sum();
        assert!((total - (10.0 * 10.0 - 2.0 * 2.0)).abs() < 1e-9);
    }

    #[test]
    fn dashes_and_fans() {
        let l = vec![square(0.0, 10.0)];
        let pat = [PatternLine { angle: 0.0, base: [0.0, 5.0], offset: [0.0, 100.0], dashes: vec![1.0, -1.0] }];
        let mut out = Vec::new();
        pattern_segments(&l, &pat, 10_000, &mut out);
        assert_eq!(out.len() / 2, 5);
        let mut tri = Vec::new();
        fill_fans(&l, &mut tri);
        assert_eq!(tri.len(), 6);
    }
}
