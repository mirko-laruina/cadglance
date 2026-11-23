//! Curve tessellation: arcs, ellipses, bulges, NURBS, fit-point spline interpolation.

use std::f64::consts::TAU;

use crate::entities::P2;

/// Number of segments for an arc of `radius` and `sweep` so the chord error stays under `tol`.
pub fn arc_steps(radius: f64, sweep: f64, tol: f64) -> usize {
    if radius <= tol {
        return 1;
    }
    let step = 2.0 * (1.0 - tol / radius).clamp(-1.0, 1.0).acos();
    ((sweep.abs() / step.max(1e-3)).ceil() as usize).clamp(2, 256)
}

pub fn arc_points(center: P2, r: f64, start: f64, mut end: f64, tol: f64) -> Vec<P2> {
    while end <= start {
        end += TAU;
    }
    let n = arc_steps(r, end - start, tol);
    (0..=n)
        .map(|i| {
            let t = start + (end - start) * i as f64 / n as f64;
            [center[0] + r * t.cos(), center[1] + r * t.sin()]
        })
        .collect()
}

pub fn ellipse_points(center: P2, major: P2, ratio: f64, start: f64, mut end: f64, tol: f64) -> Vec<P2> {
    while end <= start {
        end += TAU;
    }
    let minor = [-major[1] * ratio, major[0] * ratio];
    let r = (major[0].hypot(major[1])).max(1e-12);
    let n = arc_steps(r, end - start, tol);
    (0..=n)
        .map(|i| {
            let t = start + (end - start) * i as f64 / n as f64;
            let (s, c) = t.sin_cos();
            [center[0] + major[0] * c + minor[0] * s, center[1] + major[1] * c + minor[1] * s]
        })
        .collect()
}

/// Points of a polyline segment with bulge (tan of a quarter of the included angle), excluding `a`.
fn bulge_points(a: P2, b: P2, bulge: f64, tol: f64, out: &mut Vec<P2>) {
    if bulge.abs() < 1e-9 {
        out.push(b);
        return;
    }
    let theta = 4.0 * bulge.atan();
    let chord = (b[0] - a[0]).hypot(b[1] - a[1]);
    if chord < 1e-12 {
        out.push(b);
        return;
    }
    let r = chord / (2.0 * (theta / 2.0).sin().abs());
    let mid = [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0];
    // distance from chord midpoint to center, signed by bulge direction
    let h = r * (theta / 2.0).cos();
    let n = [-(b[1] - a[1]) / chord, (b[0] - a[0]) / chord];
    let sign = if bulge > 0.0 { 1.0 } else { -1.0 };
    let c = [mid[0] + n[0] * h * sign, mid[1] + n[1] * h * sign];
    let a0 = (a[1] - c[1]).atan2(a[0] - c[0]);
    let steps = arc_steps(r, theta, tol);
    for i in 1..=steps {
        let t = a0 + theta * i as f64 / steps as f64;
        out.push([c[0] + r * t.cos(), c[1] + r * t.sin()]);
    }
}

/// Polyline with bulges; a closed polyline repeats its first point at the end.
pub fn bulged(pts: &[P2], bulges: &[f64], closed: bool, tol: f64) -> Vec<P2> {
    let mut out = Vec::with_capacity(pts.len());
    if pts.is_empty() {
        return out;
    }
    out.push(pts[0]);
    let n = pts.len();
    let last = if closed { n } else { n - 1 };
    for i in 0..last {
        let b = bulges.get(i).copied().unwrap_or(0.0);
        bulge_points(pts[i], pts[(i + 1) % n], b, tol, &mut out);
    }
    out
}

/// NURBS evaluation with de Boor in homogeneous coordinates.
pub fn nurbs_points(degree: usize, knots: &[f64], ctrl: &[P2], weights: &[f64]) -> Option<Vec<P2>> {
    let n = ctrl.len();
    if degree == 0 || n <= degree || knots.len() != n + degree + 1 {
        return None;
    }
    let w = |i: usize| weights.get(i).copied().unwrap_or(1.0);
    let (t0, t1) = (knots[degree], knots[n]);
    if t1 <= t0 {
        return None;
    }
    // shortcut: fixed sampling density per control point, upgrade to tolerance-based if curves look faceted
    let samples = (n * 4).clamp(8, 256);
    let mut out = Vec::with_capacity(samples + 1);
    let mut d = vec![[0.0f64; 3]; degree + 1];
    for s in 0..=samples {
        let t = t0 + (t1 - t0) * s as f64 / samples as f64;
        let mut k = degree;
        while k < n - 1 && knots[k + 1] <= t {
            k += 1;
        }
        for (j, dj) in d.iter_mut().enumerate() {
            let i = k - degree + j;
            *dj = [ctrl[i][0] * w(i), ctrl[i][1] * w(i), w(i)];
        }
        for r in 1..=degree {
            for j in (r..=degree).rev() {
                let i = k - degree + j;
                let den = knots[i + degree + 1 - r] - knots[i];
                let a = if den.abs() < 1e-15 { 0.0 } else { (t - knots[i]) / den };
                for c in 0..3 {
                    d[j][c] = (1.0 - a) * d[j - 1][c] + a * d[j][c];
                }
            }
        }
        let p = d[degree];
        if p[2].abs() > 1e-15 {
            out.push([p[0] / p[2], p[1] / p[2]]);
        }
    }
    Some(out)
}

fn find_span(n_ctrl: usize, p: usize, u: f64, knots: &[f64]) -> usize {
    if u >= knots[n_ctrl] {
        return n_ctrl - 1;
    }
    let (mut lo, mut hi) = (p, n_ctrl);
    let mut mid = (lo + hi) / 2;
    while u < knots[mid] || u >= knots[mid + 1] {
        if u < knots[mid] {
            hi = mid;
        } else {
            lo = mid;
        }
        mid = (lo + hi) / 2;
    }
    mid
}

/// Nonzero basis functions N[span-p..=span] at u (Piegl & Tiller A2.2).
fn basis_funs(span: usize, u: f64, p: usize, knots: &[f64]) -> Vec<f64> {
    let mut n = vec![0.0; p + 1];
    let mut left = vec![0.0; p + 1];
    let mut right = vec![0.0; p + 1];
    n[0] = 1.0;
    for j in 1..=p {
        left[j] = u - knots[span + 1 - j];
        right[j] = knots[span + j] - u;
        let mut saved = 0.0;
        for r in 0..j {
            let den = right[r + 1] + left[j - r];
            let tmp = if den.abs() < 1e-300 { 0.0 } else { n[r] / den };
            n[r] = saved + right[r + 1] * tmp;
            saved = left[j - r] * tmp;
        }
        n[j] = saved;
    }
    n
}

/// Solve a banded system (half bandwidth `w`) without pivoting, two right-hand sides.
/// `rows[i]` holds columns i-w..=i+w. Returns None on a near-zero pivot.
fn solve_band(mut rows: Vec<Vec<f64>>, mut rhs: Vec<P2>, w: usize) -> Option<Vec<P2>> {
    let n = rows.len();
    let at = |i: usize, j: usize| j + w - i; // column j of row i in band storage
    for k in 0..n {
        let piv = rows[k][w];
        if piv.abs() < 1e-14 {
            return None;
        }
        for i in k + 1..(k + w + 1).min(n) {
            let f = rows[i][at(i, k)] / piv;
            if f == 0.0 {
                continue;
            }
            for j in k..(k + w + 1).min(n) {
                let v = rows[k][at(k, j)];
                rows[i][at(i, j)] -= f * v;
            }
            rhs[i] = [rhs[i][0] - f * rhs[k][0], rhs[i][1] - f * rhs[k][1]];
        }
    }
    let mut x = vec![[0.0; 2]; n];
    for i in (0..n).rev() {
        let mut s = rhs[i];
        let hi = (i + w + 1).min(n);
        for (j, xj) in x.iter().enumerate().take(hi).skip(i + 1) {
            let a = rows[i][at(i, j)];
            s = [s[0] - a * xj[0], s[1] - a * xj[1]];
        }
        x[i] = [s[0] / rows[i][w], s[1] / rows[i][w]];
    }
    Some(x)
}

/// Cubic spline through fit points with end derivatives (Piegl & Tiller 9.2.2).
/// `tangents` are unit directions (zero = estimate); `knot_param` 0 chord, 1 sqrt chord, 2 uniform.
/// Returns (knots, control points), or None when the points are degenerate.
pub fn interpolate_fit(fit: &[P2], tangents: [P2; 2], knot_param: i32, closed: bool) -> Option<(Vec<f64>, Vec<P2>)> {
    let mut q: Vec<P2> = Vec::with_capacity(fit.len() + 1);
    for &p in fit {
        if q.last().is_none_or(|l: &P2| (l[0] - p[0]).hypot(l[1] - p[1]) > 1e-12) {
            q.push(p);
        }
    }
    if closed && q.len() > 2 && q[0] != q[q.len() - 1] {
        q.push(q[0]);
    }
    let n = q.len().checked_sub(1)?;
    if n < 1 {
        return None;
    }
    let dist = |a: P2, b: P2| (b[0] - a[0]).hypot(b[1] - a[1]);
    let chord: f64 = q.windows(2).map(|w| dist(w[0], w[1])).sum();
    let steps: Vec<f64> = q
        .windows(2)
        .map(|w| match knot_param {
            1 => dist(w[0], w[1]).sqrt(),
            2 => 1.0,
            _ => dist(w[0], w[1]),
        })
        .collect();
    let total: f64 = steps.iter().sum();
    if total <= 0.0 || chord <= 0.0 {
        return None;
    }
    let mut ub = vec![0.0; n + 1];
    for k in 1..=n {
        ub[k] = ub[k - 1] + steps[k - 1] / total;
    }
    ub[n] = 1.0;

    let p = 3;
    let nc = n + 3;
    let mut knots = vec![0.0; p + 1];
    for j in 0..n.saturating_sub(1) {
        knots.push((ub[j] + ub[j + 1] + ub[j + 2]) / 3.0);
    }
    knots.extend([1.0; 4]);
    debug_assert_eq!(knots.len(), nc + p + 1);

    let unit = |v: P2| {
        let l = v[0].hypot(v[1]);
        (l > 1e-12).then(|| [v[0] / l, v[1] / l])
    };
    let diff = |a: P2, b: P2, du: f64| [(b[0] - a[0]) / du, (b[1] - a[1]) / du];
    let d0 = match unit(tangents[0]) {
        Some(t) => [t[0] * chord, t[1] * chord],
        None if closed => diff(q[n - 1], q[1], ub[1] + 1.0 - ub[n - 1]),
        None => diff(q[0], q[1], ub[1]),
    };
    let dn = match unit(tangents[1]) {
        Some(t) => [t[0] * chord, t[1] * chord],
        None if closed => d0,
        None => diff(q[n - 1], q[n], 1.0 - ub[n - 1]),
    };

    let w = 4;
    let mut rows = vec![vec![0.0; 2 * w + 1]; nc];
    let mut rhs = vec![[0.0; 2]; nc];
    let set = |rows: &mut Vec<Vec<f64>>, i: usize, j: usize, v: f64| -> bool {
        if j + w < i || j > i + w {
            return false;
        }
        rows[i][j + w - i] = v;
        true
    };
    set(&mut rows, 0, 0, 1.0);
    rhs[0] = q[0];
    let s0 = knots[p + 1] / p as f64;
    set(&mut rows, 1, 0, -1.0);
    set(&mut rows, 1, 1, 1.0);
    rhs[1] = [d0[0] * s0, d0[1] * s0];
    for k in 1..n {
        let span = find_span(nc, p, ub[k], &knots);
        let b = basis_funs(span, ub[k], p, &knots);
        for (r, v) in b.iter().enumerate() {
            if !set(&mut rows, k + 1, span - p + r, *v) {
                return None;
            }
        }
        rhs[k + 1] = q[k];
    }
    let sn = (1.0 - knots[nc - 1]) / p as f64;
    set(&mut rows, nc - 2, nc - 2, -1.0);
    set(&mut rows, nc - 2, nc - 1, 1.0);
    rhs[nc - 2] = [dn[0] * sn, dn[1] * sn];
    set(&mut rows, nc - 1, nc - 1, 1.0);
    rhs[nc - 1] = q[n];

    let ctrl = solve_band(rows, rhs, w)?;
    Some((knots, ctrl))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn bulge_half_circle() {
        // bulge 1 = 180 degrees; midpoint of the arc from (0,0) to (2,0) counterclockwise is (1,-1)
        let p = bulged(&[[0.0, 0.0], [2.0, 0.0]], &[1.0], false, 1e-4);
        let last = p[p.len() - 1];
        assert!((last[0] - 2.0).abs() < 1e-9 && last[1].abs() < 1e-9);
        let mid = p[p.len() / 2];
        assert!((mid[0] - 1.0).abs() < 1e-2 && (mid[1] + 1.0).abs() < 1e-2, "{mid:?}");
    }

    #[test]
    fn nurbs_line_and_quarter_circle() {
        let p = nurbs_points(1, &[0.0, 0.0, 1.0, 1.0], &[[0.0, 0.0], [2.0, 2.0]], &[]).unwrap();
        assert!(p.iter().all(|q| (q[0] - q[1]).abs() < 1e-12));
        let w = 0.5f64.sqrt();
        let p = nurbs_points(2, &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], &[1.0, w, 1.0])
            .unwrap();
        assert!(p.iter().all(|q| (q[0].hypot(q[1]) - 1.0).abs() < 1e-9));
    }

    #[test]
    fn arc_steps_bounds() {
        assert_eq!(arc_steps(0.0, PI, 0.1), 1);
        assert!(arc_steps(1000.0, TAU, 0.01) <= 256);
    }

    #[test]
    fn fit_interpolation_passes_through_points() {
        let fit = [[0.0, 0.0], [1.0, 2.0], [3.0, 3.0], [4.0, 0.0], [6.0, 1.0]];
        let (knots, ctrl) = interpolate_fit(&fit, [[0.0; 2]; 2], 0, false).unwrap();
        // evaluate exactly at the fit parameters: endpoints are interpolated
        assert_eq!(ctrl[0], fit[0]);
        assert_eq!(ctrl[ctrl.len() - 1], fit[4]);
        // every fit point lies on the sampled curve (within sampling resolution)
        let pts = nurbs_points(3, &knots, &ctrl, &[]).unwrap();
        for f in &fit {
            let d = pts.iter().map(|p| (p[0] - f[0]).hypot(p[1] - f[1])).fold(f64::MAX, f64::min);
            assert!(d < 0.15, "fit point {f:?} is {d} from the curve");
        }
    }

    #[test]
    fn fit_interpolation_two_points_is_a_line() {
        let (knots, ctrl) = interpolate_fit(&[[0.0, 0.0], [3.0, 0.0]], [[0.0; 2]; 2], 0, false).unwrap();
        let pts = nurbs_points(3, &knots, &ctrl, &[]).unwrap();
        assert!(pts.iter().all(|p| p[1].abs() < 1e-9));
    }
}
