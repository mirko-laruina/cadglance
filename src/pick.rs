//! Picking on a built `Scene`: nearest visible segment and nearest snap point (endpoint or
//! midpoint), scanned in parallel. Coordinates are world, relative to `Scene::origin`.

use crate::entities::P2;
use crate::geom::{NO_LAYER, Scene};

#[derive(Debug, Clone, Copy)]
pub struct Hit {
    pub batch: usize,
    pub inst: usize,
    /// First vertex of the segment.
    pub vertex: usize,
    /// Entity handle (0 = unknown).
    pub id: u32,
    pub dist: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SnapKind {
    End,
    Mid,
}

#[derive(Debug, Clone, Copy)]
pub struct Snap {
    pub p: P2,
    pub kind: SnapKind,
    pub dist: f64,
}

/// Instance transform (a, b, c, d, e, f): x' = a x + c y + e, y' = b x + d y + f.
pub fn inst_xf(scene: &Scene, i: usize) -> [f64; 6] {
    let a = &scene.instances[i * 6..i * 6 + 6];
    [a[0] as f64, a[1] as f64, a[2] as f64, a[3] as f64, a[4] as f64, a[5] as f64]
}

pub fn apply(m: &[f64; 6], x: f64, y: f64) -> P2 {
    [m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]]
}

/// Effective layer of a vertex layer inside instance `i` (layer-0 entities take the INSERT layer).
pub fn eff_layer(scene: &Scene, lay: u16, i: usize) -> u16 {
    let inherit = scene.inst_layers[i * 2];
    if lay == scene.layer0 && inherit != NO_LAYER { inherit } else { lay }
}

pub struct Picker {
    /// Local bounding box of each batch (relative to its center).
    boxes: Vec<[f32; 4]>,
}

impl Picker {
    pub fn new(scene: &Scene) -> Picker {
        let boxes = scene
            .blocks
            .iter()
            .map(|b| {
                let mut bb = [f32::MAX, f32::MAX, f32::MIN, f32::MIN];
                for k in b.vertex_start..b.vertex_start + b.vertex_count {
                    let (x, y) = (scene.positions[k * 2], scene.positions[k * 2 + 1]);
                    bb = [bb[0].min(x), bb[1].min(y), bb[2].max(x), bb[3].max(y)];
                }
                bb
            })
            .collect();
        Picker { boxes }
    }

    /// `visible(layer)` tells whether a layer is shown (NO_LAYER is always shown).
    pub fn pick(
        &self,
        scene: &Scene,
        w: P2,
        tol: f64,
        visible: &(dyn Fn(u16) -> bool + Sync),
        want_seg: bool,
    ) -> (Option<Hit>, Option<Snap>) {
        // Jobs: (batch, instance, vertex range) for instances whose box is near the point.
        const SPAN: usize = 1 << 16;
        let mut jobs = Vec::new();
        for (bi, b) in scene.blocks.iter().enumerate() {
            if b.vertex_count == 0 {
                continue;
            }
            let bb = self.boxes[bi];
            for i in b.inst_start..b.inst_start + b.inst_count {
                let m = inst_xf(scene, i);
                let det = m[0] * m[3] - m[1] * m[2];
                if det.abs() < 1e-30 {
                    continue;
                }
                let s = det.abs().sqrt();
                let (px, py) = (w[0] - m[4], w[1] - m[5]);
                let (lx, ly) = ((m[3] * px - m[2] * py) / det, (-m[1] * px + m[0] * py) / det);
                let tl = tol / s;
                if lx < bb[0] as f64 - tl || lx > bb[2] as f64 + tl || ly < bb[1] as f64 - tl || ly > bb[3] as f64 + tl {
                    continue;
                }
                let end = b.vertex_start + b.vertex_count;
                let mut k = b.vertex_start;
                while k < end {
                    let hi = (k + SPAN).min(end);
                    jobs.push((bi, i, k, hi, lx, ly, s));
                    k = hi;
                }
            }
        }
        let next = std::sync::atomic::AtomicUsize::new(0);
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(jobs.len().max(1));
        let results: Vec<(Option<Hit>, Option<Snap>)> = std::thread::scope(|sc| {
            let workers: Vec<_> = (0..threads)
                .map(|_| {
                    sc.spawn(|| {
                        let (mut seg, mut snap): (Option<Hit>, Option<Snap>) = (None, None);
                        loop {
                            let j = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            let Some(&(bi, i, k0, k1, lx, ly, s)) = jobs.get(j) else { break };
                            let p = &scene.positions;
                            let m = inst_xf(scene, i);
                            let mut k = k0;
                            while k + 1 < k1 {
                                let best = seg.map_or(tol, |h| h.dist).max(snap.map_or(tol, |h| h.dist)).min(tol);
                                let r = best / s;
                                let (x0, y0, x1, y1) =
                                    (p[k * 2] as f64, p[k * 2 + 1] as f64, p[k * 2 + 2] as f64, p[k * 2 + 3] as f64);
                                let far = (x0 < lx - r && x1 < lx - r)
                                    || (x0 > lx + r && x1 > lx + r)
                                    || (y0 < ly - r && y1 < ly - r)
                                    || (y0 > ly + r && y1 > ly + r);
                                if !far && visible(eff_layer(scene, scene.layers[k], i)) {
                                    if want_seg {
                                        let (dx, dy) = (x1 - x0, y1 - y0);
                                        let l2 = dx * dx + dy * dy;
                                        let t = if l2 > 0.0 { (((lx - x0) * dx + (ly - y0) * dy) / l2).clamp(0.0, 1.0) } else { 0.0 };
                                        let d = (x0 + t * dx - lx).hypot(y0 + t * dy - ly) * s;
                                        if d < seg.map_or(tol, |h| h.dist) {
                                            seg = Some(Hit { batch: bi, inst: i, vertex: k, id: scene.ids[k / 2], dist: d });
                                        }
                                    }
                                    for (vx, vy, kind) in
                                        [(x0, y0, SnapKind::End), (x1, y1, SnapKind::End), ((x0 + x1) / 2.0, (y0 + y1) / 2.0, SnapKind::Mid)]
                                    {
                                        // midpoints weigh a bit more, so endpoints win when close
                                        let d = (vx - lx).hypot(vy - ly) * s * if kind == SnapKind::Mid { 1.15 } else { 1.0 };
                                        if d < snap.map_or(tol, |h| h.dist) {
                                            snap = Some(Snap { p: apply(&m, vx, vy), kind, dist: d });
                                        }
                                    }
                                }
                                k += 2;
                            }
                        }
                        (seg, snap)
                    })
                })
                .collect();
            workers.into_iter().map(|w| w.join().expect("pick worker panicked")).collect()
        });
        let seg = results.iter().filter_map(|r| r.0).min_by(|a, b| a.dist.total_cmp(&b.dist));
        let snap = results.iter().filter_map(|r| r.1).min_by(|a, b| a.dist.total_cmp(&b.dist));
        (seg, snap)
    }
}

/// All segments of entity `id` in batch `bi`, instance `inst`, in world coordinates.
pub fn entity_segments(scene: &Scene, bi: usize, inst: usize, id: u32) -> Vec<[P2; 2]> {
    let b = &scene.blocks[bi];
    let m = inst_xf(scene, inst);
    let p = &scene.positions;
    (b.vertex_start..b.vertex_start + b.vertex_count)
        .step_by(2)
        .filter(|k| scene.ids[k / 2] == id)
        .map(|k| {
            [apply(&m, p[k * 2] as f64, p[k * 2 + 1] as f64), apply(&m, p[k * 2 + 2] as f64, p[k * 2 + 3] as f64)]
        })
        .collect()
}
