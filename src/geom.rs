//! Model space to GPU-ready data: per-block lines, fill triangles and texts, plus instance
//! transforms and colors. Colors that depend on the INSERT (ByBlock, layer 0) are resolved per
//! instance on the GPU: the vertex alpha channel tells which instance color to use.

use std::collections::HashMap;
use std::f64::consts::TAU;

use crate::aci::aci_rgb;
use crate::curves::{arc_points, bulged, ellipse_points, interpolate_fit, nurbs_points};
use crate::drawing::Drawing;
use crate::entities::{Entity, Kind, P2, Text};
use crate::hatch;
use crate::objects::Color;
use crate::text::{clean_text, json_str, mtext_paragraphs};

/// 2D affine transform: x' = a x + c y + e, y' = b x + d y + f.
pub type Xf = [f64; 6];
const IDENTITY: Xf = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
const MIRROR_X: Xf = [-1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
/// Instance colors: RGBA used for ByBlock entities, RGBA used for ByLayer entities on layer 0.
type InstColors = [[u8; 4]; 2];
type InstLayers = [u16; 2];
/// "No layer" marker in layer indices.
pub const NO_LAYER: u16 = u16::MAX;

/// Vertex color alpha: 255 = fixed color, these two = take the instance color.
const MODE_BYBLOCK: u8 = 0;
const MODE_LAYER0: u8 = 1;
/// Pattern segments allowed per hatch before the pattern is skipped as too dense.
const PATTERN_BUDGET: usize = 200_000;

fn mul(m: &Xf, n: &Xf) -> Xf {
    [
        m[0] * n[0] + m[2] * n[1],
        m[1] * n[0] + m[3] * n[1],
        m[0] * n[2] + m[2] * n[3],
        m[1] * n[2] + m[3] * n[3],
        m[0] * n[4] + m[2] * n[5] + m[4],
        m[1] * n[4] + m[3] * n[5] + m[5],
    ]
}

fn apply(m: &Xf, p: P2) -> P2 {
    [m[0] * p[0] + m[2] * p[1] + m[4], m[1] * p[0] + m[3] * p[1] + m[5]]
}

/// One block batch: ranges into the line, fill and instance arrays.
#[derive(Debug, Clone)]
pub struct BlockDraw {
    pub vertex_start: usize,
    pub vertex_count: usize,
    pub fill_start: usize,
    pub fill_count: usize,
    pub inst_start: usize,
    pub inst_count: usize,
}

pub struct Scene {
    /// Line vertices (x, y), 2 per segment, relative to each block's own center.
    pub positions: Vec<f32>,
    /// RGBA per line vertex (alpha = color mode, see MODE_*).
    pub colors: Vec<u8>,
    /// Fill triangle vertices (x, y), relative to each block's center.
    pub fill_positions: Vec<f32>,
    pub fill_colors: Vec<u8>,
    /// 6 floats per instance (a, b, c, d, e, f), translation relative to `origin`.
    pub instances: Vec<f32>,
    /// 8 bytes per instance: RGBA for ByBlock, RGBA for layer 0.
    pub inst_colors: Vec<u8>,
    /// Layer index per line vertex and per fill vertex (position in `layers_json`).
    pub layers: Vec<u16>,
    pub fill_layers: Vec<u16>,
    /// 2 per instance: layer used by layer-0 entities inside, top-level INSERT layer (or NO_LAYER).
    pub inst_layers: Vec<u16>,
    pub layers_json: String,
    /// Same layers as `layers_json`: name, color, initially visible.
    pub layer_list: Vec<(String, [u8; 3], bool)>,
    pub layer0: u16,
    pub blocks: Vec<BlockDraw>,
    /// JSON objects, one per text, in block-local coordinates relative to the block center.
    pub texts: Vec<TextItem>,
    /// Entity handle per line segment (0 = unknown), for picking.
    pub ids: Vec<u32>,
    /// Center of each batch in block coordinates (vertices are relative to it).
    pub centers: Vec<P2>,
    pub origin: P2,
    /// World bounding box relative to `origin`: min x, min y, max x, max y.
    pub bbox: [f64; 4],
    pub segments_drawn: u64,
    /// Text styles as JSON members: "handle": {wf, ob, font}.
    pub styles_json: String,
    /// Text styles: handle -> (width factor, oblique angle).
    pub styles: HashMap<u64, (f64, f64)>,
    pub triangles_drawn: u64,
}

/// A text to draw. In `Scene::texts`, points are relative to the batch center.
#[derive(Debug, Clone)]
pub struct TextItem {
    /// Batch (index into `Scene::blocks`) the text belongs to.
    pub batch: usize,
    /// 0 = TEXT/ATTRIB, 1 = MTEXT
    pub kind: u8,
    pub pos: P2,
    pub align: P2,
    pub height: f64,
    pub rotation: f64,
    pub width_factor: f64,
    pub oblique: f64,
    pub halign: i16,
    pub valign: i16,
    pub attachment: i16,
    pub width: f64,
    pub color: [u8; 4],
    pub id: u32,
    pub layer: u16,
    pub style: u64,
    /// MTEXT paragraph alignments (0 default, 1 left, 2 center, 3 right).
    pub para: Vec<u8>,
    pub value: String,
}

struct Kid {
    block: u64,
    xfs: Vec<Xf>,
    layer: u64,
    color: Color,
}

#[derive(Default)]
struct Content {
    /// Handle of the entity being added (stored per segment and per text, for picking).
    cur: u32,
    pts: Vec<P2>,
    cols: Vec<[u8; 4]>,
    ids: Vec<u32>,
    /// Layer index of the entity being added, and per line vertex / fill vertex.
    lay: u16,
    layers: Vec<u16>,
    fill_layers: Vec<u16>,
    fill: Vec<P2>,
    fill_cols: Vec<[u8; 4]>,
    texts: Vec<TextItem>,
    kids: Vec<Kid>,
}

impl Content {
    /// Append another chunk of the same block (chunks are merged in entity order).
    fn append(&mut self, mut o: Content) {
        self.pts.append(&mut o.pts);
        self.cols.append(&mut o.cols);
        self.ids.append(&mut o.ids);
        self.layers.append(&mut o.layers);
        self.fill.append(&mut o.fill);
        self.fill_cols.append(&mut o.fill_cols);
        self.fill_layers.append(&mut o.fill_layers);
        self.texts.append(&mut o.texts);
        self.kids.append(&mut o.kids);
    }

    fn line(&mut self, a: P2, b: P2, c: [u8; 4]) {
        self.pts.push(a);
        self.pts.push(b);
        self.cols.push(c);
        self.cols.push(c);
        self.ids.push(self.cur);
    }

    fn polyline(&mut self, pts: &[P2], closed: bool, c: [u8; 4]) {
        for w in pts.windows(2) {
            self.line(w[0], w[1], c);
        }
        if closed && pts.len() > 2 {
            self.line(pts[pts.len() - 1], pts[0], c);
        }
    }

    fn text(&mut self, t: &Text, value: String, c: [u8; 4]) {
        if value.trim().is_empty() {
            return;
        }
        self.texts.push(TextItem {
            batch: 0,
            kind: 0,
            pos: t.insert,
            align: t.align,
            height: t.height,
            rotation: t.rotation,
            width_factor: t.width_factor,
            oblique: t.oblique,
            halign: t.halign,
            valign: t.valign,
            attachment: 0,
            width: 0.0,
            color: c,
            id: self.cur,
            layer: self.lay,
            style: t.style,
            para: Vec::new(),
            value,
        });
    }
}

fn mirror_x_if(extrusion: &[f64; 3], p: P2) -> P2 {
    // OCS with normal (0,0,-1) maps (x, y) to world (-x, y) (arbitrary axis algorithm).
    if extrusion[2] < 0.0 { [-p[0], p[1]] } else { p }
}

fn rgba(c: [u8; 3]) -> [u8; 4] {
    [c[0], c[1], c[2], 255]
}

struct Colors<'a> {
    d: &'a Drawing,
    layer0: Option<u64>,
    /// Layers sorted by name, and handle -> position in that list.
    sorted: Vec<&'a crate::drawing::Layer>,
    index: HashMap<u64, u16>,
}

impl Colors<'_> {
    /// Index of a layer in the name-sorted layer list (unknown layers map to NO_LAYER).
    fn layer_idx(&self, layer: u64) -> u16 {
        self.index.get(&layer).copied().unwrap_or(NO_LAYER)
    }

    /// Layer list for the viewer: name, color, initially visible (not off or frozen in the DWG).
    fn layers_json(&self) -> String {
        self.sorted
            .iter()
            .map(|l| {
                let c = self.layer_rgba(l.handle);
                format!("{{\"n\":{},\"c\":[{},{},{}],\"on\":{}}}", json_str(&l.name), c[0], c[1], c[2], l.flags & 3 == 0)
            })
            .collect::<Vec<_>>()
            .join(",")
    }

    fn layer_rgba(&self, layer: u64) -> [u8; 4] {
        match self.d.layers.get(&layer).map(|l| l.color) {
            Some(Color::Rgb(v)) => rgba([(v >> 16) as u8, (v >> 8) as u8, v as u8]),
            Some(Color::Index(i)) => rgba(aci_rgb(i)),
            _ => rgba(aci_rgb(7)),
        }
    }

    fn fixed(c: Color) -> Option<[u8; 4]> {
        match c {
            Color::Index(i) => Some(rgba(aci_rgb(i))),
            Color::Rgb(v) => Some(rgba([(v >> 16) as u8, (v >> 8) as u8, v as u8])),
            _ => None,
        }
    }

    /// Vertex color, possibly deferred to the instance (MODE_*). Layer visibility is applied in the
    /// viewer, so every layer is kept.
    fn vertex(&self, layer: u64, c: Color) -> Option<[u8; 4]> {
        Some(match c {
            Color::ByBlock => [0, 0, 0, MODE_BYBLOCK],
            Color::ByLayer if Some(layer) == self.layer0 => [0, 0, 0, MODE_LAYER0],
            Color::ByLayer => self.layer_rgba(layer),
            c => Self::fixed(c).unwrap_or([255; 4]),
        })
    }

    /// Instance colors (ByBlock, layer 0) of a block inserted by an entity on `layer` with `color`.
    fn child(&self, parent: InstColors, layer: u64, color: Color) -> InstColors {
        let on_layer0 = Some(layer) == self.layer0;
        let layer_color = if on_layer0 { parent[1] } else { self.layer_rgba(layer) };
        let block = match color {
            Color::ByBlock => parent[0],
            Color::ByLayer => layer_color,
            c => Self::fixed(c).unwrap_or(parent[0]),
        };
        [block, layer_color]
    }
}

impl Scene {
    pub fn build(d: &Drawing, model_space: u64) -> Scene {
        let tol = tolerance(d, model_space);
        let mut sorted: Vec<&crate::drawing::Layer> = d.layers.values().collect();
        sorted.sort_by(|a, b| a.name.cmp(&b.name));
        let colors = Colors {
            d,
            layer0: d.layers.values().find(|l| l.name == "0").map(|l| l.handle),
            index: sorted.iter().enumerate().map(|(i, l)| (l.handle, i as u16)).collect(),
            sorted,
        };

        // 1. Content of every block, in block coordinates.
        let mut content: HashMap<u64, Content> = HashMap::new();
        // Work items: chunks of each block's entity list, processed on all cores and merged in order.
        const CHUNK: usize = 4096;
        let mut jobs: Vec<(u64, &[u64])> = Vec::new();
        for (&bh, b) in &d.blocks {
            content.insert(bh, Content::default());
            jobs.extend(b.entities.chunks(CHUNK).map(|c| (bh, c)));
        }
        let next = std::sync::atomic::AtomicUsize::new(0);
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
        let mut parts: Vec<(usize, Content)> = std::thread::scope(|sc| {
            let workers: Vec<_> = (0..threads)
                .map(|_| {
                    sc.spawn(|| {
                        let mut out = Vec::new();
                        loop {
                            let j = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            let Some(&(bh, ents)) = jobs.get(j) else { break };
                            let mut s = Content::default();
                            for h in ents {
                                let Some(e) = d.entities.get(h) else { continue };
                                if e.invisible {
                                    continue;
                                }
                                s.cur = u32::try_from(*h).unwrap_or(0);
                                s.lay = colors.layer_idx(e.layer);
                                add_entity(d, &colors, e, bh == model_space, tol, &mut s);
                                // one layer index per line vertex and per fill vertex added by this entity
                                let (nl, nf) = (s.pts.len() - s.layers.len(), s.fill.len() - s.fill_layers.len());
                                s.layers.extend(std::iter::repeat_n(s.lay, nl));
                                s.fill_layers.extend(std::iter::repeat_n(s.lay, nf));
                            }
                            out.push((j, s));
                        }
                        out
                    })
                })
                .collect();
            workers.into_iter().flat_map(|w| w.join().expect("scene worker panicked")).collect()
        });
        parts.sort_by_key(|p| p.0);
        for (j, part) in parts {
            if let Some(c) = content.get_mut(&jobs[j].0) {
                c.append(part);
            }
        }

        // 2. World instances of every block, with their inherited colors.
        let top = [rgba(aci_rgb(7)), colors.layer0.map_or(rgba(aci_rgb(7)), |l| colors.layer_rgba(l))];
        let mut instances: HashMap<u64, Vec<(Xf, InstColors, InstLayers)>> = HashMap::new();
        let top_layers = [colors.layer_idx(colors.layer0.unwrap_or(0)), NO_LAYER];
        let mut stack: Vec<(u64, Xf, InstColors, InstLayers, u32)> = vec![(model_space, IDENTITY, top, top_layers, 0)];
        while let Some((b, w, cols, lays, depth)) = stack.pop() {
            instances.entry(b).or_default().push((w, cols, lays));
            if depth >= 16 {
                continue; // recursive block reference
            }
            if let Some(c) = content.get(&b) {
                for kid in &c.kids {
                    let kc = colors.child(cols, kid.layer, kid.color);
                    let kl = colors.layer_idx(kid.layer);
                    // layer used by layer-0 entities inside; top-level INSERT layer (hides all)
                    let inherit = if Some(kid.layer) == colors.layer0 { lays[0] } else { kl };
                    let kls = [inherit, if depth == 0 { kl } else { lays[1] }];
                    for x in &kid.xfs {
                        stack.push((kid.block, mul(&w, x), kc, kls, depth + 1));
                    }
                }
            }
        }

        // 3. Bounding boxes, origin, flatten to f32 with per-block centers.
        let mut batches: Vec<(u64, P2)> = Vec::new();
        let mut world = [f64::MAX, f64::MAX, f64::MIN, f64::MIN];
        for (b, xs) in &instances {
            let Some(s) = content.get(b) else { continue };
            let text_pts = s.texts.iter().map(|t| t.pos);
            let mut bb = [f64::MAX, f64::MAX, f64::MIN, f64::MIN];
            for p in s.pts.iter().chain(s.fill.iter()).copied().chain(text_pts) {
                bb = [bb[0].min(p[0]), bb[1].min(p[1]), bb[2].max(p[0]), bb[3].max(p[1])];
            }
            if bb[0] > bb[2] {
                continue; // nothing to draw
            }
            for (x, ..) in xs {
                for corner in [[bb[0], bb[1]], [bb[2], bb[1]], [bb[0], bb[3]], [bb[2], bb[3]]] {
                    let q = apply(x, corner);
                    world = [world[0].min(q[0]), world[1].min(q[1]), world[2].max(q[0]), world[3].max(q[1])];
                }
            }
            batches.push((*b, [(bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0]));
        }
        let origin = [(world[0] + world[2]) / 2.0, (world[1] + world[3]) / 2.0];

        let mut scene = Scene {
            positions: Vec::new(),
            colors: Vec::new(),
            fill_positions: Vec::new(),
            fill_colors: Vec::new(),
            instances: Vec::new(),
            inst_colors: Vec::new(),
            layers: Vec::new(),
            fill_layers: Vec::new(),
            inst_layers: Vec::new(),
            layers_json: colors.layers_json(),
            layer_list: colors
                .sorted
                .iter()
                .map(|l| {
                    let c = colors.layer_rgba(l.handle);
                    (l.name.clone(), [c[0], c[1], c[2]], l.flags & 3 == 0)
                })
                .collect(),
            layer0: colors.layer_idx(colors.layer0.unwrap_or(0)),
            blocks: Vec::new(),
            texts: Vec::new(),
            ids: Vec::new(),
            centers: Vec::new(),
            origin,
            bbox: [world[0] - origin[0], world[1] - origin[1], world[2] - origin[0], world[3] - origin[1]],
            segments_drawn: 0,
            styles: d.styles.iter().map(|(h, s)| (*h, (s.width_factor, s.oblique))).collect(),
            styles_json: d
                .styles
                .iter()
                .map(|(h, s)| format!("\"{h}\":{{\"wf\":{},\"ob\":{},\"font\":{}}}", s.width_factor, s.oblique, json_str(&s.font)))
                .collect::<Vec<_>>()
                .join(","),
            triangles_drawn: 0,
        };
        for (b, center) in batches {
            let s = &content[&b];
            let xs = &instances[&b];
            let batch = scene.blocks.len();
            let vertex_start = scene.positions.len() / 2;
            for p in &s.pts {
                scene.positions.extend([(p[0] - center[0]) as f32, (p[1] - center[1]) as f32]);
            }
            s.cols.iter().for_each(|c| scene.colors.extend_from_slice(c));
            scene.ids.extend_from_slice(&s.ids);
            scene.layers.extend_from_slice(&s.layers);
            scene.centers.push(center);
            let fill_start = scene.fill_positions.len() / 2;
            for p in &s.fill {
                scene.fill_positions.extend([(p[0] - center[0]) as f32, (p[1] - center[1]) as f32]);
            }
            s.fill_cols.iter().for_each(|c| scene.fill_colors.extend_from_slice(c));
            scene.fill_layers.extend_from_slice(&s.fill_layers);
            let inst_start = scene.instances.len() / 6;
            for (x, cols, lays) in xs {
                // fold the block center into the translation, then move to the world origin
                let t = apply(x, center);
                for v in [x[0], x[1], x[2], x[3], t[0] - origin[0], t[1] - origin[1]] {
                    scene.instances.push(v as f32);
                }
                scene.inst_colors.extend_from_slice(&cols[0]);
                scene.inst_colors.extend_from_slice(&cols[1]);
                scene.inst_layers.extend_from_slice(lays);
            }
            for t in &s.texts {
                let mut t = t.clone();
                t.batch = batch;
                t.pos = [t.pos[0] - center[0], t.pos[1] - center[1]];
                t.align = [t.align[0] - center[0], t.align[1] - center[1]];
                scene.texts.push(t);
            }
            scene.segments_drawn += (s.pts.len() / 2 * xs.len()) as u64;
            scene.triangles_drawn += (s.fill.len() / 3 * xs.len()) as u64;
            scene.blocks.push(BlockDraw {
                vertex_start,
                vertex_count: s.pts.len(),
                fill_start,
                fill_count: s.fill.len(),
                inst_start,
                inst_count: xs.len(),
            });
        }
        scene
    }

    pub fn meta_json(&self) -> String {
        let blocks: Vec<String> = self
            .blocks
            .iter()
            .map(|b| {
                format!(
                    "[{},{},{},{},{},{}]",
                    b.vertex_start, b.vertex_count, b.fill_start, b.fill_count, b.inst_start, b.inst_count
                )
            })
            .collect();
        let centers: Vec<String> = self.centers.iter().map(|c| format!("[{},{}]", c[0], c[1])).collect();
        format!(
            "{{\"origin\":[{},{}],\"centers\":[{}],\"bbox\":[{},{},{},{}],\"blocks\":[{}],\"segments\":{},\"triangles\":{},\"vertices\":{},\"instances\":{},\"texts\":[{}],\"styles\":{{{}}},\"layers\":[{}],\"layer0\":{}}}",
            self.origin[0],
            self.origin[1],
            centers.join(","),
            self.bbox[0],
            self.bbox[1],
            self.bbox[2],
            self.bbox[3],
            blocks.join(","),
            self.segments_drawn,
            self.triangles_drawn,
            self.positions.len() / 2,
            self.instances.len() / 6,
            self.texts.iter().map(text_json).collect::<Vec<_>>().join(","),
            self.styles_json,
            self.layers_json,
            self.layer0
        )
    }
}

fn text_json(t: &TextItem) -> String {
    format!(
        "{{\"b\":{},\"id\":{},\"k\":{},\"x\":{},\"y\":{},\"ax\":{},\"ay\":{},\"h\":{},\"r\":{},\"wf\":{},\"ob\":{},\"ha\":{},\"va\":{},\"at\":{},\"w\":{},\"c\":[{},{},{},{}],\"s\":{},\"st\":{},\"pa\":[{}],\"l\":{}}}",
        t.batch,
        t.id,
        t.kind,
        t.pos[0],
        t.pos[1],
        t.align[0],
        t.align[1],
        t.height,
        t.rotation,
        t.width_factor,
        t.oblique,
        t.halign,
        t.valign,
        t.attachment,
        t.width,
        t.color[0],
        t.color[1],
        t.color[2],
        t.color[3],
        json_str(&t.value),
        t.style,
        t.para.iter().map(|a| a.to_string()).collect::<Vec<_>>().join(","),
        t.layer
    )
}

fn add_entity(d: &Drawing, colors: &Colors, e: &Entity, in_model_space: bool, tol: f64, s: &mut Content) {
    match &e.kind {
        Kind::Insert { block, insert, scale, rotation, extrusion, cols, rows, col_spacing, row_spacing, attribs } => {
            // Attributes live in the parent's coordinates.
            for a in attribs {
                if let Some(ae) = d.entities.get(a) {
                    if let Kind::Attrib { text, flags, is_def: false, .. } = &ae.kind {
                        if flags & 1 == 0 && !ae.invisible {
                            if let Some(c) = colors.vertex(ae.layer, ae.color) {
                                let saved = (s.cur, s.lay);
                                (s.cur, s.lay) = (u32::try_from(*a).unwrap_or(0), colors.layer_idx(ae.layer));
                                s.text(text, clean_text(&text.value), c);
                                (s.cur, s.lay) = saved;
                            }
                        }
                    }
                }
            }
            let Some(child) = d.blocks.get(block) else { return };
            let (sn, cs) = rotation.sin_cos();
            let base = child.base;
            let mut xfs = Vec::new();
            for ci in 0..(*cols).max(1) {
                for ri in 0..(*rows).max(1) {
                    let off = [ci as f64 * col_spacing, ri as f64 * row_spacing];
                    let t = [insert[0] + cs * off[0] - sn * off[1], insert[1] + sn * off[0] + cs * off[1]];
                    let m: Xf = [
                        cs * scale[0],
                        sn * scale[0],
                        -sn * scale[1],
                        cs * scale[1],
                        t[0] - (cs * scale[0] * base[0] - sn * scale[1] * base[1]),
                        t[1] - (sn * scale[0] * base[0] + cs * scale[1] * base[1]),
                    ];
                    xfs.push(if extrusion[2] < 0.0 { mul(&MIRROR_X, &m) } else { m });
                }
            }
            s.kids.push(Kid { block: *block, xfs, layer: e.layer, color: e.color });
            return;
        }
        // Dimension blocks are stored in world coordinates.
        Kind::Dimension { block } => {
            s.kids.push(Kid { block: *block, xfs: vec![IDENTITY], layer: e.layer, color: e.color });
            return;
        }
        _ => {}
    }
    let Some(c) = colors.vertex(e.layer, e.color) else { return };
    match &e.kind {
        Kind::Line { a, b } => s.line(*a, *b, c),
        Kind::Circle { center, radius, extrusion } => {
            let p: Vec<P2> =
                arc_points(*center, *radius, 0.0, TAU, tol).into_iter().map(|p| mirror_x_if(extrusion, p)).collect();
            s.polyline(&p, false, c);
        }
        Kind::Arc { center, radius, start, end, extrusion } => {
            let p: Vec<P2> =
                arc_points(*center, *radius, *start, *end, tol).into_iter().map(|p| mirror_x_if(extrusion, p)).collect();
            s.polyline(&p, false, c);
        }
        Kind::Ellipse { center, major, ratio, start, end } => {
            s.polyline(&ellipse_points(*center, *major, *ratio, *start, *end, tol), false, c);
        }
        Kind::LwPolyline { closed, pts, bulges, extrusion, .. } => {
            let p: Vec<P2> = bulged(pts, bulges, *closed, tol).into_iter().map(|p| mirror_x_if(extrusion, p)).collect();
            s.polyline(&p, false, c);
        }
        Kind::Polyline2d { closed, vertices, extrusion } => {
            let (mut pts, mut bul) = (Vec::new(), Vec::new());
            for v in vertices {
                if let Some(Kind::Vertex2d { p, bulge }) = d.entities.get(v).map(|e| &e.kind) {
                    pts.push(*p);
                    bul.push(*bulge);
                }
            }
            let p: Vec<P2> = bulged(&pts, &bul, *closed, tol).into_iter().map(|p| mirror_x_if(extrusion, p)).collect();
            s.polyline(&p, false, c);
        }
        Kind::Spline { degree, closed, knots, ctrl, weights, fit, tangents, knot_param } => {
            let pts = if ctrl.is_empty() {
                match interpolate_fit(fit, *tangents, *knot_param, *closed) {
                    Some((k, cp)) => nurbs_points(3, &k, &cp, &[]).unwrap_or_else(|| fit.clone()),
                    None => fit.clone(),
                }
            } else {
                nurbs_points(*degree as usize, knots, ctrl, weights).unwrap_or_else(|| ctrl.clone())
            };
            s.polyline(&pts, false, c);
        }
        Kind::Solid { corners, extrusion } => {
            // DWG corner order is 1, 2, 4, 3 around the outline; a SOLID is filled.
            let o = [corners[0], corners[1], corners[3], corners[2]].map(|p| mirror_x_if(extrusion, p));
            for tri in [[o[0], o[1], o[2]], [o[0], o[2], o[3]]] {
                for p in tri {
                    s.fill.push(p);
                    s.fill_cols.push(c);
                }
            }
        }
        Kind::Hatch { solid, paths, pattern, extrusion } => {
            let mut loops = hatch::loops(paths, tol);
            for l in loops.iter_mut() {
                l.iter_mut().for_each(|p| *p = mirror_x_if(extrusion, *p));
            }
            if *solid {
                let before = s.fill.len();
                hatch::fill_fans(&loops, &mut s.fill);
                s.fill_cols.extend(std::iter::repeat_n(c, s.fill.len() - before));
            } else {
                // Pattern lines are in OCS too: mirror the definition the same way.
                let pat: Vec<_> = if extrusion[2] < 0.0 {
                    pattern
                        .iter()
                        .map(|pl| crate::entities::PatternLine {
                            angle: std::f64::consts::PI - pl.angle,
                            base: [-pl.base[0], pl.base[1]],
                            offset: [-pl.offset[0], pl.offset[1]],
                            dashes: pl.dashes.clone(),
                        })
                        .collect()
                } else {
                    pattern.clone()
                };
                let mut segs = Vec::new();
                hatch::pattern_segments(&loops, &pat, PATTERN_BUDGET, &mut segs);
                for pair in segs.chunks_exact(2) {
                    s.line(pair[0], pair[1], c);
                }
            }
        }
        Kind::Text(t) => s.text(t, clean_text(&t.value), c),
        // Attribute definitions show their tag only where they are placed directly in model space.
        Kind::Attrib { text, tag, is_def: true, .. } if in_model_space => s.text(text, clean_text(tag), c),
        Kind::MText { insert, x_dir, rect_width, height, attachment, value, style } => {
            let (v, para) = mtext_paragraphs(value);
            if !v.trim().is_empty() {
                s.texts.push(TextItem {
                    batch: 0,
                    kind: 1,
                    pos: *insert,
                    align: *insert,
                    height: *height,
                    rotation: x_dir[1].atan2(x_dir[0]),
                    width_factor: 1.0,
                    oblique: 0.0,
                    halign: 0,
                    valign: 0,
                    attachment: *attachment,
                    width: *rect_width,
                    color: c,
                    id: s.cur,
                    layer: s.lay,
                    value: v,
                    style: *style,
                    para,
                });
            }
        }
        Kind::Proxy(prims) => {
            use crate::proxy::Prim;
            for (pc, prim) in prims {
                let Some(c) = pc.map_or(Some(c), |pc| colors.vertex(e.layer, pc)) else { continue };
                match prim {
                    Prim::Polyline { pts, closed } => s.polyline(pts, *closed, c),
                    Prim::Fill(pts) => {
                        let before = s.fill.len();
                        hatch::fill_fans(std::slice::from_ref(pts), &mut s.fill);
                        s.fill_cols.extend(std::iter::repeat_n(c, s.fill.len() - before));
                    }
                    Prim::Arc { center, radius, start, end } => s.polyline(&arc_points(*center, *radius, *start, *end, tol), false, c),
                    Prim::Text(t) => s.text(t, clean_text(&t.value), c),
                }
            }
        }
        // shortcut: POINT, VIEWPORT and custom entities without proxy graphics are not drawn
        _ => {}
    }
}

/// Chord tolerance: 1e-5 of the model space extent (about 1 px when the whole drawing spans 100k px).
fn tolerance(d: &Drawing, model_space: u64) -> f64 {
    let mut bb = [f64::MAX, f64::MAX, f64::MIN, f64::MIN];
    if let Some(b) = d.blocks.get(&model_space) {
        for h in &b.entities {
            let p = match d.entities.get(h).map(|e| &e.kind) {
                Some(Kind::Line { a, .. }) => *a,
                Some(Kind::Insert { insert, .. }) => *insert,
                Some(Kind::Arc { center, .. }) | Some(Kind::Circle { center, .. }) => *center,
                _ => continue,
            };
            bb = [bb[0].min(p[0]), bb[1].min(p[1]), bb[2].max(p[0]), bb[3].max(p[1])];
        }
    }
    let ext = (bb[2] - bb[0]).max(bb[3] - bb[1]);
    if ext.is_finite() && ext > 0.0 { ext * 1e-5 } else { 0.01 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transform_composition() {
        let rot90: Xf = [0.0, 1.0, -1.0, 0.0, 0.0, 0.0];
        let tr: Xf = [1.0, 0.0, 0.0, 1.0, 5.0, 0.0];
        assert_eq!(apply(&mul(&tr, &rot90), [1.0, 0.0]), [5.0, 1.0]);
    }
}
