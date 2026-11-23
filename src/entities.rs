//! 2D entity prescriptions (ODA spec 20.4), R2000 to R2018. Z values are read and dropped.

use crate::bits::Bits;
use crate::container::{Result, Version};
use crate::drawing::pending_chain;
use crate::objects::{Color, Frame, ent_common, ent_common_handles, tv};

pub type P2 = [f64; 2];

fn xy(p: [f64; 3]) -> P2 {
    [p[0], p[1]]
}

#[derive(Debug, Clone)]
pub struct Entity {
    pub handle: u64,
    pub layer: u64,
    pub color: Color,
    pub invisible: bool,
    /// Owner from the handle stream; only stored for entmode 0 (block members, vertices, attribs).
    pub owner: Option<u64>,
    pub kind: Kind,
    /// True if the prescription consumed the data stream exactly (format self-check).
    pub exact: bool,
    /// R2000 only: next entity in the owner's chain (0 = none).
    pub next: u64,
}

#[derive(Debug, Clone)]
pub enum HatchSeg {
    Line(P2, P2),
    Arc { center: P2, radius: f64, start: f64, end: f64, ccw: bool },
    Ellipse { center: P2, major: P2, ratio: f64, start: f64, end: f64, ccw: bool },
    Spline { degree: u32, knots: Vec<f64>, ctrl: Vec<P2>, weights: Vec<f64> },
}

/// One hatch pattern line family: lines at `angle` through `base`, repeated every `offset`,
/// with `dashes` (positive = dash, negative = gap, 0 = dot; empty = continuous).
#[derive(Debug, Clone)]
pub struct PatternLine {
    pub angle: f64,
    pub base: P2,
    pub offset: P2,
    pub dashes: Vec<f64>,
}

#[derive(Debug, Clone)]
pub enum HatchPath {
    Polyline { closed: bool, pts: Vec<P2>, bulges: Vec<f64> },
    Segments(Vec<HatchSeg>),
}

#[derive(Debug, Clone)]
pub struct Text {
    pub insert: P2,
    pub align: P2,
    pub height: f64,
    pub rotation: f64,
    pub width_factor: f64,
    pub oblique: f64,
    pub halign: i16,
    pub valign: i16,
    pub value: String,
    pub extrusion: [f64; 3],
    /// STYLE handle (0 = unknown).
    pub style: u64,
}

#[derive(Debug, Clone)]
pub enum Kind {
    Line { a: P2, b: P2 },
    Point { p: P2 },
    Circle { center: P2, radius: f64, extrusion: [f64; 3] },
    Arc { center: P2, radius: f64, start: f64, end: f64, extrusion: [f64; 3] },
    Ellipse { center: P2, major: P2, ratio: f64, start: f64, end: f64 },
    /// Fit-point splines: `tangents` are the start/end directions (zero = unset), `knot_param`
    /// 0 chord, 1 square root, 2 uniform.
    Spline {
        degree: u32,
        closed: bool,
        knots: Vec<f64>,
        ctrl: Vec<P2>,
        weights: Vec<f64>,
        fit: Vec<P2>,
        tangents: [P2; 2],
        knot_param: i32,
    },
    LwPolyline { closed: bool, pts: Vec<P2>, bulges: Vec<f64>, const_width: f64, extrusion: [f64; 3] },
    Polyline2d { closed: bool, vertices: Vec<u64>, extrusion: [f64; 3] },
    Vertex2d { p: P2, bulge: f64 },
    Solid { corners: [P2; 4], extrusion: [f64; 3] },
    Text(Text),
    /// ATTRIB (2) or ATTDEF (3). `flags` bit 0 = invisible.
    Attrib { text: Text, tag: String, flags: u8, is_def: bool },
    MText { insert: P2, x_dir: P2, rect_width: f64, height: f64, attachment: i16, value: String, style: u64 },
    Insert {
        block: u64,
        insert: P2,
        scale: [f64; 3],
        rotation: f64,
        extrusion: [f64; 3],
        cols: u16,
        rows: u16,
        col_spacing: f64,
        row_spacing: f64,
        attribs: Vec<u64>,
    },
    Dimension { block: u64 },
    /// `solid` is also true for gradient fills (drawn as solid).
    Hatch { solid: bool, paths: Vec<HatchPath>, pattern: Vec<PatternLine>, extrusion: [f64; 3] },
    Viewport {
        center: P2,
        width: f64,
        height: f64,
        view_center: P2,
        view_height: f64,
        twist: f64,
        status: u32,
        frozen_layers: Vec<u64>,
        clip_boundary: u64,
    },
    /// Type not decoded; counted only.
    Other,
}

/// Parse an entity whose frame is positioned after the EED. `kind_name` is the class DXF name for
/// types >= 500 ("LWPOLYLINE", "HATCH"), otherwise unused.
pub fn parse_entity(f: &mut Frame, kind_name: Option<&str>) -> Result<Entity> {
    let c = ent_common(f)?;
    let code = match kind_name {
        Some("LWPOLYLINE") => 77,
        Some("HATCH") => 78,
        _ => f.type_code,
    };
    let hbudget = f.handles.remaining();
    let ver = f.ver;
    let r = &mut f.main;
    let mut tail_handles = TailHandles::None;
    let kind = match code {
        19 => {
            let z_zero = r.b()?;
            let x0 = r.rd()?;
            let x1 = r.dd(x0)?;
            let y0 = r.rd()?;
            let y1 = r.dd(y0)?;
            if !z_zero {
                let z0 = r.rd()?;
                r.dd(z0)?;
            }
            r.bt()?;
            r.be()?;
            Kind::Line { a: [x0, y0], b: [x1, y1] }
        }
        27 => {
            let p = r.bd3()?;
            r.bt()?;
            r.be()?;
            r.bd()?;
            Kind::Point { p: xy(p) }
        }
        18 => {
            let center = r.bd3()?;
            let radius = r.bd()?;
            r.bt()?;
            let extrusion = r.be()?;
            Kind::Circle { center: xy(center), radius, extrusion }
        }
        17 => {
            let center = r.bd3()?;
            let radius = r.bd()?;
            r.bt()?;
            let extrusion = r.be()?;
            let start = r.bd()?;
            let end = r.bd()?;
            Kind::Arc { center: xy(center), radius, start, end, extrusion }
        }
        35 => {
            let center = r.bd3()?;
            let major = r.bd3()?;
            r.bd3()?;
            let ratio = r.bd()?;
            let start = r.bd()?;
            let end = r.bd()?;
            Kind::Ellipse { center: xy(center), major: xy(major), ratio, start, end }
        }
        36 => spline(r, ver)?,
        77 => lwpolyline(r, ver)?,
        15 => {
            let flags = r.bs()?;
            r.bs()?; // curve type
            r.bd()?;
            r.bd()?;
            r.bt()?;
            r.bd()?; // elevation
            let extrusion = r.be()?;
            let owned = if ver >= Version::R2004 { r.count_in(8, hbudget)? } else { 0 };
            tail_handles = TailHandles::Owned { owned };
            Kind::Polyline2d { closed: flags & 1 != 0, vertices: Vec::new(), extrusion }
        }
        10 => {
            r.rc()?; // flags
            let p = r.bd3()?;
            let sw = r.bd()?;
            if sw >= 0.0 {
                r.bd()?;
            }
            let bulge = r.bd()?;
            if ver >= Version::R2010 {
                r.bl()?; // vertex id
            }
            r.bd()?; // tangent dir
            Kind::Vertex2d { p: xy(p), bulge }
        }
        31 => {
            r.bt()?;
            r.bd()?;
            let corners = [r.rd2()?, r.rd2()?, r.rd2()?, r.rd2()?];
            let extrusion = r.be()?;
            Kind::Solid { corners, extrusion }
        }
        1 => {
            let t = text(r, f.strings.as_mut(), ver)?;
            tail_handles = TailHandles::Style;
            Kind::Text(t)
        }
        2 | 3 => {
            let t = text(r, f.strings.as_mut(), ver)?;
            if ver >= Version::R2010 {
                r.rc()?; // version
            }
            let attr_type = if ver >= Version::R2018 { r.rc()? } else { 1 };
            let mut tag = String::new();
            let mut flags = 0;
            if attr_type == 1 {
                tag = tv(r, f.strings.as_mut(), ver)?;
                r.bs()?;
                flags = r.rc()?;
                if ver >= Version::R2007 {
                    r.b()?; // lock position
                }
                if code == 3 {
                    if ver >= Version::R2010 {
                        r.rc()?; // version
                    }
                    tv(r, f.strings.as_mut(), ver)?; // prompt
                }
            }
            // Multi-line attributes embed a whole MTEXT: not decoded, `exact` reports it.
            tail_handles = TailHandles::Style;
            Kind::Attrib { text: t, tag, flags, is_def: code == 3 }
        }
        44 => {
            let insert = r.bd3()?;
            r.bd3()?; // extrusion
            let x_dir = r.bd3()?;
            let rect_width = r.bd()?;
            if ver >= Version::R2007 {
                r.bd()?; // rect height
            }
            let height = r.bd()?;
            let attachment = r.bs()?;
            r.bs()?; // drawing dir
            r.bd()?;
            r.bd()?; // extents
            let value = tv(r, f.strings.as_mut(), ver)?;
            r.bs()?;
            r.bd()?;
            r.b()?;
            if ver >= Version::R2004 {
                let bg = r.bl()?;
                if bg & 0x11 != 0 {
                    r.bd()?; // scale factor (spec says BL; BD matches the data)
                    cmc(r, f.strings.as_mut(), ver)?;
                    r.bl()?;
                }
            }
            let annotative_not = ver >= Version::R2018 && r.b()?;
            if annotative_not {
                r.bs()?;
                r.b()?;
                r.bl()?;
                r.bd3()?;
                r.bd3()?;
                r.bd()?;
                r.bd()?;
                r.bd()?;
                r.bd()?;
                let col_type = r.bs()?;
                if col_type != 0 {
                    let n = r.bl()?;
                    r.bd()?;
                    r.bd()?;
                    let auto_height = r.b()?;
                    r.b()?;
                    if !auto_height && col_type == 2 {
                        for _ in 0..n {
                            r.bd()?;
                        }
                    }
                }
            }
            tail_handles = if annotative_not { TailHandles::MTextApp } else { TailHandles::Style };
            Kind::MText { insert: xy(insert), x_dir: xy(x_dir), rect_width, height, attachment, value, style: 0 }
        }
        7 | 8 => {
            let insert = r.bd3()?;
            let scale = match r.bb()? {
                3 => [1.0; 3],
                1 => {
                    let y = r.dd(1.0)?;
                    let z = r.dd(1.0)?;
                    [1.0, y, z]
                }
                2 => {
                    let x = r.rd()?;
                    [x; 3]
                }
                _ => {
                    let x = r.rd()?;
                    let y = r.dd(x)?;
                    let z = r.dd(x)?;
                    [x, y, z]
                }
            };
            let rotation = r.bd()?;
            let extrusion = r.bd3()?;
            let has_attribs = r.b()?;
            let owned = if has_attribs && ver >= Version::R2004 { r.count_in(8, hbudget)? } else { 0 };
            let (mut cols, mut rows, mut col_spacing, mut row_spacing) = (1u16, 1u16, 0.0, 0.0);
            if code == 8 {
                cols = r.bs()? as u16;
                rows = r.bs()? as u16;
                col_spacing = r.bd()?;
                row_spacing = r.bd()?;
            }
            tail_handles = TailHandles::Insert { owned, has_attribs };
            Kind::Insert {
                block: 0,
                insert: xy(insert),
                scale,
                rotation,
                extrusion,
                cols,
                rows,
                col_spacing,
                row_spacing,
                attribs: Vec::new(),
            }
        }
        20..=26 => {
            dimension(r, code, f.strings.as_mut(), ver)?;
            tail_handles = TailHandles::Dimension;
            Kind::Dimension { block: 0 }
        }
        78 => hatch(r, f.strings.as_mut(), ver, &mut tail_handles, hbudget)?,
        34 => viewport(r, f.strings.as_mut(), ver, &mut tail_handles, hbudget)?,
        _ => Kind::Other,
    };

    let exact = !matches!(kind, Kind::Other) && f.data_fully_read();
    let h = ent_common_handles(f, &c)?;
    let mut kind = kind;
    match (&mut kind, tail_handles) {
        (Kind::Polyline2d { vertices, .. }, TailHandles::Owned { owned }) => {
            if ver == Version::R2000 {
                let first = f.href()?;
                *vertices = pending_chain(first, f.href()?);
            }
            for _ in 0..owned {
                vertices.push(f.href()?);
            }
        }
        (Kind::Insert { block, attribs, .. }, TailHandles::Insert { owned, has_attribs }) => {
            *block = f.href()?;
            if ver == Version::R2000 && has_attribs {
                let first = f.href()?;
                let last = f.href()?;
                if first != 0 {
                    *attribs = pending_chain(first, last);
                }
            }
            for _ in 0..owned {
                attribs.push(f.href()?);
            }
        }
        (Kind::Text(t) | Kind::Attrib { text: t, .. }, TailHandles::Style) => t.style = f.href()?,
        (Kind::MText { style, .. }, TailHandles::Style | TailHandles::MTextApp) => *style = f.href()?,
        (Kind::Dimension { block }, TailHandles::Dimension) => {
            f.href()?; // dimstyle
            *block = f.href()?;
        }
        (Kind::Viewport { frozen_layers, clip_boundary, .. }, TailHandles::Viewport { frozen }) => {
            for _ in 0..frozen {
                frozen_layers.push(f.href()?);
            }
            *clip_boundary = f.href()?;
        }
        _ => {}
    }
    Ok(Entity {
        handle: f.handle,
        layer: h.layer,
        color: c.color,
        invisible: c.invisible,
        owner: h.owner,
        kind,
        exact,
        next: h.next,
    })
}

/// R2000: the next entity of a frame whose prescription failed, so the owner's chain goes on.
pub fn entity_next(f: &mut Frame) -> Result<u64> {
    let c = ent_common(f)?;
    Ok(ent_common_handles(f, &c)?.next)
}

enum TailHandles {
    None,
    Style,
    MTextApp,
    Owned { owned: usize },
    Insert { owned: usize, has_attribs: bool },
    Dimension,
    Viewport { frozen: usize },
}

fn cmc(r: &mut Bits, s: Option<&mut Bits>, ver: Version) -> Result<()> {
    r.bs()?;
    if ver == Version::R2000 {
        return Ok(());
    }
    r.bl()?;
    let flags = r.rc()?;
    if let Some(s) = s {
        if flags & 1 != 0 {
            s.tu()?;
        }
        if flags & 2 != 0 {
            s.tu()?;
        }
    }
    Ok(())
}

fn text(r: &mut Bits, s: Option<&mut Bits>, ver: Version) -> Result<Text> {
    let flags = r.rc()?;
    if flags & 0x01 == 0 {
        r.rd()?; // elevation
    }
    let insert = r.rd2()?;
    let align = if flags & 0x02 == 0 { [r.dd(insert[0])?, r.dd(insert[1])?] } else { insert };
    let extrusion = r.be()?;
    r.bt()?;
    let oblique = if flags & 0x04 == 0 { r.rd()? } else { 0.0 };
    let rotation = if flags & 0x08 == 0 { r.rd()? } else { 0.0 };
    let height = r.rd()?;
    let width_factor = if flags & 0x10 == 0 { r.rd()? } else { 1.0 };
    let value = tv(r, s, ver)?;
    if flags & 0x20 == 0 {
        r.bs()?; // generation
    }
    let halign = if flags & 0x40 == 0 { r.bs()? } else { 0 };
    let valign = if flags & 0x80 == 0 { r.bs()? } else { 0 };
    Ok(Text { insert, align, height, rotation, width_factor, oblique, halign, valign, value, extrusion, style: 0 })
}

fn spline(r: &mut Bits, ver: Version) -> Result<Kind> {
    let scenario = r.bl()?;
    let (flags1, knot_param) = if ver >= Version::R2013 { (r.bl()?, r.bl()?) } else { (0, 0) };
    let degree = r.bl()? as u32;
    let mut closed = flags1 & 4 != 0;
    let (mut nknots, mut nctrl, mut nfit, mut weighted) = (0usize, 0usize, 0usize, false);
    let mut tangents = [[0.0; 2]; 2];
    // R2013+: the "method fit points" flag means fit data only, also when scenario is 1
    // (observed on AC1032: the data stream ends right after the fit points).
    let fit_only = scenario == 2 || flags1 & 1 != 0;
    if fit_only {
        r.bd()?;
        tangents = [xy(r.bd3()?), xy(r.bd3()?)];
        nfit = r.count()?;
    } else if scenario == 1 {
        r.b()?; // rational
        closed = r.b()?;
        r.b()?; // periodic
        r.bd()?;
        r.bd()?;
        nknots = r.count()?;
        nctrl = r.count()?;
        weighted = r.b()?;
    }
    let mut knots = Vec::with_capacity(nknots);
    for _ in 0..nknots {
        knots.push(r.bd()?);
    }
    let mut ctrl = Vec::with_capacity(nctrl);
    let mut weights = Vec::new();
    for _ in 0..nctrl {
        ctrl.push(xy(r.bd3()?));
        if weighted {
            weights.push(r.bd()?);
        }
    }
    let mut fit = Vec::with_capacity(nfit);
    for _ in 0..nfit {
        fit.push(xy(r.bd3()?));
    }
    Ok(Kind::Spline { degree, closed, knots, ctrl, weights, fit, tangents, knot_param })
}

fn lwpolyline(r: &mut Bits, ver: Version) -> Result<Kind> {
    let flag = r.bs()?;
    let const_width = if flag & 4 != 0 { r.bd()? } else { 0.0 };
    if flag & 8 != 0 {
        r.bd()?;
    }
    if flag & 2 != 0 {
        r.bd()?;
    }
    let extrusion = if flag & 1 != 0 { r.bd3()? } else { [0.0, 0.0, 1.0] };
    let n = r.count()?;
    let nbulges = if flag & 16 != 0 { r.count()? } else { 0 };
    let nids = if flag & 1024 != 0 && ver >= Version::R2010 { r.count()? } else { 0 };
    let nwidths = if flag & 32 != 0 { r.count()? } else { 0 };
    let mut pts: Vec<P2> = Vec::with_capacity(n);
    if n > 0 {
        pts.push(r.rd2()?);
        for k in 1..n {
            let prev = pts[k - 1];
            pts.push([r.dd(prev[0])?, r.dd(prev[1])?]);
        }
    }
    let mut bulges = Vec::with_capacity(nbulges);
    for _ in 0..nbulges {
        bulges.push(r.bd()?);
    }
    for _ in 0..nids {
        r.bl()?;
    }
    for _ in 0..nwidths {
        r.bd()?;
        r.bd()?;
    }
    Ok(Kind::LwPolyline { closed: flag & 512 != 0, pts, bulges, const_width, extrusion })
}

fn dimension(r: &mut Bits, code: u16, s: Option<&mut Bits>, ver: Version) -> Result<()> {
    if ver >= Version::R2010 {
        r.rc()?; // version
    }
    r.bd3()?;
    r.rd2()?;
    r.bd()?;
    r.rc()?;
    tv(r, s, ver)?;
    for _ in 0..6 {
        // text rotation, horizontal dir, insert scale x/y/z, insert rotation
        r.bd()?;
    }
    r.bs()?;
    r.bs()?;
    r.bd()?;
    r.bd()?;
    if ver >= Version::R2007 {
        r.b()?;
        r.b()?;
        r.b()?;
    }
    r.rd2()?; // 12-pt
    match code {
        20 => {
            r.bd3()?;
            r.bd3()?;
            r.bd3()?;
            r.rc()?;
        }
        21 => {
            r.bd3()?;
            r.bd3()?;
            r.bd3()?;
            r.bd()?;
            r.bd()?;
        }
        22 => {
            r.bd3()?;
            r.bd3()?;
            r.bd3()?;
            r.bd()?;
        }
        23 => {
            r.rd2()?;
            r.bd3()?;
            r.bd3()?;
            r.bd3()?;
            r.bd3()?;
        }
        24 => {
            r.rd2()?;
            r.bd3()?;
            r.bd3()?;
            r.bd3()?;
            r.bd3()?;
            r.bd3()?;
        }
        _ => {
            // 25 radius, 26 diameter
            r.bd3()?;
            r.bd3()?;
            r.bd()?;
        }
    }
    Ok(())
}

fn hatch(r: &mut Bits, mut s: Option<&mut Bits>, ver: Version, tail: &mut TailHandles, hbudget: usize) -> Result<Kind> {
    let mut gradient = 0;
    if ver >= Version::R2004 {
        gradient = r.bl()?;
        r.bl()?;
        r.bd()?;
        r.bd()?;
        r.bl()?;
        r.bd()?;
        let ncolors = r.bl()?;
        for _ in 0..ncolors {
            r.bd()?;
            r.bs()?;
            r.bl()?;
            r.rc()?;
        }
        tv(r, s.as_deref_mut(), ver)?; // gradient name
    }
    r.bd()?; // z
    let extrusion = r.bd3()?;
    tv(r, s, ver)?; // pattern name
    let solid = r.b()?;
    r.b()?;
    let npaths = r.count()?;
    let mut paths = Vec::with_capacity(npaths);
    let mut any_pixel = false;
    let mut nbound = 0usize;
    for _ in 0..npaths {
        let flag = r.bl()?;
        any_pixel |= flag & 4 != 0;
        if flag & 2 == 0 {
            let nseg = r.count()?;
            let mut segs = Vec::with_capacity(nseg);
            for _ in 0..nseg {
                segs.push(match r.rc()? {
                    1 => HatchSeg::Line(r.rd2()?, r.rd2()?),
                    2 => HatchSeg::Arc {
                        center: r.rd2()?,
                        radius: r.bd()?,
                        start: r.bd()?,
                        end: r.bd()?,
                        ccw: r.b()?,
                    },
                    3 => HatchSeg::Ellipse {
                        center: r.rd2()?,
                        major: r.rd2()?,
                        ratio: r.bd()?,
                        start: r.bd()?,
                        end: r.bd()?,
                        ccw: r.b()?,
                    },
                    4 => {
                        let degree = r.bl()? as u32;
                        let rational = r.b()?;
                        r.b()?;
                        let nk = r.count()?;
                        let nc = r.count()?;
                        let mut knots = Vec::with_capacity(nk);
                        for _ in 0..nk {
                            knots.push(r.bd()?);
                        }
                        let mut ctrl = Vec::with_capacity(nc);
                        let mut weights = Vec::new();
                        for _ in 0..nc {
                            ctrl.push(r.rd2()?);
                            if rational {
                                weights.push(r.bd()?);
                            }
                        }
                        if ver >= Version::R2010 {
                            let nfit = r.count()?;
                            for _ in 0..nfit {
                                r.rd2()?;
                            }
                            r.rd2()?; // start tangent
                            r.rd2()?; // end tangent
                        }
                        HatchSeg::Spline { degree, knots, ctrl, weights }
                    }
                    t => {
                        return Err(crate::container::Error(format!("hatch: unknown edge type {t}")));
                    }
                });
            }
            paths.push(HatchPath::Segments(segs));
        } else {
            let has_bulges = r.b()?;
            let closed = r.b()?;
            let n = r.count()?;
            let mut pts = Vec::with_capacity(n);
            let mut bulges = Vec::new();
            for _ in 0..n {
                pts.push(r.rd2()?);
                if has_bulges {
                    bulges.push(r.bd()?);
                }
            }
            paths.push(HatchPath::Polyline { closed, pts, bulges });
        }
        nbound += r.count_in(8, hbudget)?;
    }
    r.bs()?; // style
    r.bs()?; // pattern type
    let mut pattern = Vec::new();
    if !solid {
        r.bd()?;
        r.bd()?;
        r.b()?;
        let n = r.bs()?;
        for _ in 0..n {
            // Already scaled and rotated: these are the effective pattern lines (DXF 53/43/44/45/46/49).
            let angle = r.bd()?;
            let base = r.bd2()?;
            let offset = r.bd2()?;
            let nd = r.bs()?;
            let mut dashes = Vec::new();
            for _ in 0..nd {
                dashes.push(r.bd()?);
            }
            pattern.push(PatternLine { angle, base, offset, dashes });
        }
    }
    if any_pixel {
        r.bd()?;
    }
    let nseed = r.bl()?;
    for _ in 0..nseed {
        r.rd2()?;
    }
    *tail = TailHandles::Owned { owned: nbound };
    Ok(Kind::Hatch { solid: solid || gradient != 0, paths, pattern, extrusion })
}

fn viewport(r: &mut Bits, s: Option<&mut Bits>, ver: Version, tail: &mut TailHandles, hbudget: usize) -> Result<Kind> {
    let center = r.bd3()?;
    let width = r.bd()?;
    let height = r.bd()?;
    r.bd3()?; // target
    r.bd3()?; // direction
    let twist = r.bd()?;
    let view_height = r.bd()?;
    for _ in 0..4 {
        r.bd()?; // lens, front, back, snap angle
    }
    let view_center = r.rd2()?;
    for _ in 0..3 {
        r.rd2()?;
    }
    r.bs()?;
    if ver >= Version::R2007 {
        r.bs()?; // grid major
    }
    let frozen = r.count_in(8, hbudget)?;
    let status = r.bl()? as u32;
    let mut s = s;
    tv(r, s.as_deref_mut(), ver)?; // style sheet
    r.rc()?;
    r.b()?;
    r.b()?;
    r.bd3()?;
    r.bd3()?;
    r.bd3()?;
    r.bd()?;
    r.bs()?;
    if ver >= Version::R2004 {
        r.bs()?; // shadeplot mode
    }
    if ver >= Version::R2007 {
        r.b()?;
        r.rc()?;
        r.bd()?;
        r.bd()?;
        cmc(r, s, ver)?;
    }
    *tail = TailHandles::Viewport { frozen };
    Ok(Kind::Viewport {
        center: xy(center),
        width,
        height,
        view_center,
        view_height,
        twist,
        status,
        frozen_layers: Vec::new(),
        clip_boundary: 0,
    })
}
