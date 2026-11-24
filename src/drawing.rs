//! Whole-drawing loading: object index, classes, block records, layouts, layers, entities.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::classes::{Class, parse_classes};
use crate::container::{Container, Result, Version};
use crate::entities::{Entity, Kind, entity_next, parse_entity};
use crate::handles::parse_object_map;
use crate::objects::{Color, Frame, cmc_color, obj_common, obj_common_handles, tv};

pub const TYPE_BLOCK_HEADER: u16 = 49;
pub const TYPE_LAYER: u16 = 51;
pub const TYPE_LAYOUT: u16 = 82;
pub const TYPE_STYLE: u16 = 53;

#[derive(Debug, Clone)]
pub struct BlockRecord {
    pub handle: u64,
    pub name: String,
    pub base: [f64; 2],
    pub entities: Vec<u64>,
    pub layout: u64,
    pub exact: bool,
}

/// R2000 owners store only the first and last member; members link to the next one.
const CHAIN_PENDING: u64 = u64::MAX;

#[derive(Debug, Clone)]
pub struct Layout {
    pub handle: u64,
    pub name: String,
    pub tab_order: i32,
    pub block_record: u64,
    pub viewports: Vec<u64>,
    pub exact: bool,
}


/// Text style (STYLE table entry).
#[derive(Debug, Clone)]
pub struct TextStyle {
    pub name: String,
    /// Font file, e.g. "romans.shx" or "arial.ttf".
    pub font: String,
    pub width_factor: f64,
    pub oblique: f64,
}

#[derive(Debug, Clone)]
pub struct Layer {
    pub handle: u64,
    pub name: String,
    pub flags: i16,
    pub color: Color,
}

#[derive(Debug, Default, Clone)]
pub struct Timings {
    pub read_container: Duration,
    pub decompress: Duration,
    pub index: Duration,
    pub parse: Duration,
    /// Sequential insertion of parsed objects into the maps.
    pub merge: Duration,
}

pub struct Drawing {
    pub version: String,
    pub classes: Vec<Class>,
    /// Object type code for every handle.
    pub types: HashMap<u64, u16>,
    pub blocks: HashMap<u64, BlockRecord>,
    pub layouts: Vec<Layout>,
    pub layers: HashMap<u64, Layer>,
    pub styles: HashMap<u64, TextStyle>,
    pub entities: HashMap<u64, Entity>,
    /// Objects whose frame or prescription failed: (handle, type, error).
    pub errors: Vec<(u64, u16, String)>,
    pub timings: Timings,
}

impl Drawing {
    pub fn type_name(&self, code: u16) -> String {
        type_name(&self.classes, code)
    }
}

pub fn type_name(classes: &[Class], code: u16) -> String {
    let fixed = match code {
        1 => "TEXT",
        2 => "ATTRIB",
        3 => "ATTDEF",
        4 => "BLOCK",
        5 => "ENDBLK",
        6 => "SEQEND",
        7 => "INSERT",
        8 => "MINSERT",
        10 => "VERTEX_2D",
        11 => "VERTEX_3D",
        15 => "POLYLINE_2D",
        16 => "POLYLINE_3D",
        17 => "ARC",
        18 => "CIRCLE",
        19 => "LINE",
        20..=26 => "DIMENSION",
        27 => "POINT",
        28 => "3DFACE",
        31 => "SOLID",
        32 => "TRACE",
        33 => "SHAPE",
        34 => "VIEWPORT",
        35 => "ELLIPSE",
        36 => "SPLINE",
        37 => "REGION",
        38 => "3DSOLID",
        40 => "RAY",
        41 => "XLINE",
        42 => "DICTIONARY",
        44 => "MTEXT",
        45 => "LEADER",
        46 => "TOLERANCE",
        47 => "MLINE",
        48 => "BLOCK_CONTROL",
        49 => "BLOCK_HEADER",
        50 => "LAYER_CONTROL",
        51 => "LAYER",
        52 => "STYLE_CONTROL",
        53 => "STYLE",
        56 => "LTYPE_CONTROL",
        57 => "LTYPE",
        60 => "VIEW_CONTROL",
        61 => "VIEW",
        62 => "UCS_CONTROL",
        63 => "UCS",
        64 => "VPORT_CONTROL",
        65 => "VPORT",
        66 => "APPID_CONTROL",
        67 => "APPID",
        68 => "DIMSTYLE_CONTROL",
        69 => "DIMSTYLE",
        70 => "VP_ENT_HDR_CONTROL",
        71 => "VP_ENT_HDR",
        72 => "GROUP",
        73 => "MLINESTYLE",
        74 => "OLE2FRAME",
        76 => "LONG_TRANSACTION",
        77 => "LWPOLYLINE",
        78 => "HATCH",
        79 => "XRECORD",
        80 => "ACDBPLACEHOLDER",
        81 => "VBA_PROJECT",
        82 => "LAYOUT",
        _ => "",
    };
    if !fixed.is_empty() {
        return fixed.to_string();
    }
    classes
        .iter()
        .find(|c| c.number == code)
        .map_or_else(|| format!("type {code}"), |c| c.dxf_name.clone())
}

/// True for type codes that are graphical entities.
pub fn is_entity_type(classes: &[Class], code: u16) -> bool {
    match code {
        1..=41 | 44..=47 | 77 | 78 => true,
        c if c >= 500 => classes.iter().any(|k| k.number == c && k.is_entity),
        _ => false,
    }
}

enum Parsed {
    Entity(Entity),
    Block(BlockRecord),
    Layout(Layout),
    Layer(Layer),
    Style(TextStyle),
    Skip,
}

impl Drawing {
    pub fn load(raw: &[u8], threads: usize) -> Result<Drawing> {
        let t0 = Instant::now();
        let c = Container::parse(raw)?;
        let read_container = t0.elapsed();

        let t1 = Instant::now();
        let objects = c.section("AcDb:AcDbObjects", threads)?;
        let handles = c.section("AcDb:Handles", 1)?;
        let classes = parse_classes(&c.section("AcDb:Classes", 1)?, c.ver)?;
        let decompress = t1.elapsed();

        let t2 = Instant::now();
        let map = parse_object_map(&handles)?;
        let index = t2.elapsed();

        let t3 = Instant::now();
        let threads = threads.max(1);
        let chunk = map.len().div_ceil(threads).max(1);
        let broken_next = std::sync::Mutex::new(HashMap::new());
        let results: Vec<Vec<(u64, u16, Result<Parsed>)>> = std::thread::scope(|s| {
            let workers: Vec<_> = map
                .chunks(chunk)
                .map(|part| {
                    let (objects, classes, ver, broken_next) = (&objects, &classes, c.ver, &broken_next);
                    s.spawn(move || {
                        part.iter()
                            .map(|&(h, off)| match Frame::parse(objects, off, ver) {
                                Ok(mut f) => {
                                    let t = f.type_code;
                                    let res = parse_one(&mut f, classes);
                                    if res.is_err() && ver <= Version::R2000 && is_entity_type(classes, t) {
                                        // Keep the chain link of a failed entity.
                                        if let Some(next) = Frame::parse(objects, off, ver).ok().and_then(|mut f| entity_next(&mut f).ok()) {
                                            broken_next.lock().unwrap().insert(h, next);
                                        }
                                    }
                                    (h, t, res)
                                }
                                Err(e) => (h, 0, Err(e)),
                            })
                            .collect()
                    })
                })
                .collect();
            workers.into_iter().map(|w| w.join().expect("parser thread panicked")).collect()
        });

        let parse = t3.elapsed();
        let t4 = Instant::now();
        let mut d = Drawing {
            version: c.version.clone(),
            classes,
            types: HashMap::with_capacity(map.len()),
            blocks: HashMap::new(),
            layouts: Vec::new(),
            layers: HashMap::new(),
            styles: HashMap::new(),
            entities: HashMap::with_capacity(map.len()),
            errors: Vec::new(),
            timings: Timings::default(),
        };
        for (h, t, res) in results.into_iter().flatten() {
            d.types.insert(h, t);
            match res {
                Ok(Parsed::Entity(e)) => {
                    d.entities.insert(h, e);
                }
                Ok(Parsed::Block(b)) => {
                    d.blocks.insert(h, b);
                }
                Ok(Parsed::Layout(l)) => d.layouts.push(l),
                Ok(Parsed::Style(s)) => {
                    d.styles.insert(h, s);
                }
                Ok(Parsed::Layer(l)) => {
                    d.layers.insert(h, l);
                }
                Ok(Parsed::Skip) => {}
                Err(e) => d.errors.push((h, t, e.0)),
            }
        }
        d.layouts.sort_by_key(|l| l.tab_order);
        if c.ver <= Version::R2000 {
            d.resolve_chains(&broken_next.into_inner().unwrap());
        }
        d.timings = Timings { read_container, decompress, index, parse, merge: t4.elapsed() };
        Ok(d)
    }
}

impl Drawing {
    /// R2000: expand (first, last) member pairs into the full lists by following `Entity::next`.
    /// `broken_next` holds the links of entities whose prescription failed.
    fn resolve_chains(&mut self, broken_next: &HashMap<u64, u64>) {
        let entities = &self.entities;
        let chain = |v: &mut Vec<u64>| {
            if v.len() != 3 || v[0] != CHAIN_PENDING {
                return;
            }
            let (first, last) = (v[1], v[2]);
            v.clear();
            let mut h = first;
            // The guard stops a broken or cyclic chain.
            while h != 0 && v.len() <= entities.len() {
                v.push(h);
                if h == last {
                    break;
                }
                h = entities.get(&h).map(|e| e.next).or_else(|| broken_next.get(&h).copied()).unwrap_or(0);
            }
        };
        for b in self.blocks.values_mut() {
            chain(&mut b.entities);
        }
        let mut pending: Vec<(u64, Vec<u64>)> = Vec::new();
        for (&h, e) in &self.entities {
            if let Kind::Polyline2d { vertices: v, .. } | Kind::Insert { attribs: v, .. } = &e.kind {
                if v.first() == Some(&CHAIN_PENDING) {
                    let mut v = v.clone();
                    chain(&mut v);
                    pending.push((h, v));
                }
            }
        }
        for (h, list) in pending {
            if let Some(e) = self.entities.get_mut(&h) {
                if let Kind::Polyline2d { vertices: v, .. } | Kind::Insert { attribs: v, .. } = &mut e.kind {
                    *v = list;
                }
            }
        }
    }
}

/// R2000 chain placeholder for a member list: resolved by `Drawing::resolve_chains`.
pub(crate) fn pending_chain(first: u64, last: u64) -> Vec<u64> {
    vec![CHAIN_PENDING, first, last]
}

fn parse_one(f: &mut Frame, classes: &[Class]) -> Result<Parsed> {
    let code = f.type_code;
    let class_name = (code >= 500).then(|| type_name(classes, code));
    match code {
        TYPE_BLOCK_HEADER => block_header(f).map(Parsed::Block),
        TYPE_LAYER => layer(f).map(Parsed::Layer),
        TYPE_STYLE => text_style(f).map(Parsed::Style),
        TYPE_LAYOUT => layout(f).map(Parsed::Layout),
        c if c >= 500 && class_name.as_deref() == Some("LAYOUT") => layout(f).map(Parsed::Layout),
        // Unsupported entity types still get their common part (layer, owner) and Kind::Other.
        c if is_entity_type(classes, c) => parse_entity(f, class_name.as_deref()).map(Parsed::Entity),
        _ => Ok(Parsed::Skip),
    }
}

fn block_header(f: &mut Frame) -> Result<BlockRecord> {
    let c = obj_common(f)?;
    let name = f.tv()?;
    let hbudget = f.handles.remaining();
    let ver = f.ver;
    let r = &mut f.main;
    table_entry_flags(r, ver)?;
    r.b()?; // anonymous
    r.b()?; // has attdefs
    let is_xref = r.b()?;
    let overlaid = r.b()?;
    if ver >= Version::R2000 {
        r.b()?; // loaded
    }
    let owned = if ver >= Version::R2004 { r.count_in(8, hbudget)? } else { 0 };
    let base = r.bd3()?;
    let base = [base[0], base[1]];
    tv(r, f.strings.as_mut(), ver)?; // xref path
    let mut inserts = 0usize;
    if ver >= Version::R2000 {
        while r.rc()? != 0 {
            inserts += 1;
        }
        tv(r, f.strings.as_mut(), ver)?; // description
        let preview = r.count_in(8, r.remaining())?;
        r.skip_bytes(preview)?;
    }
    if ver >= Version::R2007 {
        r.bs()?; // insert units
        r.b()?; // explodable
        r.rc()?; // block scaling
    }
    let exact = f.data_fully_read();

    obj_common_handles(f, &c)?;
    f.href()?; // NULL
    f.href()?; // BLOCK entity
    let mut entities = Vec::with_capacity(owned);
    if ver <= Version::R2000 {
        if !is_xref && !overlaid {
            let first = f.href()?;
            let last = f.href()?;
            if first != 0 {
                entities = pending_chain(first, last);
            }
        }
    } else {
        for _ in 0..owned {
            entities.push(f.href()?);
        }
    }
    f.href()?; // ENDBLK
    for _ in 0..inserts {
        f.href()?;
    }
    let layout = if ver >= Version::R2000 { f.href()? } else { 0 };
    Ok(BlockRecord { handle: f.handle, name, base, entities, layout, exact })
}

/// Table entry flags after the name. The spec lists `64-flag B, xrefindex+1 BS, Xdep B`;
/// R2000 and R2004 have all three; R2007+ data has only the BS (checked by hand on AC1032).
fn table_entry_flags(r: &mut crate::bits::Bits, ver: Version) -> Result<()> {
    if ver <= Version::R2004 {
        r.b()?; // 64-flag
        r.bs()?; // xref index + 1
        r.b()?; // xdep
    } else {
        r.bs()?; // xref index + 1
    }
    Ok(())
}

fn layer(f: &mut Frame) -> Result<Layer> {
    obj_common(f)?;
    let name = f.tv()?;
    let r = &mut f.main;
    table_entry_flags(r, f.ver)?;
    if f.ver < Version::R2000 {
        // R13-R14: one bit per flag, in the order of the R2000+ flag bits; on = !off.
        let frozen = r.b()? as i16;
        let off = !r.b()? as i16;
        let frozen_new = r.b()? as i16;
        let locked = r.b()? as i16;
        let mut flags = frozen | off << 1 | frozen_new << 2 | locked << 3;
        let index = r.bs()?;
        if index < 0 {
            flags |= 2; // negative index = layer off
        }
        if !f.data_fully_read() {
            return Err(crate::container::Error("LAYER: data stream not fully read".into()));
        }
        return Ok(Layer { handle: f.handle, name, flags, color: cmc_color(index.abs()) });
    }
    let flags = r.bs()?; // frozen 1, off 2, frozen in new vp 4, locked 8, plot 16, lineweight 0x3E0
    if f.ver <= Version::R2000 {
        let color = cmc_color(r.bs()?.abs()); // negative index = layer off
        if !f.data_fully_read() {
            return Err(crate::container::Error("LAYER: data stream not fully read".into()));
        }
        return Ok(Layer { handle: f.handle, name, flags, color });
    }
    r.bs()?; // CMC: index
    let rgb = r.bl()? as u32; // CMC: method byte + value
    let color = match rgb >> 24 {
        0xC2 => Color::Rgb(rgb & 0x00FF_FFFF),
        0xC3 => Color::Index(rgb as u8),
        _ => Color::Index(7),
    };
    let cflags = r.rc()?;
    if cflags & 1 != 0 {
        f.tv()?;
    }
    if cflags & 2 != 0 {
        f.tv()?;
    }
    if !f.data_fully_read() {
        return Err(crate::container::Error("LAYER: data stream not fully read".into()));
    }
    Ok(Layer { handle: f.handle, name, flags, color })
}

fn layout(f: &mut Frame) -> Result<Layout> {
    let c = obj_common(f)?;
    let hbudget = f.handles.remaining();
    let ver = f.ver;
    let r = &mut f.main;
    let mut s = f.strings.as_mut();
    // PLOTSETTINGS part
    tv(r, s.as_deref_mut(), ver)?; // page setup name
    tv(r, s.as_deref_mut(), ver)?; // printer / config
    r.bs()?;
    for _ in 0..6 {
        r.bd()?;
    }
    tv(r, s.as_deref_mut(), ver)?; // paper size
    r.bd2()?;
    r.bs()?;
    r.bs()?;
    r.bs()?;
    r.bd2()?;
    r.bd2()?;
    if ver <= Version::R2000 {
        tv(r, s.as_deref_mut(), ver)?; // plot view name
    }
    r.bd()?;
    r.bd()?;
    tv(r, s.as_deref_mut(), ver)?; // style sheet
    r.bs()?;
    r.bd()?;
    r.bd2()?;
    if ver >= Version::R2004 {
        r.bs()?;
        r.bs()?;
        r.bs()?;
    }
    // LAYOUT part
    let name = tv(r, s, ver)?;
    let tab_order = r.bl()?;
    r.bs()?;
    r.bd3()?;
    r.rd2()?;
    r.rd2()?;
    r.bd3()?;
    r.bd3()?;
    r.bd3()?;
    r.bd()?;
    r.bs()?;
    r.bd3()?;
    r.bd3()?;
    // spec says RL; AC1032 data has a BL. R2000 has no viewport list.
    let nvp = if ver >= Version::R2004 { r.count_in(8, hbudget)? } else { 0 };
    let exact = f.data_fully_read();

    obj_common_handles(f, &c)?;
    if ver >= Version::R2004 {
        f.href()?; // plot view
    }
    if ver >= Version::R2007 {
        f.href()?; // visual style
    }
    let block_record = f.href()?;
    f.href()?; // last active viewport
    f.href()?; // base ucs
    f.href()?; // named ucs
    let mut viewports = Vec::with_capacity(nvp);
    for _ in 0..nvp {
        viewports.push(f.href()?);
    }
    Ok(Layout { handle: f.handle, name, tab_order, block_record, viewports, exact })
}

fn text_style(f: &mut Frame) -> Result<TextStyle> {
    obj_common(f)?;
    let name = f.tv()?;
    let r = &mut f.main;
    table_entry_flags(r, f.ver)?;
    r.b()?; // vertical
    r.b()?; // shape file
    r.bd()?; // fixed height
    let width_factor = r.bd()?;
    let oblique = r.bd()?;
    r.rc()?; // generation
    r.bd()?; // last height
    let font = tv(r, f.strings.as_mut(), f.ver)?;
    tv(r, f.strings.as_mut(), f.ver)?; // big font
    if !f.data_fully_read() {
        return Err(crate::container::Error("STYLE: data stream not fully read".into()));
    }
    Ok(TextStyle { name, font, width_factor, oblique })
}
