//! Object framing (ODA spec 20.1/20.2): size, handle stream, string stream (R2007+), common data.

use crate::bits::{Bits, HandleRef};
use crate::container::{Error, Result, Version, err};
use crate::crc::crc8;

/// One object split into its three streams.
pub struct Frame<'a> {
    pub type_code: u16,
    pub handle: u64,
    /// Object data, positioned right after the EED. Ends where the string data begins.
    pub main: Bits<'a>,
    /// R2007+ string stream (TU strings), if present.
    pub strings: Option<Bits<'a>>,
    /// Handle stream.
    pub handles: Bits<'a>,
    pub crc_ok: bool,
    pub ver: Version,
}

impl<'a> Frame<'a> {
    /// `objects` is the whole AcDb:AcDbObjects section, `offset` comes from the object map.
    pub fn parse(objects: &'a [u8], offset: usize, ver: Version) -> Result<Self> {
        if ver == Version::R2000 {
            return Self::parse_r2000(objects, offset);
        }
        let mut r = Bits::at(objects, offset * 8, objects.len() * 8);
        let size = r.ms()? as usize;
        // R2010+: size of the handle stream. R2007: size of the data in bits, after the type.
        let handle_bits = if ver >= Version::R2010 { Some(r.umc()? as usize) } else { None };
        let start = r.pos;
        let end = start + size * 8;
        let crc_at = end / 8;
        let stored = objects
            .get(crc_at..crc_at + 2)
            .map(|s| u16::from_le_bytes([s[0], s[1]]))
            .ok_or_else(|| Error(format!("object at {offset:#x}: size {size} past end of section")))?;
        let crc_ok = crc8(0xC0C1, &objects[offset..crc_at]) == stored;
        let (type_code, handle_start) = match handle_bits {
            Some(hb) => {
                if hb > size * 8 {
                    return err(format!("object at {offset:#x}: handle stream larger than object"));
                }
                r.end = end - hb;
                (r.ot()?, end - hb)
            }
            None => {
                r.end = end;
                let t = r.bs()? as u16;
                let hs = start + r.rl()? as usize;
                if hs > end || hs < r.pos {
                    return err(format!("object at {offset:#x}: data size does not fit"));
                }
                r.end = hs;
                (t, hs)
            }
        };
        let handle = r.h()?.value;
        loop {
            let n = r.bs()? as u16 as usize;
            if n == 0 {
                break;
            }
            r.h()?; // EED app handle
            r.skip_bytes(n)?;
        }

        // String stream (spec 20.1, R2007+): a flag in the last bit before the handle stream.
        let flag_pos = handle_start
            .checked_sub(1)
            .ok_or_else(|| Error(format!("object at {offset:#x}: empty data stream")))?;
        let mut strings = None;
        let mut main_end = if ver >= Version::R2007 { flag_pos } else { handle_start };
        if ver >= Version::R2007 && Bits::at(objects, flag_pos, handle_start).b()? {
            let mut p = flag_pos.checked_sub(16).ok_or_else(|| Error("string stream size underflow".into()))?;
            let mut len = Bits::at(objects, p, p + 16).rs()? as usize;
            if len & 0x8000 != 0 {
                p = p.checked_sub(16).ok_or_else(|| Error("string stream size underflow".into()))?;
                let hi = Bits::at(objects, p, p + 16).rs()? as usize;
                len = (len & 0x7FFF) | (hi << 15);
            }
            let s = p.checked_sub(len).filter(|&s| s >= r.pos).ok_or_else(|| {
                Error(format!("object at {offset:#x}: string stream of {len} bits does not fit"))
            })?;
            strings = Some(Bits::at(objects, s, p));
            main_end = s;
        }
        r.end = main_end;
        Ok(Frame {
            type_code,
            handle,
            main: r,
            strings,
            handles: Bits::at(objects, handle_start, end),
            crc_ok,
            ver,
        })
    }

    /// R2000 (spec 20.1): MS size, BS type, RL size of the data in bits, handle; strings inline.
    fn parse_r2000(objects: &'a [u8], offset: usize) -> Result<Self> {
        let mut r = Bits::at(objects, offset * 8, objects.len() * 8);
        let size = r.ms()? as usize;
        let start = r.pos;
        let end = start + size * 8;
        let crc_at = end / 8;
        let stored = objects
            .get(crc_at..crc_at + 2)
            .map(|s| u16::from_le_bytes([s[0], s[1]]))
            .ok_or_else(|| Error(format!("object at {offset:#x}: size {size} past end of file")))?;
        let crc_ok = crc8(0xC0C1, &objects[offset..crc_at]) == stored;
        r.end = end;
        let type_code = r.bs()? as u16;
        let data_bits = r.rl()? as usize;
        let handle_start = start + data_bits;
        if handle_start > end || handle_start < r.pos {
            return err(format!("object at {offset:#x}: data size {data_bits} bits does not fit"));
        }
        r.end = handle_start;
        let handle = r.h()?.value;
        loop {
            let n = r.bs()? as u16 as usize;
            if n == 0 {
                break;
            }
            r.h()?; // EED app handle
            r.skip_bytes(n)?;
        }
        Ok(Frame {
            type_code,
            handle,
            main: r,
            strings: None,
            handles: Bits::at(objects, handle_start, end),
            crc_ok,
            ver: Version::R2000,
        })
    }

    /// Next TV string: from the string stream (R2007+) or inline in the data stream (R2000, R2004).
    pub fn tv(&mut self) -> Result<String> {
        tv(&mut self.main, self.strings.as_mut(), self.ver)
    }

    /// Next handle from the handle stream, resolved to an absolute handle.
    pub fn href(&mut self) -> Result<u64> {
        Ok(self.handles.h()?.absolute(self.handle))
    }

    pub fn href_raw(&mut self) -> Result<HandleRef> {
        self.handles.h()
    }

    /// True if the data stream was consumed exactly (self-check for a parsed prescription).
    pub fn data_fully_read(&self) -> bool {
        self.main.pos == self.main.end
    }
}

/// Next TV string for prescriptions that hold the data stream `r` and the string stream `s` apart.
/// R2007+ without a string stream: every string is empty.
pub fn tv(r: &mut Bits, s: Option<&mut Bits>, ver: Version) -> Result<String> {
    match s {
        Some(s) => s.tu(),
        None if ver <= Version::R2004 => r.tv8(),
        None => Ok(String::new()),
    }
}

/// Common data of a non-entity object (after EED).
pub struct ObjCommon {
    pub num_reactors: u32,
    pub has_xdic: bool,
}

pub fn obj_common(f: &mut Frame) -> Result<ObjCommon> {
    let num_reactors = f.main.bl()? as u32;
    let has_xdic = f.ver == Version::R2000 || !f.main.b()?; // R2004+: "xdic missing" bit
    if f.ver >= Version::R2013 {
        f.main.b()?; // has DS binary data
    }
    Ok(ObjCommon { num_reactors, has_xdic })
}

/// Skip owner, reactors and xdictionary at the start of an object's handle stream.
pub fn obj_common_handles(f: &mut Frame, c: &ObjCommon) -> Result<u64> {
    let owner = f.href()?;
    for _ in 0..c.num_reactors {
        f.href()?;
    }
    if c.has_xdic {
        f.href()?;
    }
    Ok(owner)
}

/// ENC entity color (spec 2.11).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Color {
    ByLayer,
    ByBlock,
    Index(u8),
    Rgb(u32),
}

pub struct EntCommon {
    pub entmode: u8,
    pub num_reactors: u32,
    pub has_xdic: bool,
    pub color: Color,
    color_has_handle: bool,
    pub ltype_flags: u8,
    pub plotstyle_flags: u8,
    pub material_flags: u8,
    pub visual_styles: u8,
    pub invisible: bool,
    pub lineweight: u8,
    pub linetype_scale: f64,
    /// Proxy graphics bytes (spec ch. 29), when the entity carries them.
    pub proxy: Option<Vec<u8>>,
    /// R2000: previous/next entity handles are stored (the "nolinks" bit was 0).
    links: bool,
}

/// Common entity data (spec 20.4.1).
pub fn ent_common(f: &mut Frame) -> Result<EntCommon> {
    let ver = f.ver;
    let r = &mut f.main;
    let mut proxy = None;
    if r.b()? {
        let n = if ver <= Version::R2007 { r.rl()? as usize } else { r.bll()? as usize };
        if n > r.remaining() / 8 {
            return err(format!("proxy graphics of {n} bytes past end of object"));
        }
        proxy = Some((0..n).map(|_| r.rc()).collect::<Result<Vec<u8>>>()?);
    }
    let entmode = r.bb()?;
    let num_reactors = r.bl()? as u32;
    let mut links = false;
    let (has_xdic, color, color_has_handle);
    if ver == Version::R2000 {
        has_xdic = true;
        links = !r.b()?; // nolinks
        color = cmc_color(r.bs()?);
        color_has_handle = false;
    } else {
        has_xdic = !r.b()?;
        if ver >= Version::R2013 {
            r.b()?; // has DS binary data
        }
        // No "nolinks" bit here: the spec lists it, but AC1032 data does not have it (checked by hand).
        (color, color_has_handle) = enc_color(r)?;
    }
    let linetype_scale = r.bd()?;
    let ltype_flags = r.bb()?;
    let plotstyle_flags = r.bb()?;
    let (mut material_flags, mut visual_styles) = (0, 0);
    if ver >= Version::R2007 {
        material_flags = r.bb()?;
        r.rc()?; // shadow flags
    }
    if ver >= Version::R2010 {
        visual_styles = (r.b()? as u8) + (r.b()? as u8) + (r.b()? as u8);
    }
    let invisible = r.bs()? & 1 != 0;
    let lineweight = r.rc()?;
    Ok(EntCommon {
        entmode,
        num_reactors,
        has_xdic,
        color,
        color_has_handle,
        ltype_flags,
        plotstyle_flags,
        material_flags,
        visual_styles,
        invisible,
        lineweight,
        linetype_scale,
        proxy,
        links,
    })
}

/// CMC color index (R2000 and table entries): 0 BYBLOCK, 256 BYLAYER.
pub fn cmc_color(v: i16) -> Color {
    match v {
        0 => Color::ByBlock,
        1..=255 => Color::Index(v as u8),
        _ => Color::ByLayer,
    }
}

fn enc_color(r: &mut Bits) -> Result<(Color, bool)> {
    let v = r.bs()? as u16;
    let flags = v & 0xE000;
    let mut color = match v & 0x01FF {
        0 => Color::ByBlock,
        i if i >= 256 => Color::ByLayer,
        i => Color::Index(i as u8),
    };
    if flags & 0x8000 != 0 {
        color = Color::Rgb(r.bl()? as u32 & 0x00FF_FFFF);
    }
    if flags & 0x2000 != 0 {
        r.bl()?; // transparency
    }
    Ok((color, flags & 0x4000 != 0))
}

pub struct EntHandles {
    pub owner: Option<u64>,
    pub layer: u64,
    /// R2000 only: next entity in the owner's chain (0 = none).
    pub next: u64,
}

/// Common entity handle data (spec 20.4.2, R2010+).
pub fn ent_common_handles(f: &mut Frame, c: &EntCommon) -> Result<EntHandles> {
    let owner = if c.entmode == 0 { Some(f.href()?) } else { None };
    for _ in 0..c.num_reactors {
        f.href()?;
    }
    if c.has_xdic {
        f.href()?;
    }
    let mut next = 0;
    if f.ver == Version::R2000 {
        // Without links the chain is implied: previous = handle - 1, next = handle + 1.
        next = f.handle + 1;
        if c.links {
            f.href()?; // previous
            next = f.href()?;
        }
    }
    if c.color_has_handle {
        f.href()?;
    }
    let layer = f.href()?;
    if c.ltype_flags == 3 {
        f.href()?;
    }
    if c.material_flags == 3 {
        f.href()?;
    }
    if c.plotstyle_flags == 3 {
        f.href()?;
    }
    for _ in 0..c.visual_styles {
        f.href()?;
    }
    Ok(EntHandles { owner, layer, next })
}
