//! Object framing for R2010+ (ODA spec 20.1/20.2): size, handle stream, string stream, common data.

use crate::bits::{Bits, HandleRef};
use crate::container::{Error, Result, err};
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
}

impl<'a> Frame<'a> {
    /// `objects` is the whole AcDb:AcDbObjects section, `offset` comes from the object map.
    pub fn parse(objects: &'a [u8], offset: usize) -> Result<Self> {
        let mut r = Bits::at(objects, offset * 8, objects.len() * 8);
        let size = r.ms()? as usize;
        let handle_bits = r.umc()? as usize;
        let start = r.pos;
        let end = start + size * 8;
        let crc_at = end / 8;
        let stored = objects
            .get(crc_at..crc_at + 2)
            .map(|s| u16::from_le_bytes([s[0], s[1]]))
            .ok_or_else(|| Error(format!("object at {offset:#x}: size {size} past end of section")))?;
        let crc_ok = crc8(0xC0C1, &objects[offset..crc_at]) == stored;
        if handle_bits > size * 8 {
            return err(format!("object at {offset:#x}: handle stream larger than object"));
        }
        let handle_start = end - handle_bits;

        r.end = handle_start;
        let type_code = r.ot()?;
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
        let mut main_end = flag_pos;
        if Bits::at(objects, flag_pos, handle_start).b()? {
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
        })
    }

    /// Next TV string (TU from the string stream). Missing string stream gives "".
    pub fn tv(&mut self) -> Result<String> {
        match self.strings.as_mut() {
            Some(s) => s.tu(),
            None => Ok(String::new()),
        }
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

/// Common data of a non-entity object (after EED).
pub struct ObjCommon {
    pub num_reactors: u32,
    pub has_xdic: bool,
}

pub fn obj_common(f: &mut Frame) -> Result<ObjCommon> {
    let num_reactors = f.main.bl()? as u32;
    let has_xdic = !f.main.b()?;
    f.main.b()?; // R2013+: has DS binary data
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
}

/// Common entity data (spec 20.4.1, R2010+ fields).
pub fn ent_common(f: &mut Frame) -> Result<EntCommon> {
    let r = &mut f.main;
    if r.b()? {
        let n = r.bll()? as usize;
        r.skip_bytes(n)?; // proxy graphics
    }
    let entmode = r.bb()?;
    let num_reactors = r.bl()? as u32;
    let has_xdic = !r.b()?;
    r.b()?; // R2013+: has DS binary data
    // No "nolinks" bit here: the spec lists it, but AC1032 data does not have it (checked by hand).
    let (color, color_has_handle) = enc_color(r)?;
    let linetype_scale = r.bd()?;
    let ltype_flags = r.bb()?;
    let plotstyle_flags = r.bb()?;
    let material_flags = r.bb()?;
    r.rc()?; // shadow flags
    let visual_styles = (r.b()? as u8) + (r.b()? as u8) + (r.b()? as u8);
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
    })
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
    Ok(EntHandles { owner, layer })
}
