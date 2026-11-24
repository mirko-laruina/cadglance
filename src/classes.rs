//! AcDb:Classes (ODA spec 10.2 / 5.8): maps object types >= 500 to class names.

use crate::bits::Bits;
use crate::container::{Error, Result, Version, err};
use crate::objects::tv;

const SENTINEL: [u8; 16] =
    [0x8D, 0xA1, 0xC4, 0xB8, 0xC4, 0xA9, 0xF8, 0xC5, 0xC0, 0xDC, 0xF4, 0x5F, 0xE7, 0xCF, 0xB6, 0x8A];

#[derive(Debug, Clone)]
pub struct Class {
    pub number: u16,
    pub dxf_name: String,
    pub cpp_name: String,
    pub is_entity: bool,
}

/// Layout observed on AC1032 (bytes): SN(16), RL size, RL size-high, RL size in bits (at 24),
/// then the bit data (at 28). The string stream sits at the end of the bit range.
pub fn parse_classes(data: &[u8], ver: Version) -> Result<Vec<Class>> {
    if data.get(..16) != Some(&SENTINEL[..]) {
        return err("classes: bad sentinel");
    }
    let rl = |o: usize| -> Result<u32> {
        data.get(o..o + 4)
            .map(|s| u32::from_le_bytes(s.try_into().unwrap()))
            .ok_or_else(|| Error("classes: truncated".into()))
    };
    if ver <= Version::R2004 {
        // Spec 10.1/10.2: RL size in bytes, then the bit data at byte 20 with inline TV strings.
        let end = (20 + rl(16)? as usize) * 8;
        if end > data.len() * 8 {
            return err("classes: size past end of section");
        }
        let mut r = Bits::at(data, 20 * 8, end);
        // Up to 7 padding bits can follow the last class.
        return read_classes(&mut r, None, ver, 8);
    }
    // R2007 has no size-high RL: the bit-size field is at byte 20 instead of 24.
    let bits_at = if ver == Version::R2007 { 20 } else { 24 };
    let bit_size = rl(bits_at)? as usize;
    // The size in bits counts from the bit-size field itself; the bit data starts right after it.
    let end = bits_at * 8 + bit_size;
    if end > data.len() * 8 {
        return err("classes: bit size past end of section");
    }

    // String stream, same rule as objects (spec 20.1).
    let flag_pos = end - 1;
    let mut main_end = flag_pos;
    let mut strings = None;
    if Bits::at(data, flag_pos, end).b()? {
        let mut p = flag_pos - 16;
        let mut len = Bits::at(data, p, p + 16).rs()? as usize;
        if len & 0x8000 != 0 {
            p -= 16;
            let hi = Bits::at(data, p, p + 16).rs()? as usize;
            len = (len & 0x7FFF) | (hi << 15);
        }
        let s = p.checked_sub(len).ok_or_else(|| Error("classes: bad string stream size".into()))?;
        strings = Some(Bits::at(data, s, p));
        main_end = s;
    }
    let mut strings = strings.ok_or_else(|| Error("classes: no string stream".into()))?;

    let mut r = Bits::at(data, (bits_at + 4) * 8, main_end);
    read_classes(&mut r, Some(&mut strings), ver, 9)
}

/// Class records after the section header. R2004+ starts with a preamble and stores 5 BL counters
/// per class. Strings come from `strings` (R2007+) or inline. Stops when fewer than `min_left` bits remain.
fn read_classes(r: &mut Bits, mut strings: Option<&mut Bits>, ver: Version, min_left: usize) -> Result<Vec<Class>> {
    if ver >= Version::R2004 {
        r.bs()?; // maximum class number
        r.rc()?;
        r.rc()?;
        r.b()?;
    }
    let mut out = Vec::new();
    while r.end - r.pos >= min_left {
        let number = r.bs()? as u16;
        r.bs()?; // version / proxy flags
        let _app = tv(r, strings.as_deref_mut(), ver)?;
        let cpp_name = tv(r, strings.as_deref_mut(), ver)?;
        let dxf_name = tv(r, strings.as_deref_mut(), ver)?;
        r.b()?; // was a zombie
        let item_class = r.bs()? as u16;
        if ver >= Version::R2004 {
            for _ in 0..5 {
                r.bl()?; // instance count, dwg version, maintenance version, 2 unknown
            }
        }
        out.push(Class { number, dxf_name, cpp_name, is_entity: item_class == 0x1F2 });
    }
    Ok(out)
}
