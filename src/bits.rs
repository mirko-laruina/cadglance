//! Bit-level reader for the DWG bit codes (ODA spec ch. 2). Bits are read MSB first.

use crate::container::{Error, Result};

#[derive(Clone)]
pub struct Bits<'a> {
    data: &'a [u8],
    /// Current position in bits from the start of `data`.
    pub pos: usize,
    /// Exclusive end in bits; reads past it fail.
    pub end: usize,
}

/// Handle reference: CODE.COUNTER.BYTES (spec 2.13).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HandleRef {
    pub code: u8,
    pub value: u64,
}

impl HandleRef {
    /// Absolute handle; `owner` is the handle of the object that holds the reference.
    pub fn absolute(self, owner: u64) -> u64 {
        match self.code {
            0x6 => owner.wrapping_add(1),
            0x8 => owner.wrapping_sub(1),
            0xA => owner.wrapping_add(self.value),
            0xC => owner.wrapping_sub(self.value),
            _ => self.value,
        }
    }
}

impl<'a> Bits<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Bits { data, pos: 0, end: data.len() * 8 }
    }

    pub fn at(data: &'a [u8], pos: usize, end: usize) -> Self {
        Bits { data, pos, end: end.min(data.len() * 8) }
    }

    fn eof(&self, n: usize) -> Error {
        Error(format!("bit stream overrun: need {n} bits at {} (end {})", self.pos, self.end))
    }

    #[inline]
    pub fn b(&mut self) -> Result<bool> {
        if self.pos >= self.end {
            return Err(self.eof(1));
        }
        let v = (self.data[self.pos >> 3] >> (7 - (self.pos & 7))) & 1;
        self.pos += 1;
        Ok(v == 1)
    }

    /// Read `n` (<= 8) bits as an unsigned value, MSB first.
    #[inline]
    fn bits(&mut self, n: usize) -> Result<u8> {
        if self.pos + n > self.end {
            return Err(self.eof(n));
        }
        let byte = self.pos >> 3;
        let shift = self.pos & 7;
        let hi = (self.data[byte] as u16) << 8;
        let lo = self.data.get(byte + 1).copied().unwrap_or(0) as u16;
        let v = ((hi | lo) << shift) >> (16 - n);
        self.pos += n;
        Ok(v as u8)
    }

    pub fn bb(&mut self) -> Result<u8> {
        self.bits(2)
    }

    /// 3B: 1-3 bits, stop at the first zero bit.
    pub fn b3(&mut self) -> Result<u8> {
        let mut v = 0u8;
        for _ in 0..3 {
            let bit = self.b()?;
            v = (v << 1) | bit as u8;
            if !bit {
                break;
            }
        }
        Ok(v)
    }

    pub fn rc(&mut self) -> Result<u8> {
        self.bits(8)
    }

    pub fn rs(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes([self.rc()?, self.rc()?]))
    }

    pub fn rl(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes([self.rc()?, self.rc()?, self.rc()?, self.rc()?]))
    }

    pub fn rd(&mut self) -> Result<f64> {
        let mut b = [0u8; 8];
        for x in b.iter_mut() {
            *x = self.rc()?;
        }
        Ok(f64::from_le_bytes(b))
    }

    pub fn rd2(&mut self) -> Result<[f64; 2]> {
        Ok([self.rd()?, self.rd()?])
    }

    pub fn skip_bytes(&mut self, n: usize) -> Result<()> {
        if self.pos + n * 8 > self.end {
            return Err(self.eof(n * 8));
        }
        self.pos += n * 8;
        Ok(())
    }

    pub fn bs(&mut self) -> Result<i16> {
        Ok(match self.bb()? {
            0 => self.rs()? as i16,
            1 => self.rc()? as i16,
            2 => 0,
            _ => 256,
        })
    }

    pub fn bl(&mut self) -> Result<i32> {
        match self.bb()? {
            0 => Ok(self.rl()? as i32),
            1 => Ok(self.rc()? as i32),
            2 => Ok(0),
            _ => Err(Error(format!("invalid BL code 11 at bit {}", self.pos - 2))),
        }
    }

    /// BL used as an element count: rejects negative values and counts larger than
    /// `bits_per_item` times the bits left in `budget_bits`.
    pub fn count_in(&mut self, bits_per_item: usize, budget_bits: usize) -> Result<usize> {
        let v = self.bl()?;
        if v < 0 || (v as usize).saturating_mul(bits_per_item) > budget_bits {
            return Err(Error(format!("implausible count {v} at bit {}", self.pos)));
        }
        Ok(v as usize)
    }

    /// BL count of items stored in this same stream (each item takes at least one bit).
    pub fn count(&mut self) -> Result<usize> {
        let left = self.end.saturating_sub(self.pos);
        self.count_in(1, left)
    }

    pub fn remaining(&self) -> usize {
        self.end.saturating_sub(self.pos)
    }

    /// BLL: byte count in 3 fixed bits, then that many bytes LSB first. The spec says the count is a
    /// 3B (1-3 bits); AC1032 proxy-graphics sizes only decode with 3 fixed bits.
    pub fn bll(&mut self) -> Result<u64> {
        let n = self.bits(3)?;
        let mut v = 0u64;
        for k in 0..n {
            v |= (self.rc()? as u64) << (8 * k);
        }
        Ok(v)
    }

    pub fn bd(&mut self) -> Result<f64> {
        match self.bb()? {
            0 => self.rd(),
            1 => Ok(1.0),
            2 => Ok(0.0),
            _ => Err(Error(format!("invalid BD code 11 at bit {}", self.pos - 2))),
        }
    }

    pub fn bd2(&mut self) -> Result<[f64; 2]> {
        Ok([self.bd()?, self.bd()?])
    }

    pub fn bd3(&mut self) -> Result<[f64; 3]> {
        Ok([self.bd()?, self.bd()?, self.bd()?])
    }

    /// DD: bit double with default (spec 2.9).
    pub fn dd(&mut self, default: f64) -> Result<f64> {
        let mut b = default.to_le_bytes();
        match self.bb()? {
            0 => {}
            1 => {
                for x in b[..4].iter_mut() {
                    *x = self.rc()?;
                }
            }
            2 => {
                b[4] = self.rc()?;
                b[5] = self.rc()?;
                for x in b[..4].iter_mut() {
                    *x = self.rc()?;
                }
            }
            _ => return self.rd(),
        }
        Ok(f64::from_le_bytes(b))
    }

    /// BE: bit extrusion (R2000+).
    pub fn be(&mut self) -> Result<[f64; 3]> {
        if self.b()? { Ok([0.0, 0.0, 1.0]) } else { self.bd3() }
    }

    /// BT: bit thickness (R2000+).
    pub fn bt(&mut self) -> Result<f64> {
        if self.b()? { Ok(0.0) } else { self.bd() }
    }

    /// MC: modular char, signed (0x40 in the last byte negates).
    pub fn mc(&mut self) -> Result<i64> {
        let mut v = 0i64;
        let mut shift = 0;
        loop {
            let b = self.rc()?;
            if b & 0x80 == 0 {
                let neg = b & 0x40 != 0;
                v |= ((b & 0x3F) as i64) << shift;
                return Ok(if neg { -v } else { v });
            }
            v |= ((b & 0x7F) as i64) << shift;
            shift += 7;
            if shift > 63 {
                return Err(Error("MC too long".into()));
            }
        }
    }

    /// MC without sign interpretation (handle stream size, object map handle deltas).
    pub fn umc(&mut self) -> Result<u64> {
        let mut v = 0u64;
        let mut shift = 0;
        loop {
            let b = self.rc()?;
            v |= ((b & 0x7F) as u64) << shift;
            if b & 0x80 == 0 {
                return Ok(v);
            }
            shift += 7;
            if shift > 63 {
                return Err(Error("MC too long".into()));
            }
        }
    }

    /// MS: modular short (object sizes).
    pub fn ms(&mut self) -> Result<u32> {
        let mut v = 0u32;
        let mut shift = 0;
        loop {
            let w = self.rs()?;
            v |= ((w & 0x7FFF) as u32) << shift;
            if w & 0x8000 == 0 {
                return Ok(v);
            }
            shift += 15;
            if shift > 30 {
                return Err(Error("MS too long".into()));
            }
        }
    }

    pub fn h(&mut self) -> Result<HandleRef> {
        let b = self.rc()?;
        let (code, counter) = (b >> 4, b & 0x0F);
        if counter > 8 {
            return Err(Error(format!("handle with {counter} bytes at bit {}", self.pos - 8)));
        }
        let mut value = 0u64;
        for _ in 0..counter {
            value = (value << 8) | self.rc()? as u64;
        }
        Ok(HandleRef { code, value })
    }

    /// OT: object type, R2010+ form (spec 2.12).
    pub fn ot(&mut self) -> Result<u16> {
        Ok(match self.bb()? {
            0 => self.rc()? as u16,
            1 => self.rc()? as u16 + 0x1F0,
            _ => self.rs()?,
        })
    }

    /// TU: BS character count followed by UTF-16LE (R2007+ string stream).
    pub fn tu(&mut self) -> Result<String> {
        let n = self.bs()? as u16 as usize;
        let mut units = Vec::with_capacity(n);
        for _ in 0..n {
            units.push(self.rs()?);
        }
        while units.last() == Some(&0) {
            units.pop();
        }
        Ok(String::from_utf16_lossy(&units))
    }

    pub fn align_byte(&mut self) {
        self.pos = self.pos.div_ceil(8) * 8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pack a string of '0'/'1' (spaces ignored) into bytes, MSB first.
    fn pack(s: &str) -> Vec<u8> {
        let bits: Vec<u8> = s.bytes().filter(|c| *c == b'0' || *c == b'1').map(|c| c - b'0').collect();
        bits.chunks(8)
            .map(|c| c.iter().enumerate().fold(0u8, |acc, (i, b)| acc | (b << (7 - i))))
            .collect()
    }

    #[test]
    fn bitshort_spec_example() {
        // spec 2.2: 00 00000001 00000001 | 10 | 11 | 01 00001111 | 10
        let d = pack("0000000001000000011011010000111110");
        let mut r = Bits::new(&d);
        assert_eq!(r.bs().unwrap(), 257);
        assert_eq!(r.bs().unwrap(), 0);
        assert_eq!(r.bs().unwrap(), 256);
        assert_eq!(r.bs().unwrap(), 15);
        assert_eq!(r.bs().unwrap(), 0);
    }

    #[test]
    fn bitlong_spec_example() {
        let d = pack("000000000100000001000000000000000010010000111110");
        let mut r = Bits::new(&d);
        assert_eq!(r.bl().unwrap(), 257);
        assert_eq!(r.bl().unwrap(), 0);
        assert_eq!(r.bl().unwrap(), 15);
        assert_eq!(r.bl().unwrap(), 0);
    }

    #[test]
    fn modular_char_spec_examples() {
        assert_eq!(Bits::new(&[0b1000_0010, 0b0010_0100]).mc().unwrap(), 4610);
        assert_eq!(Bits::new(&[0b1110_1001, 0b1001_0111, 0b1110_0110, 0b0011_0101]).mc().unwrap(), 112_823_273);
        assert_eq!(Bits::new(&[0b1000_0101, 0b0100_1011]).mc().unwrap(), -1413);
    }

    #[test]
    fn modular_short_spec_example() {
        assert_eq!(Bits::new(&[0b0011_0001, 0b1111_0100, 0b1000_1101, 0b0000_0000]).ms().unwrap(), 4_650_033);
    }

    #[test]
    fn handle_spec_example() {
        // 5.2.05.E7
        let h = Bits::new(&[0b0101_0010, 0x05, 0xE7]).h().unwrap();
        assert_eq!(h, HandleRef { code: 5, value: 0x5E7 });
        assert_eq!(HandleRef { code: 0xA, value: 3 }.absolute(0x10), 0x13);
        assert_eq!(HandleRef { code: 0x8, value: 0 }.absolute(0x10), 0x0F);
    }

    #[test]
    fn bitdouble_and_unaligned_raw() {
        // 01 (1.0) | 10 (0.0) | 00 + RD(2.5) unaligned by 6 bits
        let mut s = String::from("011000");
        for b in 2.5f64.to_le_bytes() {
            s.push_str(&format!("{b:08b}"));
        }
        let d = pack(&s);
        let mut r = Bits::new(&d);
        assert_eq!(r.bd().unwrap(), 1.0);
        assert_eq!(r.bd().unwrap(), 0.0);
        assert_eq!(r.bd().unwrap(), 2.5);
    }

    #[test]
    fn dd_patches_default() {
        // code 01: 4 bytes replace the low 4 bytes of the default
        let def = 1.0f64;
        let mut want = def.to_le_bytes();
        want[..4].copy_from_slice(&[1, 2, 3, 4]);
        let mut s = String::from("01");
        for b in [1u8, 2, 3, 4] {
            s.push_str(&format!("{b:08b}"));
        }
        let d = pack(&s);
        assert_eq!(Bits::new(&d).dd(def).unwrap(), f64::from_le_bytes(want));
    }

    #[test]
    fn overrun_is_an_error() {
        let mut r = Bits::new(&[0xFF]);
        assert!(r.rs().is_err());
    }
}
