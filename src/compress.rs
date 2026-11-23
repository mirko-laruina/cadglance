//! R2004+ LZ77 variant (ODA spec 4.7).

use crate::container::{Error, Result};

struct Src<'a> {
    b: &'a [u8],
    i: usize,
}

impl Src<'_> {
    #[inline]
    fn byte(&mut self) -> Result<u8> {
        let v = *self.b.get(self.i).ok_or_else(|| Error("compressed stream truncated".into()))?;
        self.i += 1;
        Ok(v)
    }

    fn literal_len(&mut self, op: u8) -> Result<usize> {
        let mut n = (op & 0x0F) as usize;
        if n == 0 {
            n = 0x0F;
            let mut b = self.byte()?;
            while b == 0 {
                n += 0xFF;
                b = self.byte()?;
            }
            n += b as usize;
        }
        Ok(n + 3)
    }

    fn long_len(&mut self) -> Result<usize> {
        let mut n = 0usize;
        let mut b = self.byte()?;
        if b == 0 {
            n = 0xFF;
            b = self.byte()?;
            while b == 0 {
                n += 0xFF;
                b = self.byte()?;
            }
        }
        Ok(n + b as usize)
    }

    /// Returns (offset, literal count).
    fn two_byte_offset(&mut self) -> Result<(usize, usize)> {
        let b1 = self.byte()? as usize;
        let b2 = self.byte()? as usize;
        Ok(((b1 >> 2) | (b2 << 6), b1 & 3))
    }
}

/// Decompress `src` into `dst`; returns the number of bytes written.
/// Fails if the output would not fit in `dst`.
pub fn decompress_into(src: &[u8], dst: &mut [u8]) -> Result<usize> {
    let mut s = Src { b: src, i: 0 };
    let mut o = 0usize;

    fn literal(s: &mut Src, dst: &mut [u8], o: &mut usize, n: usize) -> Result<()> {
        let from = s.b.get(s.i..s.i + n).ok_or_else(|| Error("literal run truncated".into()))?;
        dst.get_mut(*o..*o + n)
            .ok_or_else(|| Error("decompressed data larger than page".into()))?
            .copy_from_slice(from);
        s.i += n;
        *o += n;
        Ok(())
    }

    let mut op = s.byte()?;
    if op & 0xF0 == 0 {
        let n = s.literal_len(op)?;
        literal(&mut s, dst, &mut o, n)?;
        op = s.byte()?;
    }
    loop {
        let (count, offset, lit) = match op {
            0x11 => break,
            // bit 3 adds 0x4000 to the offset; length 0 in bits 0-2 means a long length
            0x10..=0x1F => {
                let count = match (op & 7) as usize {
                    0 => s.long_len()? + 9,
                    n => n + 2,
                };
                let (off, lit) = s.two_byte_offset()?;
                (count, off + (((op & 8) as usize) << 11) + 0x3FFF, lit)
            }
            0x20 => {
                let count = s.long_len()? + 0x21;
                let (off, lit) = s.two_byte_offset()?;
                (count, off, lit)
            }
            0x21..=0x3F => {
                let (off, lit) = s.two_byte_offset()?;
                (op as usize - 0x1E, off, lit)
            }
            0x40..=0xFF => {
                let count = (op >> 4) as usize - 1;
                let off = ((s.byte()? as usize) << 2) | ((op & 0x0C) >> 2) as usize;
                (count, off, (op & 3) as usize)
            }
            _ => return Err(Error(format!("bad compression opcode {op:#x} at {}", s.i - 1))),
        };
        if offset + 1 > o || o + count > dst.len() {
            return Err(Error(format!("bad back-reference (offset {offset}, count {count}) at output {o} of {}", dst.len())));
        }
        // Byte by byte: source and destination may overlap.
        for k in 0..count {
            dst[o + k] = dst[o + k - offset - 1];
        }
        o += count;
        if lit == 0 {
            op = s.byte()?;
            if op < 0x10 {
                let n = s.literal_len(op)?;
                literal(&mut s, dst, &mut o, n)?;
                op = s.byte()?;
            }
        } else {
            literal(&mut s, dst, &mut o, lit)?;
            op = s.byte()?;
        }
    }
    Ok(o)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(src: &[u8], cap: usize) -> Vec<u8> {
        let mut out = vec![0u8; cap];
        let n = decompress_into(src, &mut out).unwrap();
        out.truncate(n);
        out
    }

    #[test]
    fn literal_only() {
        // op 0x02 -> literal of 2+3 = 5 bytes, then 0x11 end
        assert_eq!(run(&[0x02, b'a', b'b', b'c', b'd', b'e', 0x11], 16), b"abcde");
    }

    #[test]
    fn short_back_reference() {
        // "abcd" literal, then op 0x22: count 4, two-byte offset 3 (b1 = 3<<2), lit 0 -> repeats "abcd"
        let out = run(&[0x01, b'a', b'b', b'c', b'd', 0x22, 0x0C, 0x00, 0x11], 16);
        assert_eq!(out, b"abcdabcd");
    }

    #[test]
    fn overlapping_back_reference() {
        // "abcd" literal, then op 0x50: count 5-1 = 4, offset 0 (repeat last byte), lit 0
        assert_eq!(run(&[0x01, b'a', b'b', b'c', b'd', 0x50, 0x00, 0x11], 16), b"abcddddd");
    }

    #[test]
    fn far_offset_uses_bit3() {
        // 0x19: count (1)+2 = 3, offset = two-byte + 0x4000 + 0x3FFF.
        let mut src = Vec::new();
        let lit_len = 0x8000usize;
        // literal_len(0) = 0x0F + 0xFF*k + b + 3
        let mut rem = lit_len - 3 - 0x0F;
        src.push(0x00);
        while rem > 0xFF {
            src.push(0x00);
            rem -= 0xFF;
        }
        src.push(rem as u8);
        let data: Vec<u8> = (0..lit_len).map(|i| (i % 251) as u8).collect();
        src.extend_from_slice(&data);
        // copy 3 bytes from distance 0x8000 (offset 0x7FFF = 0x3FFF + 0x4000 + 0)
        src.extend_from_slice(&[0x19, 0x00, 0x00, 0x11]);
        let out = run(&src, lit_len + 3);
        assert_eq!(&out[lit_len..], &data[..3]);
    }

    #[test]
    fn rejects_bad_back_reference() {
        let mut out = [0u8; 8];
        assert!(decompress_into(&[0x22, 0x0C, 0x00, 0x11], &mut out).is_err());
    }
}
