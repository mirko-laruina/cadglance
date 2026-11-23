//! R2007 (AC1021) page decoding: Reed-Solomon de-interleaving and the R21 LZ77 variant (ODA spec 5.10, 5.13).
//! Parity bytes are skipped, not checked: a damaged file gives a decode error further on.

use crate::container::{Error, Result, err};

/// Data bytes of `blocks` RS codewords (255 bytes, `k` data bytes each). Interleaved: byte i of block j
/// is at i * blocks + j. Not interleaved: all data blocks first, then the parity blocks.
pub fn rs_data(src: &[u8], blocks: usize, k: usize, interleaved: bool) -> Result<Vec<u8>> {
    if src.len() < blocks * 255 {
        return err(format!("RS data: {} bytes, need {}", src.len(), blocks * 255));
    }
    if !interleaved {
        return Ok(src[..blocks * k].to_vec());
    }
    let mut out = Vec::with_capacity(blocks * k);
    for j in 0..blocks {
        for i in 0..k {
            out.push(src[i * blocks + j]);
        }
    }
    Ok(out)
}

/// Literal copy order for 1..=32 bytes (spec 5.10.1): (size, source offset) parts, applied recursively.
const PARTS: [&[(usize, usize)]; 33] = [
    &[],
    &[(1, 0)],
    &[(1, 1), (1, 0)],
    &[(1, 2), (1, 1), (1, 0)],
    &[(1, 0), (1, 1), (1, 2), (1, 3)],
    &[(1, 4), (4, 0)],
    &[(1, 5), (4, 1), (1, 0)],
    &[(2, 5), (4, 1), (1, 0)],
    &[(4, 0), (4, 4)],
    &[(1, 8), (8, 0)],
    &[(1, 9), (8, 1), (1, 0)],
    &[(2, 9), (8, 1), (1, 0)],
    &[(4, 8), (8, 0)],
    &[(1, 12), (4, 8), (8, 0)],
    &[(1, 13), (4, 9), (8, 1), (1, 0)],
    &[(2, 13), (4, 9), (8, 1), (1, 0)],
    &[(8, 8), (8, 0)],
    &[(8, 9), (1, 8), (8, 0)],
    &[(1, 17), (16, 1), (1, 0)],
    &[(3, 16), (16, 0)],
    &[(4, 16), (16, 0)],
    &[(1, 20), (4, 16), (16, 0)],
    &[(2, 20), (4, 16), (16, 0)],
    &[(3, 20), (4, 16), (16, 0)],
    &[(8, 16), (16, 0)],
    &[(8, 17), (1, 16), (16, 0)],
    &[(1, 25), (8, 17), (1, 16), (16, 0)],
    &[(2, 25), (8, 17), (1, 16), (16, 0)],
    &[(4, 24), (8, 16), (16, 0)],
    &[(1, 28), (4, 24), (8, 16), (16, 0)],
    &[(2, 28), (4, 24), (8, 16), (16, 0)],
    &[(1, 30), (4, 26), (8, 18), (16, 2), (2, 0)],
    &[(16, 16), (16, 0)],
];

fn order(n: usize, base: usize, out: &mut Vec<usize>) {
    if n == 1 {
        out.push(base);
        return;
    }
    for &(size, off) in PARTS[n] {
        order(size, base + off, out);
    }
}

fn literal(src: &[u8], i: usize, len: usize, dst: &mut Vec<u8>) -> Result<()> {
    let s = src.get(i..i + len).ok_or_else(|| Error("R21 decompress: literal past end".into()))?;
    let mut idx = Vec::with_capacity(32);
    let mut p = 0;
    while p < len {
        let n = (len - p).min(32);
        idx.clear();
        order(n, 0, &mut idx);
        dst.extend(idx.iter().map(|&k| s[p + k]));
        p += n;
    }
    Ok(())
}

struct Src<'a> {
    b: &'a [u8],
    i: usize,
}

impl Src<'_> {
    fn u8(&mut self) -> Result<usize> {
        let v = *self.b.get(self.i).ok_or_else(|| Error("R21 decompress: input truncated".into()))?;
        self.i += 1;
        Ok(v as usize)
    }

    /// Literal length for opcode `op` (spec 5.10.1).
    fn literal_len(&mut self, op: usize) -> Result<usize> {
        let mut len = op + 8;
        if len == 0x17 {
            let n = self.u8()?;
            len += n;
            if n == 0xFF {
                loop {
                    let n = self.u8()? | (self.u8()? << 8);
                    len += n;
                    if n != 0xFFFF {
                        break;
                    }
                }
            }
        }
        Ok(len)
    }

    /// Back-reference (offset, length) for opcode `op`; returns the next opcode too (spec 5.10.2).
    fn instruction(&mut self, op: usize) -> Result<(usize, usize, usize)> {
        Ok(match op >> 4 {
            0 => {
                let mut len = (op & 0xF) + 0x13;
                let off = self.u8()?;
                let op = self.u8()?;
                len += (op >> 3) & 0x10;
                (((op & 0x78) << 5) + 1 + off, len, op)
            }
            1 => {
                let len = (op & 0xF) + 3;
                let off = self.u8()?;
                let op = self.u8()?;
                (((op & 0xF8) << 5) + 1 + off, len, op)
            }
            2 => {
                let mut off = self.u8()?;
                off |= self.u8()? << 8;
                let mut len = op & 7;
                let op = if op & 8 == 0 {
                    let op = self.u8()?;
                    len += op & 0xF8;
                    op
                } else {
                    off += 1;
                    len += self.u8()? << 3;
                    let op = self.u8()?;
                    len += ((op & 0xF8) << 8) + 0x100;
                    op
                };
                (off, len, op)
            }
            _ => {
                let len = op >> 4;
                let off = op & 15;
                let op = self.u8()?;
                (((op & 0xF8) << 1) + off + 1, len, op)
            }
        })
    }
}

/// Decompress `src` into exactly `size` bytes.
pub fn decompress21(src: &[u8], size: usize) -> Result<Vec<u8>> {
    let mut s = Src { b: src, i: 0 };
    let mut dst = Vec::with_capacity(size);
    let mut op = s.u8()?;
    let mut lit = 0;
    if op >> 4 == 2 {
        // opcode, 2 skipped bytes, then the first literal length (checked on AC1021 file headers)
        s.i = 3;
        lit = s.u8()? & 7;
    }
    loop {
        if lit == 0 {
            lit = s.literal_len(op)?;
        }
        literal(src, s.i, lit, &mut dst)?;
        s.i += lit;
        if s.i >= src.len() {
            break;
        }
        // back-references until the opcode announces the next literal
        op = s.u8()?;
        let (mut off, mut len, mut next) = s.instruction(op)?;
        loop {
            if off == 0 || off > dst.len() {
                return err(format!("R21 decompress: bad offset {off} at output {}", dst.len()));
            }
            let from = dst.len() - off;
            for k in 0..len {
                dst.push(dst[from + k]);
            }
            lit = next & 7;
            if lit != 0 || s.i >= src.len() {
                break;
            }
            op = s.u8()?;
            if op >> 4 == 0 {
                break;
            }
            if op >> 4 == 15 {
                op &= 15;
            }
            (off, len, next) = s.instruction(op)?;
        }
        if s.i >= src.len() {
            break;
        }
    }
    if dst.len() < size {
        return err(format!("R21 decompress: {} bytes, expected {size}", dst.len()));
    }
    dst.truncate(size);
    Ok(dst)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literal_order_is_a_permutation() {
        for n in 1..=32 {
            let mut v = Vec::new();
            order(n, 0, &mut v);
            v.sort();
            assert_eq!(v, (0..n).collect::<Vec<_>>(), "n = {n}");
        }
    }

    #[test]
    fn rs_interleaved() {
        // 2 blocks: block 0 = 0..255, block 1 = 100..355 (mod 256), interleaved byte by byte
        let mut src = vec![0u8; 510];
        for i in 0..255 {
            src[2 * i] = i as u8;
            src[2 * i + 1] = (i + 100) as u8;
        }
        let d = rs_data(&src, 2, 239, true).unwrap();
        assert_eq!(d.len(), 478);
        assert_eq!((d[0], d[238], d[239], d[240]), (0, 238, 100, 101));
    }
}
