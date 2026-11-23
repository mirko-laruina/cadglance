//! AcDb:Handles, the object map (ODA spec ch. 23): handle -> offset in AcDb:AcDbObjects.

use crate::bits::Bits;
use crate::container::{Error, Result, err};

/// Returns (handle, offset in bytes into AcDb:AcDbObjects), in file order.
pub fn parse_object_map(data: &[u8]) -> Result<Vec<(u64, usize)>> {
    let mut out = Vec::new();
    let mut o = 0usize;
    loop {
        let size = data
            .get(o..o + 2)
            .map(|s| u16::from_be_bytes([s[0], s[1]]) as usize)
            .ok_or_else(|| Error("object map truncated".into()))?;
        if size == 2 {
            return Ok(out);
        }
        if size < 2 || o + size > data.len() {
            return err(format!("object map: bad block size {size} at {o}"));
        }
        // The size counts the 2 size bytes but not the 2 CRC bytes after the block.
        let block = &data[o + 2..o + size];
        let mut r = Bits::new(block);
        // Deltas restart from 0 in every block (observed on AC1032; the spec does not say).
        let (mut handle, mut loc) = (0u64, 0i64);
        while r.pos < r.end {
            handle += r.umc()?;
            loc += r.mc()?;
            if loc < 0 {
                return err(format!("object map: negative offset for handle {handle:#x}"));
            }
            out.push((handle, loc as usize));
        }
        o += size + 2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_blocks_with_deltas() {
        // block 1: (1, 0x10), (+2, +0x20); block 2 restarts from 0: (5, 0x30), (+1, -0x05); end block (size 2)
        let mut d = vec![0x00, 0x06, 0x01, 0x10, 0x02, 0x20, 0xAA, 0xBB];
        d.extend_from_slice(&[0x00, 0x06, 0x05, 0x30, 0x01, 0x45, 0xAA, 0xBB]);
        d.extend_from_slice(&[0x00, 0x02, 0xAA, 0xBB]);
        assert_eq!(parse_object_map(&d).unwrap(), vec![(1, 0x10), (3, 0x30), (5, 0x30), (6, 0x2B)]);
    }
}
