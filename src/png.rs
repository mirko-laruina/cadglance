//! Minimal PNG writer (RGBA8, zlib "stored" blocks: no compression, no dependencies).

fn crc32(data: &[u8], mut c: u32) -> u32 {
    for &b in data {
        c ^= b as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
        }
    }
    c
}

fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let c = crc32(data, crc32(kind, 0xFFFF_FFFF)) ^ 0xFFFF_FFFF;
    out.extend_from_slice(&c.to_be_bytes());
}

/// Encode `rgba` (width * height * 4 bytes, rows top to bottom) as PNG.
pub fn encode_rgba(width: u32, height: u32, rgba: &[u8]) -> Vec<u8> {
    assert_eq!(rgba.len(), width as usize * height as usize * 4, "rgba size");
    let row = width as usize * 4;
    // filter byte 0 before each row
    let mut raw = Vec::with_capacity((row + 1) * height as usize);
    for r in rgba.chunks_exact(row) {
        raw.push(0);
        raw.extend_from_slice(r);
    }
    let mut z = vec![0x78, 0x01];
    let (mut a, mut b) = (1u32, 0u32);
    for x in &raw {
        a = (a + *x as u32) % 65521;
        b = (b + a) % 65521;
    }
    let blocks: Vec<&[u8]> = raw.chunks(65535).collect();
    for (i, blk) in blocks.iter().enumerate() {
        z.push(u8::from(i + 1 == blocks.len()));
        let n = blk.len() as u16;
        z.extend_from_slice(&n.to_le_bytes());
        z.extend_from_slice(&(!n).to_le_bytes());
        z.extend_from_slice(blk);
    }
    z.extend_from_slice(&((b << 16) | a).to_be_bytes());

    let mut out = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.extend_from_slice(&[8, 6, 0, 0, 0]); // 8 bit, RGBA, deflate, adaptive filter, no interlace
    chunk(&mut out, b"IHDR", &ihdr);
    chunk(&mut out, b"IDAT", &z);
    chunk(&mut out, b"IEND", &[]);
    out
}

#[cfg(test)]
mod tests {
    #[test]
    fn crc_and_layout() {
        // CRC-32 of "IEND" is the well-known AE 42 60 82
        assert_eq!(super::crc32(b"IEND", 0xFFFF_FFFF) ^ 0xFFFF_FFFF, 0xAE42_6082);
        let png = super::encode_rgba(2, 1, &[255, 0, 0, 255, 0, 255, 0, 255]);
        assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n");
        assert_eq!(&png[png.len() - 8..png.len() - 4], b"IEND");
    }
}
