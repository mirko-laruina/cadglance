//! 16-bit CRC used after each object (ODA spec 2.14.1; table = CRC-16/ARC, poly 0xA001).

const TABLE: [u16; 256] = {
    let mut t = [0u16; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u16;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 { (c >> 1) ^ 0xA001 } else { c >> 1 };
            k += 1;
        }
        t[i] = c;
        i += 1;
    }
    t
};

pub fn crc8(seed: u16, data: &[u8]) -> u16 {
    data.iter().fold(seed, |dx, &b| (dx >> 8) ^ TABLE[((b as u16 ^ dx) & 0xFF) as usize])
}

#[cfg(test)]
mod tests {
    #[test]
    fn table_matches_spec_prefix() {
        assert_eq!(&super::TABLE[..8], &[0x0000, 0xC0C1, 0xC181, 0x0140, 0xC301, 0x03C0, 0x0280, 0xC241]);
        assert_eq!(super::TABLE[255], 0x4040);
    }
}
