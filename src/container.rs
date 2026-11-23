//! File container. R2004+ (ODA spec ch. 4): encrypted file header, page map, section map and
//! compressed data pages. R2000 (spec ch. 3): section locators to plain sections, no compression.

use std::collections::HashMap;
use std::fmt;

use crate::compress::decompress_into;

#[derive(Debug)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn err<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error(msg.into()))
}

/// File format versions this reader knows, in release order. R2004/R2007 are not supported, so
/// a field marked "R2004+" or "R2007+" in the spec applies to R2010 and later here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Version {
    R2000,
    R2010,
    R2013,
    R2018,
}

const R2000_HEADER_SENTINEL: [u8; 16] =
    [0x95, 0xA0, 0x4E, 0x28, 0x99, 0x82, 0x1A, 0xE5, 0x5E, 0x41, 0xE0, 0x5F, 0x9D, 0x3A, 0x4D, 0x00];

const PAGE_MAP_TYPE: u32 = 0x4163_0E3B;
const SECTION_MAP_TYPE: u32 = 0x4163_003B;
const DATA_PAGE_TYPE: u32 = 0x4163_043B;
const DATA_PAGE_MASK: u32 = 0x4164_536B;

fn u32le(b: &[u8], o: usize) -> Result<u32> {
    b.get(o..o + 4)
        .map(|s| u32::from_le_bytes(s.try_into().unwrap()))
        .ok_or_else(|| Error(format!("read u32 out of bounds at {o:#x}")))
}

fn i32le(b: &[u8], o: usize) -> Result<i32> {
    u32le(b, o).map(|v| v as i32)
}

fn u64le(b: &[u8], o: usize) -> Result<u64> {
    b.get(o..o + 8)
        .map(|s| u64::from_le_bytes(s.try_into().unwrap()))
        .ok_or_else(|| Error(format!("read u64 out of bounds at {o:#x}")))
}

/// One logical section from the section map.
#[derive(Debug, Clone)]
pub struct SectionInfo {
    pub name: String,
    pub size: u64,
    pub max_page_size: u64,
    pub compressed: bool,
    pub encrypted: bool,
    /// (page number, start offset in the logical section)
    pub pages: Vec<(i32, u64)>,
}

pub struct Container<'a> {
    pub raw: &'a [u8],
    pub version: String,
    pub maint_version: u8,
    pub ver: Version,
    pub sections: Vec<SectionInfo>,
    page_addr: HashMap<i32, usize>,
    /// R2000 only: section name -> (file offset, size) from the section locators.
    plain: HashMap<String, (usize, usize)>,
}

impl<'a> Container<'a> {
    pub fn parse(raw: &'a [u8]) -> Result<Self> {
        if raw.len() < 0x100 {
            return err("file too small");
        }
        let version = String::from_utf8_lossy(&raw[..6]).into_owned();
        let ver = match version.as_str() {
            "AC1015" => Version::R2000,
            "AC1024" => Version::R2010,
            "AC1027" => Version::R2013,
            "AC1032" => Version::R2018,
            _ => return err(format!("unsupported version {version} (supported: AC1015, AC1024, AC1027, AC1032)")),
        };
        let maint_version = raw[0x0B];
        if ver == Version::R2000 {
            let plain = r2000_locators(raw)?;
            return Ok(Container { raw, version, maint_version, ver, sections: Vec::new(), page_addr: HashMap::new(), plain });
        }

        let mut hdr = [0u8; 0x6C];
        hdr.copy_from_slice(&raw[0x80..0x80 + 0x6C]);
        let mut seed: u32 = 1;
        for b in hdr.iter_mut() {
            seed = seed.wrapping_mul(0x343FD).wrapping_add(0x269EC3);
            *b ^= (seed >> 16) as u8;
        }
        if &hdr[..11] != b"AcFssFcAJMB" {
            return err("bad R2004 file header signature");
        }
        let page_map_addr = u64le(&hdr, 0x54)? as usize + 0x100;
        let section_map_id = i32le(&hdr, 0x5C)?;

        let page_map = system_page(raw, page_map_addr, PAGE_MAP_TYPE)?;
        let mut page_addr = HashMap::new();
        let (mut addr, mut o) = (0x100usize, 0usize);
        while o + 8 <= page_map.len() {
            let (num, size) = (i32le(&page_map, o)?, i32le(&page_map, o + 4)?);
            o += 8;
            if num < 0 {
                o += 16; // gap entry: parent, left, right, 0
            } else {
                page_addr.insert(num, addr);
            }
            addr += size as usize;
        }

        let sm_addr = *page_addr
            .get(&section_map_id)
            .ok_or_else(|| Error("section map page not in page map".into()))?;
        let sm = system_page(raw, sm_addr, SECTION_MAP_TYPE)?;
        let ndesc = i32le(&sm, 0)?;
        let mut o = 20;
        let mut sections = Vec::new();
        for _ in 0..ndesc {
            let size = u64le(&sm, o)?;
            let npages = i32le(&sm, o + 8)? as usize;
            let max_page_size = u32le(&sm, o + 12)? as u64;
            let compressed = i32le(&sm, o + 20)? == 2;
            let encrypted = i32le(&sm, o + 28)? != 0;
            let name_bytes = sm
                .get(o + 32..o + 96)
                .ok_or_else(|| Error("section map truncated".into()))?;
            let name = name_bytes.iter().take_while(|&&c| c != 0).map(|&c| c as char).collect();
            o += 96;
            let mut pages = Vec::with_capacity(npages);
            for k in 0..npages {
                pages.push((i32le(&sm, o + 16 * k)?, u64le(&sm, o + 16 * k + 8)?));
            }
            o += 16 * npages;
            sections.push(SectionInfo { name, size, max_page_size, compressed, encrypted, pages });
        }

        Ok(Container { raw, version, maint_version, ver, sections, page_addr, plain: HashMap::new() })
    }

    pub fn section_info(&self, name: &str) -> Option<&SectionInfo> {
        self.sections.iter().find(|s| s.name == name)
    }

    /// Decompress a whole logical section. Pages are decompressed on `threads` threads.
    /// R2000: "AcDb:AcDbObjects" is the whole file, because object map offsets are file offsets.
    pub fn section(&self, name: &str, threads: usize) -> Result<Vec<u8>> {
        if self.ver == Version::R2000 {
            if name == "AcDb:AcDbObjects" {
                // shortcut: copies the whole file; switch to a borrowed slice if memory matters on big R2000 files.
                return Ok(self.raw.to_vec());
            }
            let &(at, size) = self.plain.get(name).ok_or_else(|| Error(format!("section {name} not found")))?;
            return self
                .raw
                .get(at..at + size)
                .map(<[u8]>::to_vec)
                .ok_or_else(|| Error(format!("section {name} past end of file")));
        }
        let info = self
            .section_info(name)
            .ok_or_else(|| Error(format!("section {name} not found")))?;
        if info.encrypted {
            return err(format!("section {name} is encrypted (password protected files are not supported)"));
        }
        let mut pages = info.pages.clone();
        pages.sort_by_key(|p| p.1);
        // The last page may decompress past the logical size (padding): give it a full page, truncate after.
        let padded = pages.last().map_or(0, |p| p.1 + info.max_page_size).max(info.size);
        let mut out = vec![0u8; padded as usize];

        // Split the output into disjoint slices, one per page, so threads can write in place.
        let mut jobs: Vec<(i32, &mut [u8])> = Vec::with_capacity(pages.len());
        let mut rest: &mut [u8] = &mut out;
        let mut cursor = 0u64;
        for (k, &(num, start)) in pages.iter().enumerate() {
            if start != cursor {
                return err(format!("section {name}: page {num} starts at {start}, expected {cursor}"));
            }
            let end = pages.get(k + 1).map_or(padded, |p| p.1);
            let (head, tail) = std::mem::take(&mut rest).split_at_mut((end - start) as usize);
            jobs.push((num, head));
            rest = tail;
            cursor = end;
        }

        let threads = threads.max(1).min(jobs.len().max(1));
        let chunk = jobs.len().div_ceil(threads).max(1);
        let compressed = info.compressed;
        std::thread::scope(|s| -> Result<()> {
            let handles: Vec<_> = jobs
                .chunks_mut(chunk)
                .map(|group| {
                    s.spawn(move || -> Result<()> {
                        for (num, dst) in group.iter_mut() {
                            self.data_page(*num, compressed, dst)?;
                        }
                        Ok(())
                    })
                })
                .collect();
            for h in handles {
                h.join().map_err(|_| Error("decompression thread panicked".into()))??;
            }
            Ok(())
        })?;
        drop(jobs);
        out.truncate(info.size as usize);
        Ok(out)
    }

    fn data_page(&self, num: i32, compressed: bool, dst: &mut [u8]) -> Result<()> {
        let a = *self
            .page_addr
            .get(&num)
            .ok_or_else(|| Error(format!("data page {num} not in page map")))?;
        let mut h = [0u32; 8];
        for (k, w) in h.iter_mut().enumerate() {
            *w = u32le(self.raw, a + 4 * k)? ^ (DATA_PAGE_MASK ^ a as u32);
        }
        if h[0] != DATA_PAGE_TYPE {
            return err(format!("data page {num}: bad page type {:#x}", h[0]));
        }
        let csize = h[2] as usize;
        let src = self
            .raw
            .get(a + 32..a + 32 + csize)
            .ok_or_else(|| Error(format!("data page {num} truncated")))?;
        if compressed {
            decompress_into(src, dst)?;
        } else {
            let n = dst.len().min(src.len());
            dst[..n].copy_from_slice(&src[..n]);
        }
        Ok(())
    }
}

/// R2000 section locators at 0x15 (spec 3.2.6): record number, seeker, size.
fn r2000_locators(raw: &[u8]) -> Result<HashMap<String, (usize, usize)>> {
    let n = u32le(raw, 0x15)? as usize;
    if !(3..=6).contains(&n) {
        return err(format!("R2000 header: {n} section locators"));
    }
    let mut out = HashMap::new();
    for k in 0..n {
        let o = 0x19 + 9 * k;
        let num = *raw.get(o).ok_or_else(|| Error("R2000 header truncated".into()))?;
        let (at, size) = (u32le(raw, o + 1)? as usize, u32le(raw, o + 5)? as usize);
        let name = match num {
            0 => "AcDb:Header",
            1 => "AcDb:Classes",
            2 => "AcDb:Handles",
            _ => continue,
        };
        out.insert(name.to_string(), (at, size));
    }
    let end = 0x19 + 9 * n;
    let stored = raw.get(end..end + 2).map(|s| u16::from_le_bytes([s[0], s[1]])).unwrap_or(0);
    let xor = [0xA598, 0x8101, 0x3CC4, 0x8461][n - 3];
    if crate::crc::crc8(0, &raw[..end]) ^ xor != stored {
        return err("R2000 header: bad CRC");
    }
    if raw.get(end + 2..end + 18) != Some(&R2000_HEADER_SENTINEL[..]) {
        return err("R2000 header: bad sentinel");
    }
    for name in ["AcDb:Classes", "AcDb:Handles"] {
        if !out.contains_key(name) {
            return err(format!("R2000 header: no locator for {name}"));
        }
    }
    Ok(out)
}

fn system_page(raw: &[u8], addr: usize, expected_type: u32) -> Result<Vec<u8>> {
    let typ = u32le(raw, addr)?;
    if typ != expected_type {
        return err(format!("system page at {addr:#x}: type {typ:#x}, expected {expected_type:#x}"));
    }
    let dsize = u32le(raw, addr + 4)? as usize;
    let csize = u32le(raw, addr + 8)? as usize;
    let ctype = u32le(raw, addr + 12)?;
    let src = raw
        .get(addr + 20..addr + 20 + csize)
        .ok_or_else(|| Error(format!("system page at {addr:#x} truncated")))?;
    if ctype != 2 {
        return Ok(src.to_vec());
    }
    let mut out = vec![0u8; dsize];
    let n = decompress_into(src, &mut out)?;
    if n != dsize {
        return err(format!("system page at {addr:#x}: decompressed {n} B, expected {dsize}"));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal R2000 header: 3 locators (header, classes, object map), CRC, sentinel.
    fn r2000_header() -> Vec<u8> {
        let mut d = b"AC1015".to_vec();
        d.resize(0x15, 0);
        d.extend(3u32.to_le_bytes());
        for (num, at, size) in [(0u8, 0x100u32, 0x40u32), (1, 0x200, 0x30), (2, 0x300, 0x20)] {
            d.push(num);
            d.extend(at.to_le_bytes());
            d.extend(size.to_le_bytes());
        }
        let crc = crate::crc::crc8(0, &d) ^ 0xA598;
        d.extend(crc.to_le_bytes());
        d.extend(R2000_HEADER_SENTINEL);
        d
    }

    #[test]
    fn r2000_section_locators() {
        let m = r2000_locators(&r2000_header()).unwrap();
        assert_eq!(m["AcDb:Classes"], (0x200, 0x30));
        assert_eq!(m["AcDb:Handles"], (0x300, 0x20));
    }

    #[test]
    fn r2000_bad_crc() {
        let mut d = r2000_header();
        d[0x1A] ^= 1;
        assert!(r2000_locators(&d).is_err());
    }
}
