//! R2004+ file container (ODA spec ch. 4): encrypted file header, page map,
//! section map and compressed data pages.

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
    pub sections: Vec<SectionInfo>,
    page_addr: HashMap<i32, usize>,
}

impl<'a> Container<'a> {
    pub fn parse(raw: &'a [u8]) -> Result<Self> {
        if raw.len() < 0x100 {
            return err("file too small");
        }
        let version = String::from_utf8_lossy(&raw[..6]).into_owned();
        if !matches!(version.as_str(), "AC1024" | "AC1027" | "AC1032") {
            return err(format!("unsupported version {version} (only R2010+ supported)"));
        }
        let maint_version = raw[0x0B];

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

        Ok(Container { raw, version, maint_version, sections, page_addr })
    }

    pub fn section_info(&self, name: &str) -> Option<&SectionInfo> {
        self.sections.iter().find(|s| s.name == name)
    }

    /// Decompress a whole logical section. Pages are decompressed on `threads` threads.
    pub fn section(&self, name: &str, threads: usize) -> Result<Vec<u8>> {
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
