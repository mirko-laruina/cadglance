//! Proxy entity graphics (ODA spec ch. 29): the simplified drawing an application stores inside entities
//! that other readers may not know (MULTILEADER, ACAD_TABLE, custom entities). Little-endian bytes,
//! a header (RL total size, RL count) and then chunks of (RL size, RL type, data). 2D only: Z is dropped.

use crate::entities::{P2, Text};
use crate::objects::Color;

#[derive(Debug, Clone)]
pub enum Prim {
    Polyline { pts: Vec<P2>, closed: bool },
    /// Filled polygon (fill was on).
    Fill(Vec<P2>),
    /// Arc from `start` counterclockwise to `end` (radians); a circle is 0..TAU.
    Arc { center: P2, radius: f64, start: f64, end: f64 },
    Text(Text),
}

/// One primitive and the color set before it (None = the entity color).
pub type ProxyPrim = (Option<Color>, Prim);

struct Rd<'a> {
    b: &'a [u8],
    o: usize,
}

impl Rd<'_> {
    fn rl(&mut self) -> Option<u32> {
        let v = u32::from_le_bytes(self.b.get(self.o..self.o + 4)?.try_into().ok()?);
        self.o += 4;
        Some(v)
    }
    fn rd(&mut self) -> Option<f64> {
        let v = f64::from_le_bytes(self.b.get(self.o..self.o + 8)?.try_into().ok()?);
        self.o += 8;
        Some(v)
    }
    fn p3(&mut self) -> Option<[f64; 3]> {
        Some([self.rd()?, self.rd()?, self.rd()?])
    }
    fn pts(&mut self) -> Option<Vec<[f64; 3]>> {
        let n = self.rl()? as usize;
        if n > self.b.len() / 24 {
            return None;
        }
        (0..n).map(|_| self.p3()).collect()
    }
    /// PS: 8-bit string, zero terminated, padded to 4 bytes.
    fn ps(&mut self) -> Option<String> {
        let rest = self.b.get(self.o..)?;
        let n = rest.iter().position(|&c| c == 0)?;
        let s = rest[..n].iter().map(|&c| c as char).collect();
        self.o += (n + 1).div_ceil(4) * 4;
        Some(s)
    }
    /// PUS: UTF-16LE string, terminated by a zero unit, padded to 4 bytes.
    fn pus(&mut self) -> Option<String> {
        let rest = self.b.get(self.o..)?;
        let units: Vec<u16> = rest.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).take_while(|&u| u != 0).collect();
        self.o += ((units.len() + 1) * 2).div_ceil(4) * 4;
        Some(String::from_utf16_lossy(&units))
    }
}

/// 4x4 row-major model transform, applied to (x, y, z, 1).
type M4 = [f64; 16];

fn xf(m: &M4, p: [f64; 3]) -> P2 {
    [m[0] * p[0] + m[1] * p[1] + m[2] * p[2] + m[3], m[4] * p[0] + m[5] * p[1] + m[6] * p[2] + m[7]]
}

fn mul(a: &M4, b: &M4) -> M4 {
    let mut r = [0.0; 16];
    for i in 0..4 {
        for j in 0..4 {
            r[i * 4 + j] = (0..4).map(|k| a[i * 4 + k] * b[k * 4 + j]).sum();
        }
    }
    r
}

const ID: M4 = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];

/// Color from SUBENT_TRUECOLOR: the high byte is the method (0xC2 RGB, 0xC3 index, 0xC1 BYBLOCK).
fn true_color(v: u32) -> Color {
    match v >> 24 {
        0xC2 => Color::Rgb(v & 0x00FF_FFFF),
        0xC3 => Color::Index(v as u8),
        0xC1 => Color::ByBlock,
        _ => Color::ByLayer,
    }
}

/// Text primitive: position, direction, height, width factor, oblique (types 10, 11, 36, 38).
fn text(pos: [f64; 3], dir: [f64; 3], m: &M4, value: String, height: f64, width_factor: f64, oblique: f64) -> Prim {
    let p = xf(m, pos);
    let d = xf(m, [pos[0] + dir[0], pos[1] + dir[1], pos[2] + dir[2]]);
    let (dx, dy) = (d[0] - p[0], d[1] - p[1]);
    // The model transform scales the height too.
    let scale = (m[0] * m[0] + m[4] * m[4]).sqrt();
    Prim::Text(Text {
        insert: p,
        align: p,
        height: height * scale,
        rotation: dy.atan2(dx),
        width_factor: if width_factor > 0.0 { width_factor } else { 1.0 },
        oblique,
        halign: 0,
        valign: 0,
        value,
        extrusion: [0.0, 0.0, 1.0],
        style: 0,
    })
}

/// Parse proxy graphics. Unknown or damaged chunks are skipped; a damaged header gives no primitives.
pub fn parse(data: &[u8]) -> Vec<ProxyPrim> {
    let mut out = Vec::new();
    let mut r = Rd { b: data, o: 0 };
    let (Some(_total), Some(count)) = (r.rl(), r.rl()) else { return out };
    let mut color = None;
    let mut fill = false;
    let mut stack: Vec<M4> = vec![ID];
    for _ in 0..count {
        let start = r.o;
        let (Some(size), Some(typ)) = (r.rl(), r.rl()) else { break };
        let size = size as usize;
        if size < 8 || start + size > data.len() {
            break;
        }
        let mut c = Rd { b: &data[..start + size], o: start + 8 };
        let m = *stack.last().unwrap_or(&ID);
        let prim = chunk(typ, &mut c, &m, &mut color, &mut fill, &mut stack);
        if let Some(p) = prim {
            out.push((color, p));
        }
        r.o = start + size;
    }
    out
}

fn chunk(typ: u32, c: &mut Rd, m: &M4, color: &mut Option<Color>, fill: &mut bool, stack: &mut Vec<M4>) -> Option<Prim> {
    match typ {
        // CIRCLE: center, radius, normal
        2 => {
            let (center, radius) = (c.p3()?, c.rd()?);
            let ctr = xf(m, center);
            let r = (xf(m, [center[0] + radius, center[1], center[2]])[0] - ctr[0]).hypot(xf(m, [center[0] + radius, center[1], center[2]])[1] - ctr[1]);
            Some(Prim::Arc { center: ctr, radius: r, start: 0.0, end: std::f64::consts::TAU })
        }
        // CIRCULARARC: center, radius, normal, start vector, sweep
        4 => {
            let (center, radius, normal, sv, sweep) = (c.p3()?, c.rd()?, c.p3()?, c.p3()?, c.rd()?);
            let ctr = xf(m, center);
            let s = xf(m, [center[0] + sv[0], center[1] + sv[1], center[2] + sv[2]]);
            let a0 = (s[1] - ctr[1]).atan2(s[0] - ctr[0]);
            let r = radius * (m[0] * m[0] + m[4] * m[4]).sqrt();
            // A negative normal turns the sweep clockwise in 2D.
            let (start, end) = if normal[2] < 0.0 { (a0 - sweep, a0) } else { (a0, a0 + sweep) };
            Some(Prim::Arc { center: ctr, radius: r, start, end })
        }
        // POLYLINE, POLYGON, polyline with normal
        6 | 32 => Some(Prim::Polyline { pts: c.pts()?.into_iter().map(|p| xf(m, p)).collect(), closed: false }),
        7 => {
            let pts: Vec<P2> = c.pts()?.into_iter().map(|p| xf(m, p)).collect();
            Some(if *fill { Prim::Fill(pts) } else { Prim::Polyline { pts, closed: true } })
        }
        // TEXT (8-bit) and Unicode TEXT: point, normal, direction, height, width, oblique, string
        10 | 36 => {
            let (pos, _n, dir, h, w, o) = (c.p3()?, c.p3()?, c.p3()?, c.rd()?, c.rd()?, c.rd()?);
            let s = if typ == 10 { c.ps()? } else { c.pus()? };
            Some(text(pos, dir, m, s, h, w, o))
        }
        // TEXT2 (8-bit) and Unicode TEXT2: point, normal, direction, string, length, raw, height, width, oblique
        11 | 38 => {
            let (pos, _n, dir) = (c.p3()?, c.p3()?, c.p3()?);
            let s = if typ == 11 { c.ps()? } else { c.pus()? };
            c.rl()?;
            c.rl()?;
            let (h, w, o) = (c.rd()?, c.rd()?, c.rd()?);
            Some(text(pos, dir, m, s, h, w, o))
        }
        14 => {
            let v = c.rl()?;
            *color = Some(match v {
                0 => Color::ByBlock,
                1..=255 => Color::Index(v as u8),
                _ => Color::ByLayer,
            });
            None
        }
        22 => {
            *color = Some(true_color(c.rl()?));
            None
        }
        20 => {
            *fill = c.rl()? != 0;
            None
        }
        29 | 30 => {
            let mut t = [0.0; 16];
            for v in t.iter_mut() {
                *v = c.rd()?;
            }
            stack.push(mul(m, &t));
            None
        }
        31 => {
            if stack.len() > 1 {
                stack.pop();
            }
            None
        }
        _ => None,
    }
}

/// Chunk types and their counts, for diagnostics.
pub fn type_counts(data: &[u8], counts: &mut std::collections::BTreeMap<u32, usize>) {
    let mut r = Rd { b: data, o: 0 };
    let (Some(_), Some(count)) = (r.rl(), r.rl()) else { return };
    for _ in 0..count {
        let start = r.o;
        let (Some(size), Some(typ)) = (r.rl(), r.rl()) else { return };
        if size < 8 || start + size as usize > data.len() {
            return;
        }
        *counts.entry(typ).or_default() += 1;
        r.o = start + size as usize;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk_bytes(typ: u32, body: &[u8]) -> Vec<u8> {
        let mut v = ((body.len() + 8) as u32).to_le_bytes().to_vec();
        v.extend(typ.to_le_bytes());
        v.extend(body);
        v
    }

    #[test]
    fn polyline_color_and_unicode_text() {
        let mut body = Vec::new();
        body.extend(2u32.to_le_bytes());
        for p in [[0.0, 0.0, 0.0], [10.0, 5.0, 0.0]] {
            for v in p {
                body.extend(f64::to_le_bytes(v));
            }
        }
        let mut t = Vec::new();
        for v in [1.0, 2.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 2.5, 1.0, 0.0] {
            t.extend(f64::to_le_bytes(v));
        }
        t.extend([b'H', 0, b'i', 0, 0, 0, 0, 0]); // "Hi", terminator, padding to 4
        let mut d = Vec::new();
        d.extend(chunk_bytes(22, &0xC200_FF00u32.to_le_bytes()));
        d.extend(chunk_bytes(6, &body));
        d.extend(chunk_bytes(36, &t));
        let mut all = ((d.len() + 8) as u32).to_le_bytes().to_vec();
        all.extend(3u32.to_le_bytes());
        all.extend(d);
        let p = parse(&all);
        assert_eq!(p.len(), 2);
        assert!(matches!(p[0], (Some(Color::Rgb(0x00FF00)), Prim::Polyline { .. })));
        match &p[1].1 {
            Prim::Text(t) => assert_eq!((t.value.as_str(), t.insert, t.height), ("Hi", [1.0, 2.0], 2.5)),
            other => panic!("{other:?}"),
        }
    }
}
