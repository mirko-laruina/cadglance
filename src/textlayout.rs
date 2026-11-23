//! Text layout for viewers (same rules as the web viewer's layoutText): alignment, MTEXT
//! attachment and paragraph alignment, word wrap, and width calibration against the widths
//! AutoCAD stored in the DWG (the viewer font is not the CAD font).

use std::collections::HashMap;

use crate::entities::P2;
use crate::geom::TextItem;

/// Font metrics in font pixels at the atlas size.
pub trait Metrics {
    /// Advance of `ch`, or None if the font has no glyph for it.
    fn advance(&self, ch: char) -> Option<f64>;
    /// Advance used for missing glyphs.
    fn fallback_advance(&self) -> f64;
    /// Cap height (height of "H").
    fn cap(&self) -> f64;
}

/// One glyph in the text frame (u along the baseline, v up), drawing units. `k` = drawing units
/// per font pixel, `wf` = horizontal stretch.
#[derive(Debug, Clone, Copy)]
pub struct Placed {
    pub ch: char,
    pub u: f64,
    pub v: f64,
    pub k: f64,
    pub wf: f64,
}

#[derive(Debug, Clone)]
pub struct Layout {
    /// Text frame origin, block-local (relative to the batch center).
    pub anchor: P2,
    pub rot: f64,
    /// tan(oblique angle): u += v * ob.
    pub ob: f64,
    pub glyphs: Vec<Placed>,
    /// Bounding box in the text frame: u0, v0, u1, v1.
    pub bx: [f64; 4],
}

impl Layout {
    /// Text frame point to block-local coordinates.
    pub fn local(&self, u: f64, v: f64) -> P2 {
        let (s, c) = self.rot.sin_cos();
        let uu = u + v * self.ob;
        [self.anchor[0] + uu * c - v * s, self.anchor[1] + uu * s + v * c]
    }
}

pub fn font_width(s: &str, h: f64, wf: f64, m: &dyn Metrics) -> f64 {
    let w: f64 = s.chars().map(|c| m.advance(c).unwrap_or(m.fallback_advance())).sum();
    w * (h / m.cap()) * wf
}

/// Width AutoCAD computed for a centered/right/middle TEXT: distance along the baseline between
/// the stored alignment point and the derived insertion point.
pub fn cad_width(t: &TextItem) -> Option<f64> {
    if t.kind != 0 || !matches!(t.halign, 1 | 2 | 4) {
        return None;
    }
    let (s, c) = t.rotation.sin_cos();
    let proj = (t.align[0] - t.pos[0]) * c + (t.align[1] - t.pos[1]) * s;
    let w = if t.halign == 2 { proj } else { 2.0 * proj };
    (w > 0.0).then_some(w)
}

/// Per style: median of (AutoCAD width / font width) over the texts where both are known.
pub fn style_ratios(texts: &[TextItem], m: &dyn Metrics) -> HashMap<u64, f64> {
    let mut by: HashMap<u64, Vec<f64>> = HashMap::new();
    for t in texts {
        let fw = font_width(&t.value, t.height, if t.width_factor > 0.0 { t.width_factor } else { 1.0 }, m);
        if let (Some(cad), true) = (cad_width(t), fw > 0.0) {
            by.entry(t.style).or_default().push(cad / fw);
        }
    }
    by.into_iter()
        .map(|(st, mut v)| {
            v.sort_by(f64::total_cmp);
            (st, v[v.len() / 2])
        })
        .collect()
}

/// Scale that stretches the viewer font to the CAD width for this text.
pub fn text_scale(t: &TextItem, ratios: &HashMap<u64, f64>, m: &dyn Metrics) -> f64 {
    let fw = font_width(&t.value, t.height, if t.width_factor > 0.0 { t.width_factor } else { 1.0 }, m);
    match cad_width(t) {
        Some(cad) if fw > 0.0 => cad / fw,
        _ => ratios.get(&t.style).copied().unwrap_or(1.0),
    }
}

pub fn layout(t: &TextItem, styles: &HashMap<u64, (f64, f64)>, m: &dyn Metrics, scale: f64) -> Layout {
    let style = styles.get(&t.style).copied();
    let mut h = t.height;
    let mut rot = t.rotation;
    let base_wf = if t.kind == 1 {
        style.map_or(1.0, |s| if s.0 > 0.0 { s.0 } else { 1.0 })
    } else if t.width_factor > 0.0 {
        t.width_factor
    } else {
        1.0
    };
    let mut wf = base_wf * scale;
    let ob = if t.kind == 1 { style.map_or(0.0, |s| s.1) } else { t.oblique };

    let mut anchor = t.pos;
    let mut placed: Vec<(String, f64, f64)> = Vec::new(); // line, du, dv
    if t.kind == 1 {
        let lw = |s: &str, h: f64, wf: f64| font_width(s, h, wf, m);
        let mut lines: Vec<(String, u8)> = Vec::new();
        for (pi, para) in t.value.split('\n').enumerate() {
            let pa = t.para.get(pi).copied().unwrap_or(0);
            if t.width > 0.0 {
                let mut cur = String::new();
                for word in para.split(' ') {
                    let next = if cur.is_empty() { word.to_string() } else { format!("{cur} {word}") };
                    if !cur.is_empty() && lw(&next, h, wf) > t.width * 1.001 {
                        lines.push((std::mem::take(&mut cur), pa));
                        cur = word.to_string();
                    } else {
                        cur = next;
                    }
                }
                lines.push((cur, pa));
            } else {
                lines.push((para.to_string(), pa));
            }
        }
        // Attachment places the reference box around the insertion point; paragraph alignment
        // places each line inside the box, defaulting to the attachment column.
        let ls = h * 5.0 / 3.0;
        let at = if (1..=9).contains(&t.attachment) { t.attachment } else { 1 };
        let (col, row) = ((at - 1) % 3, (at - 1) / 3);
        let widths: Vec<f64> = lines.iter().map(|(l, _)| lw(l, h, wf)).collect();
        let box_w = if t.width > 0.0 { t.width } else { widths.iter().copied().fold(0.0, f64::max) };
        let box_left = match col {
            0 => 0.0,
            1 => -box_w / 2.0,
            _ => -box_w,
        };
        let total = h + (lines.len().saturating_sub(1)) as f64 * ls;
        let y0 = match row {
            0 => -h,
            1 => -h + total / 2.0,
            _ => -h + total,
        };
        for (i, ((ln, pa), w)) in lines.into_iter().zip(widths).enumerate() {
            let a = if pa == 0 { col as u8 + 1 } else { pa };
            let du = box_left
                + match a {
                    1 => 0.0,
                    2 => (box_w - w) / 2.0,
                    _ => box_w - w,
                };
            placed.push((ln, du, y0 - i as f64 * ls));
        }
    } else {
        let ln = t.value.replace('\n', " ");
        let w = font_width(&ln, h, wf, m);
        let (mut du, mut dv) = (0.0, 0.0);
        let (dx, dy) = (t.align[0] - t.pos[0], t.align[1] - t.pos[1]);
        let l = dx.hypot(dy);
        if matches!(t.halign, 3 | 5) && l > 0.0 && w > 0.0 {
            rot = dy.atan2(dx);
            if t.halign == 5 {
                wf *= l / w;
            } else {
                h *= l / w;
            }
        } else if t.halign != 0 || t.valign != 0 {
            anchor = t.align;
            du = match t.halign {
                1 | 4 => -w / 2.0,
                2 => -w,
                _ => 0.0,
            };
            dv = if t.halign == 4 {
                -h / 2.0
            } else {
                match t.valign {
                    1 => h * 0.25,
                    2 => -h / 2.0,
                    3 => -h,
                    _ => 0.0,
                }
            };
        }
        placed.push((ln, du, dv));
    }

    let k = h / m.cap();
    let mut glyphs = Vec::new();
    let mut bx = [f64::MAX, f64::MAX, f64::MIN, f64::MIN];
    for (ln, du, dv) in placed {
        let mut u = du;
        for ch in ln.chars() {
            let adv = m.advance(ch);
            if adv.is_some() && !ch.is_whitespace() {
                glyphs.push(Placed { ch, u, v: dv, k, wf });
            }
            u += adv.unwrap_or(m.fallback_advance()) * k * wf;
        }
        bx = [bx[0].min(du), bx[1].min(dv - h * 0.25), bx[2].max(u), bx[3].max(dv + h)];
    }
    Layout { anchor, rot, ob: ob.tan(), glyphs, bx }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Mono;
    impl Metrics for Mono {
        fn advance(&self, _: char) -> Option<f64> {
            Some(53.4)
        }
        fn fallback_advance(&self) -> f64 {
            48.0
        }
        fn cap(&self) -> f64 {
            68.7
        }
    }

    fn text(kind: u8, value: &str) -> TextItem {
        TextItem {
            batch: 0,
            kind,
            pos: [0.0, 0.0],
            align: [0.0, 0.0],
            height: 4.0,
            rotation: 0.0,
            width_factor: 1.0,
            oblique: 0.0,
            halign: 0,
            valign: 0,
            attachment: 1,
            width: 0.0,
            color: [255; 4],
            id: 0,
            layer: 0,
            style: 5,
            para: vec![],
            value: value.into(),
        }
    }

    #[test]
    fn mtext_paragraph_centered_in_box() {
        // "\pxqc;7": centered paragraph in a 100 wide box, top-left attachment
        let mut t = text(1, "7");
        t.height = 5.0;
        t.width = 100.0;
        t.para = vec![2];
        let l = layout(&t, &HashMap::new(), &Mono, 1.0);
        let g = l.glyphs[0];
        let center = g.u + 53.4 * g.k * g.wf / 2.0;
        assert!((center - 50.0).abs() < 1e-9, "{center}");
    }

    #[test]
    fn centered_text_matches_cad_width() {
        let mut t = text(0, "7");
        t.halign = 4;
        t.align = [2.5, 2.0];
        let s = text_scale(&t, &HashMap::new(), &Mono);
        let l = layout(&t, &HashMap::new(), &Mono, s);
        assert!(((l.bx[2] - l.bx[0]) - cad_width(&t).unwrap()).abs() < 1e-9);
        assert!((l.local(l.bx[0], 0.0)[0]).abs() < 1e-9, "starts at the insertion point");
    }
}
