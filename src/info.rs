//! Entity properties for the viewer's info panel, as JSON.
//! Values are in block coordinates; each carries a kind so the viewer can map it to world
//! coordinates: "pt" (point), "len" (length), "area", "ang" (degrees), "num", "text".

use crate::curves::{bulged, interpolate_fit, nurbs_points};
use crate::drawing::Drawing;
use crate::entities::{Kind, P2};
use crate::hatch;
use crate::objects::Color;
use crate::text::{clean_mtext, clean_text, json_str};

use InfoValue as V;

/// A property value and how to map it to world coordinates.
#[derive(Debug, Clone)]
pub enum InfoValue {
    Pt(P2),
    Len(f64),
    Area(f64),
    Ang(f64),
    Num(f64),
    Text(String),
}

fn mirror(extrusion: &[f64; 3], p: P2) -> P2 {
    if extrusion[2] < 0.0 { [-p[0], p[1]] } else { p }
}

fn length(pts: &[P2]) -> f64 {
    pts.windows(2).map(|w| (w[1][0] - w[0][0]).hypot(w[1][1] - w[0][1])).sum()
}

/// Shoelace area of a closed point loop (absolute value).
fn area(pts: &[P2]) -> f64 {
    let n = pts.len();
    (0..n).map(|i| pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1]).sum::<f64>().abs() / 2.0
}

/// Properties of one entity: values are in block coordinates (see `InfoValue`).
pub struct EntityInfo {
    pub handle: u64,
    pub type_name: String,
    pub layer: String,
    pub color: String,
    pub values: Vec<(String, InfoValue)>,
}

pub fn entity_details(d: &Drawing, h: u64) -> Option<EntityInfo> {
    let e = d.entities.get(&h)?;
    let tol = 1e-3;
    let deg = |r: f64| r.to_degrees();
    let mut v: Vec<(&str, V)> = Vec::new();
    match &e.kind {
        Kind::Line { a, b } => {
            v.push(("Start", V::Pt(*a)));
            v.push(("End", V::Pt(*b)));
            v.push(("Length", V::Len((b[0] - a[0]).hypot(b[1] - a[1]))));
            v.push(("Angle", V::Ang(deg((b[1] - a[1]).atan2(b[0] - a[0])))));
        }
        Kind::Circle { center, radius, extrusion } => {
            v.push(("Center", V::Pt(mirror(extrusion, *center))));
            v.push(("Radius", V::Len(*radius)));
            v.push(("Diameter", V::Len(2.0 * radius)));
            v.push(("Circumference", V::Len(std::f64::consts::TAU * radius)));
            v.push(("Area", V::Area(std::f64::consts::PI * radius * radius)));
        }
        Kind::Arc { center, radius, start, end, extrusion } => {
            let mut sweep = end - start;
            while sweep <= 0.0 {
                sweep += std::f64::consts::TAU;
            }
            v.push(("Center", V::Pt(mirror(extrusion, *center))));
            v.push(("Radius", V::Len(*radius)));
            v.push(("Start angle", V::Ang(deg(*start))));
            v.push(("End angle", V::Ang(deg(*end))));
            v.push(("Sweep", V::Num(deg(sweep))));
            v.push(("Arc length", V::Len(radius * sweep)));
        }
        Kind::Ellipse { center, major, ratio, .. } => {
            let a = major[0].hypot(major[1]);
            v.push(("Center", V::Pt(*center)));
            v.push(("Semi-major axis", V::Len(a)));
            v.push(("Semi-minor axis", V::Len(a * ratio)));
            v.push(("Axis ratio", V::Num(*ratio)));
        }
        Kind::LwPolyline { closed, pts, bulges, extrusion, .. } => {
            let p: Vec<P2> = bulged(pts, bulges, *closed, tol).into_iter().map(|q| mirror(extrusion, q)).collect();
            v.push(("Vertices", V::Num(pts.len() as f64)));
            v.push(("Closed", V::Text(if *closed { "yes" } else { "no" }.into())));
            v.push(("Length", V::Len(length(&p))));
            if *closed {
                v.push(("Area", V::Area(area(&p))));
            }
            if let Some(first) = p.first() {
                v.push(("First vertex", V::Pt(*first)));
            }
        }
        Kind::Polyline2d { closed, vertices, extrusion } => {
            let (mut pts, mut bul) = (Vec::new(), Vec::new());
            for vh in vertices {
                if let Some(Kind::Vertex2d { p, bulge }) = d.entities.get(vh).map(|e| &e.kind) {
                    pts.push(*p);
                    bul.push(*bulge);
                }
            }
            let p: Vec<P2> = bulged(&pts, &bul, *closed, tol).into_iter().map(|q| mirror(extrusion, q)).collect();
            v.push(("Vertices", V::Num(pts.len() as f64)));
            v.push(("Length", V::Len(length(&p))));
            if *closed {
                v.push(("Area", V::Area(area(&p))));
            }
        }
        Kind::Spline { degree, closed, knots, ctrl, weights, fit, tangents, knot_param } => {
            let p = if ctrl.is_empty() {
                interpolate_fit(fit, *tangents, *knot_param, *closed)
                    .and_then(|(k, c)| nurbs_points(3, &k, &c, &[]))
                    .unwrap_or_else(|| fit.clone())
            } else {
                nurbs_points(*degree as usize, knots, ctrl, weights).unwrap_or_else(|| ctrl.clone())
            };
            v.push(("Degree", V::Num(if ctrl.is_empty() { 3.0 } else { *degree as f64 })));
            if ctrl.is_empty() {
                v.push(("Fit points", V::Num(fit.len() as f64)));
            } else {
                v.push(("Control points", V::Num(ctrl.len() as f64)));
            }
            v.push(("Length (approx.)", V::Len(length(&p))));
            if let (Some(a), Some(b)) = (p.first(), p.last()) {
                v.push(("Start", V::Pt(*a)));
                v.push(("End", V::Pt(*b)));
            }
        }
        Kind::Hatch { solid, paths, pattern, extrusion } => {
            let loops = hatch::loops(paths, tol);
            v.push(("Type", V::Text(if *solid { "solid fill".into() } else { "pattern".into() })));
            v.push(("Boundaries", V::Num(paths.len() as f64)));
            if !*solid {
                v.push(("Pattern lines", V::Num(pattern.len() as f64)));
            }
            if let Some(outer) = loops.iter().max_by(|a, b| area(a).total_cmp(&area(b))) {
                let holes: f64 = loops.iter().map(|l| area(l)).sum::<f64>() - area(outer);
                v.push(("Outer boundary area", V::Area(area(outer))));
                v.push(("Area (outer − inner)", V::Area(area(outer) - holes)));
                if let Some(p) = outer.first() {
                    v.push(("First vertex", V::Pt(mirror(extrusion, *p))));
                }
            }
        }
        Kind::Solid { corners, extrusion } => {
            let o = [corners[0], corners[1], corners[3], corners[2]].map(|p| mirror(extrusion, p));
            v.push(("Area", V::Area(area(&o))));
            v.push(("First vertex", V::Pt(o[0])));
        }
        Kind::Text(t) | Kind::Attrib { text: t, .. } => {
            if let Kind::Attrib { tag, .. } = &e.kind {
                v.push(("Tag", V::Text(tag.clone())));
            }
            v.push(("Text", V::Text(clean_text(&t.value))));
            v.push(("Position", V::Pt(t.insert)));
            v.push(("Height", V::Len(t.height)));
            v.push(("Rotation", V::Ang(deg(t.rotation))));
        }
        Kind::MText { insert, x_dir, rect_width, height, value, .. } => {
            v.push(("Text", V::Text(clean_mtext(value))));
            v.push(("Position", V::Pt(*insert)));
            v.push(("Height", V::Len(*height)));
            v.push(("Box width", V::Len(*rect_width)));
            v.push(("Rotation", V::Ang(deg(x_dir[1].atan2(x_dir[0])))));
        }
        _ => {}
    }

    let layer = d.layers.get(&e.layer).map_or_else(|| format!("#{:X}", e.layer), |l| l.name.clone());
    let color = match e.color {
        Color::ByLayer => "ByLayer".to_string(),
        Color::ByBlock => "ByBlock".to_string(),
        Color::Index(i) => format!("ACI {i}"),
        Color::Rgb(c) => format!("RGB {},{},{}", c >> 16 & 0xFF, c >> 8 & 0xFF, c & 0xFF),
    };
    let type_name = d.types.get(&h).map_or_else(|| "?".to_string(), |&t| d.type_name(t));
    Some(EntityInfo { handle: h, type_name, layer, color, values: v.into_iter().map(|(n, x)| (n.to_string(), x)).collect() })
}

/// JSON for the web viewer.
pub fn entity_info(d: &Drawing, h: u64) -> Option<String> {
    let i = entity_details(d, h)?;
    let values: Vec<String> = i
        .values
        .iter()
        .map(|(name, val)| {
            let (kind, value) = match val {
                V::Pt(p) => ("pt", format!("[{},{}]", p[0], p[1])),
                V::Len(x) => ("len", x.to_string()),
                V::Area(x) => ("area", x.to_string()),
                V::Ang(x) => ("ang", x.to_string()),
                V::Num(x) => ("num", x.to_string()),
                V::Text(s) => ("text", json_str(s)),
            };
            format!("[{},\"{kind}\",{value}]", json_str(name))
        })
        .collect();
    Some(format!(
        "{{\"handle\":\"{:X}\",\"type\":{},\"layer\":{},\"color\":{},\"values\":[{}]}}",
        i.handle,
        json_str(&i.type_name),
        json_str(&i.layer),
        json_str(&i.color),
        values.join(",")
    ))
}
