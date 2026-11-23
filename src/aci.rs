//! AutoCAD Color Index (ACI) to RGB, standard 256-color table.

/// RGB of an ACI index. 0 (ByBlock) and 256 (ByLayer) are resolved by the caller; 0 maps to white.
pub fn aci_rgb(i: u8) -> [u8; 3] {
    match i {
        0 | 7 => [255, 255, 255],
        1 => [255, 0, 0],
        2 => [255, 255, 0],
        3 => [0, 255, 0],
        4 => [0, 255, 255],
        5 => [0, 0, 255],
        6 => [255, 0, 255],
        8 => [128, 128, 128],
        9 => [192, 192, 192],
        250..=255 => {
            let g = [51, 80, 105, 130, 190, 255][(i - 250) as usize];
            [g, g, g]
        }
        _ => {
            // 10-249: 24 hues every 15 degrees; within each group of 10, pairs of
            // (full, half saturation) at values 255, 165, 127, 76, 38.
            let k = (i - 10) as u32;
            let hue = (k / 10) as f64 * 15.0;
            let shade = (k % 10) as usize;
            let v = [255.0, 255.0, 165.0, 165.0, 127.0, 127.0, 76.0, 76.0, 38.0, 38.0][shade];
            let s = if shade % 2 == 0 { 1.0 } else { 0.5 };
            hsv(hue, s, v)
        }
    }
}

/// HSV with v in 0..=255; channels truncated like the AutoCAD table.
fn hsv(h: f64, s: f64, v: f64) -> [u8; 3] {
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;
    let (r, g, b) = match (h / 60.0) as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    [(r + m) as u8, (g + m) as u8, (b + m) as u8]
}

#[cfg(test)]
mod tests {
    use super::aci_rgb;

    #[test]
    fn known_entries() {
        assert_eq!(aci_rgb(10), [255, 0, 0]);
        assert_eq!(aci_rgb(11), [255, 127, 127]);
        assert_eq!(aci_rgb(13), [165, 82, 82]);
        assert_eq!(aci_rgb(30), [255, 127, 0]);
        assert_eq!(aci_rgb(40), [255, 191, 0]);
        assert_eq!(aci_rgb(140), [0, 191, 255]);
        assert_eq!(aci_rgb(150), [0, 127, 255]);
        assert_eq!(aci_rgb(250), [51, 51, 51]);
    }
}
