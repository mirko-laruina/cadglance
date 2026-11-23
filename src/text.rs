//! Text cleanup for display: TEXT %% codes and MTEXT inline formatting codes.

/// Replace TEXT control codes (%%c, %%d, %%p, %%nnn; %%u/%%o toggles are dropped).
pub fn clean_text(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let b: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < b.len() {
        if b[i] == '%' && b.get(i + 1) == Some(&'%') {
            match b.get(i + 2).map(|c| c.to_ascii_lowercase()) {
                Some('c') => out.push('Ø'),
                Some('d') => out.push('°'),
                Some('p') => out.push('±'),
                Some('%') => out.push('%'),
                Some('u') | Some('o') | Some('k') => {}
                Some(c) if c.is_ascii_digit() => {
                    let digits: String = b[i + 2..].iter().take(3).take_while(|c| c.is_ascii_digit()).collect();
                    if let Some(ch) = digits.parse::<u32>().ok().and_then(char::from_u32) {
                        out.push(ch);
                    }
                    i += 2 + digits.len();
                    continue;
                }
                _ => {
                    out.push('%');
                    i += 1;
                    continue;
                }
            }
            i += 3;
            continue;
        }
        out.push(b[i]);
        i += 1;
    }
    out
}

/// Strip MTEXT formatting codes; paragraphs (\P) become '\n'. Stacked fractions become "a/b".
pub fn clean_mtext(s: &str) -> String {
    mtext_paragraphs(s).0
}

/// Paragraph alignment from `\pxq?;`: 0 = attachment default, 1 left, 2 center, 3 right.
fn para_align(arg: &[char]) -> Option<u8> {
    let q = arg.iter().position(|&c| c == 'q')?;
    match arg.get(q + 1) {
        Some('l') | Some('j') | Some('d') => Some(1),
        Some('c') => Some(2),
        Some('r') => Some(3),
        _ => None,
    }
}

/// Like `clean_mtext`, plus the alignment of each paragraph (one entry per '\n'-separated line).
pub fn mtext_paragraphs(s: &str) -> (String, Vec<u8>) {
    let b: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut aligns = vec![0u8];
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            '{' | '}' => i += 1,
            '\\' if i + 1 < b.len() => {
                let c = b[i + 1];
                i += 2;
                match c {
                    'P' | 'X' => {
                        out.push('\n');
                        let cur = *aligns.last().unwrap_or(&0);
                        aligns.push(cur);
                    }
                    'p' => {
                        let start = i;
                        while i < b.len() && b[i] != ';' {
                            i += 1;
                        }
                        if let (Some(a), Some(last)) = (para_align(&b[start..i]), aligns.last_mut()) {
                            *last = a;
                        }
                        i += 1;
                    }
                    '~' => out.push('\u{a0}'),
                    '\\' | '{' | '}' => out.push(c),
                    'U' if b.get(i) == Some(&'+') => {
                        let hex: String = b[i + 1..].iter().take(4).collect();
                        if let Some(ch) = u32::from_str_radix(&hex, 16).ok().and_then(char::from_u32) {
                            out.push(ch);
                            i += 5;
                        }
                    }
                    'S' => {
                        // \Snum^den; or num/den; or num#den;
                        while i < b.len() && b[i] != ';' {
                            out.push(if matches!(b[i], '^' | '#') { '/' } else { b[i] });
                            i += 1;
                        }
                        i += 1;
                    }
                    // codes with an argument up to ';'
                    'f' | 'F' | 'H' | 'W' | 'Q' | 'T' | 'A' | 'C' | 'c' => {
                        while i < b.len() && b[i] != ';' {
                            i += 1;
                        }
                        i += 1;
                    }
                    // toggles without argument: underline, overline, strike, ...
                    _ => {}
                }
            }
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    (clean_text(&out), aligns)
}

/// JSON string literal.
pub fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_codes() {
        assert_eq!(clean_text("%%c10 %%d %%p0.1 %%uX%%u 100%%%"), "Ø10 ° ±0.1 X 100%");
    }

    #[test]
    fn mtext_codes() {
        assert_eq!(clean_mtext("{\\fArial|b1;Title}\\PLine \\H2.5x;two"), "Title\nLine two");
        assert_eq!(clean_mtext("\\S1^2; in\\~x \\\\ \\U+00B0"), "1/2 in\u{a0}x \\ °");
    }

    #[test]
    fn json_escaping() {
        assert_eq!(json_str("a\"b\\c\nd\u{1}"), "\"a\\\"b\\\\c\\nd\\u0001\"");
    }
}

#[cfg(test)]
mod paragraph_tests {
    use super::mtext_paragraphs;

    #[test]
    fn paragraph_alignment() {
        assert_eq!(mtext_paragraphs("\\pxqc;7"), ("7".to_string(), vec![2]));
        assert_eq!(mtext_paragraphs("a\\P\\pxqr;b\\Pc"), ("a\nb\nc".to_string(), vec![0, 3, 3]));
    }
}
