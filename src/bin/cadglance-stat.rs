//! cadglance-stat: timings, entity counts per layout, viewports, decode self-checks.
//! Prints counts and names of structural objects only (layouts, types), never drawing content.

use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use cadglance::drawing::Drawing;
use cadglance::entities::Kind;

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: cadglance-stat <file.dwg>");
        std::process::exit(2);
    });
    let t0 = Instant::now();
    let raw = std::fs::read(&path).unwrap_or_else(|e| {
        eprintln!("{path}: {e}");
        std::process::exit(1);
    });
    let t_read = t0.elapsed();
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let d = Drawing::load(&raw, threads).unwrap_or_else(|e| {
        eprintln!("{path}: {e}");
        std::process::exit(1);
    });
    let t = &d.timings;
    println!("version {}  objects {}  threads {threads}", d.version, d.types.len());
    println!(
        "time: read file {:?}, container {:?}, decompress {:?}, object map {:?}, parse {:?}, merge {:?}, total {:?}",
        t_read,
        t.read_container,
        t.decompress,
        t.index,
        t.parse,
        t.merge,
        t0.elapsed()
    );

    // Self-checks
    let mut inexact: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for e in d.entities.values() {
        if matches!(e.kind, Kind::Other) {
            continue;
        }
        let k = inexact.entry(d.type_name(d.types[&e.handle])).or_default();
        k.0 += 1;
        if !e.exact {
            k.1 += 1;
        }
    }
    println!("\ndecode check (entities whose data stream was consumed exactly):");
    for (name, (n, bad)) in &inexact {
        println!("  {name:14} {n:>9}  inexact {bad}");
    }
    let mut errs: BTreeMap<String, (usize, String)> = BTreeMap::new();
    for (h, ty, e) in &d.errors {
        let k = errs.entry(d.type_name(*ty)).or_insert((0, format!("{h:#x} {e}")));
        k.0 += 1;
    }
    println!("parse errors: {}", d.errors.len());
    for (name, (n, first)) in &errs {
        println!("  {name:14} {n:>9}  first: {first}");
    }
    let bad_blocks = d.blocks.values().filter(|b| !b.exact).count();
    let bad_layouts = d.layouts.iter().filter(|l| !l.exact).count();
    println!("block records {} (inexact {bad_blocks}), layouts {} (inexact {bad_layouts})", d.blocks.len(), d.layouts.len());

    // Expanded primitive count per block (what a viewer must draw), memoized.
    let mut memo: HashMap<u64, u64> = HashMap::new();
    let by_name: HashMap<&str, u64> = d.blocks.values().map(|b| (b.name.as_str(), b.handle)).collect();

    let report = |title: &str, block: u64, memo: &mut HashMap<u64, u64>| {
        let Some(b) = d.blocks.get(&block) else {
            println!("\n{title}: block record {block:#x} not found");
            return;
        };
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        let mut viewports = 0usize;
        for h in &b.entities {
            let name = d.types.get(h).map_or_else(|| "MISSING".into(), |&t| d.type_name(t));
            if name == "VIEWPORT" {
                viewports += 1;
            }
            *counts.entry(name).or_default() += 1;
        }
        let expanded = expand(&d, block, memo, 0);
        println!("\n{title}  [{} #{:#x}]", b.name, b.handle);
        println!("  direct entities {}, expanded primitives {expanded}", b.entities.len());
        if viewports > 0 {
            // The first VIEWPORT of a paper space layout is the sheet itself.
            println!("  VIEWPORT entities {viewports} (model space views: {})", viewports.saturating_sub(1));
        }
        let mut v: Vec<_> = counts.into_iter().collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        for (name, n) in v {
            println!("    {name:14} {n:>9}");
        }
    };

    if let Some((_, &ms)) = by_name.iter().find(|(n, _)| n.eq_ignore_ascii_case("*Model_Space")) {
        report("MODEL SPACE", ms, &mut memo);
    }
    for l in &d.layouts {
        if d.blocks.get(&l.block_record).is_some_and(|b| b.name.eq_ignore_ascii_case("*Model_Space")) {
            continue;
        }
        report(&format!("LAYOUT \"{}\" (tab {})", l.name, l.tab_order), l.block_record, &mut memo);
    }
}

/// Number of drawable primitives in a block once INSERTs/DIMENSIONs are expanded.
fn expand(d: &Drawing, block: u64, memo: &mut HashMap<u64, u64>, depth: u32) -> u64 {
    if let Some(&n) = memo.get(&block) {
        return n;
    }
    if depth > 32 {
        return 0; // recursive block reference
    }
    memo.insert(block, 0);
    let mut n = 0u64;
    if let Some(b) = d.blocks.get(&block) {
        for h in &b.entities {
            n += match d.entities.get(h).map(|e| &e.kind) {
                Some(Kind::Insert { block, cols, rows, attribs, .. }) => {
                    expand(d, *block, memo, depth + 1) * (*cols).max(1) as u64 * (*rows).max(1) as u64
                        + attribs.len() as u64
                }
                Some(Kind::Dimension { block }) => expand(d, *block, memo, depth + 1),
                Some(Kind::Polyline2d { vertices, .. }) => vertices.len() as u64,
                Some(Kind::Viewport { .. }) => 0,
                _ => 1,
            };
        }
    }
    memo.insert(block, n);
    n
}
