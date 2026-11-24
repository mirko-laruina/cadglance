//! Classes of one sample file per version. Samples live outside the repo, in
//! `../samples/acadsharp` (or `CADGLANCE_SAMPLES`); the test is skipped when the folder is missing.

use std::path::PathBuf;

use cadglance::classes::parse_classes;
use cadglance::container::Container;

#[test]
fn classes_of_every_version() {
    let dir = std::env::var_os("CADGLANCE_SAMPLES")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../samples/acadsharp"));
    if !dir.is_dir() {
        eprintln!("skipped: no samples in {}", dir.display());
        return;
    }
    for magic in ["AC1015", "AC1018", "AC1021", "AC1024", "AC1027", "AC1032"] {
        let path = dir.join(format!("sample_{magic}.dwg"));
        let raw = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let c = Container::parse(&raw).unwrap_or_else(|e| panic!("{magic}: {e:?}"));
        let data = c.section("AcDb:Classes", 1).unwrap_or_else(|e| panic!("{magic}: {e:?}"));
        let classes = parse_classes(&data, c.ver).unwrap_or_else(|e| panic!("{magic}: {e:?}"));
        assert!(!classes.is_empty(), "{magic}: no classes");
        assert!(classes.iter().all(|k| k.number >= 500 && !k.dxf_name.is_empty()), "{magic}: bad class");
    }
}
