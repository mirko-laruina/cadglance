# CadGlance

CadGlance opens `.dwg` files and shows their layouts. You can pan, zoom, show and hide layers, measure distances and areas, search text and export the view to PNG.

It has two viewers with the same features: a desktop app and a web page served on your computer.

## Build

Requires [Rust](https://rustup.rs) 1.88 or later.

```sh
cargo build --release                       # web viewer and cadglance-stat
cargo build --release --features desktop    # desktop viewer
```

## Usage

```sh
# Desktop viewer
./target/release/cadglance drawing.dwg
./target/release/cadglance drawing.dwg --software    # without a graphics card

# Web viewer, then open http://127.0.0.1:8080/
./target/release/cadglance-web drawing.dwg 8080

# File statistics and decoder checks
./target/release/cadglance-stat drawing.dwg
```
