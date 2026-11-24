//! CadGlance web: load a .dwg file, build the scene in memory and serve a WebGL2 viewer on 127.0.0.1.
//! Nothing derived from the drawing is written to disk.

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::time::Instant;

use cadglance::drawing::Drawing;
use cadglance::geom::Scene;
use cadglance::info::entity_info;

const INDEX: &str = include_str!("../../web/index.html");

struct Payload {
    drawing: Drawing,
    meta: Vec<u8>,
    positions: Vec<u8>,
    colors: Vec<u8>,
    ids: Vec<u8>,
    fill_positions: Vec<u8>,
    fill_colors: Vec<u8>,
    instances: Vec<u8>,
    inst_colors: Vec<u8>,
    layers: Vec<u8>,
    fill_layers: Vec<u8>,
    inst_layers: Vec<u8>,
}

fn u16_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().unwrap_or_else(|| {
        eprintln!("usage: cadglance-web <file.dwg> [port]");
        std::process::exit(2);
    });
    let port: u16 = args.next().map_or(8080, |p| p.parse().expect("port must be a number"));

    let t0 = Instant::now();
    let raw = std::fs::read(&path).unwrap_or_else(|e| {
        eprintln!("{path}: {e}");
        std::process::exit(1);
    });
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let d = Drawing::load(&raw, threads).unwrap_or_else(|e| {
        eprintln!("{path}: {e}");
        std::process::exit(1);
    });
    drop(raw);
    let t_load = t0.elapsed();
    let Some(ms) = d.blocks.values().find(|b| b.name.eq_ignore_ascii_case("*Model_Space")).map(|b| b.handle) else {
        eprintln!("{path}: no *Model_Space block");
        std::process::exit(1);
    };
    let t1 = Instant::now();
    let scene = Scene::build(&d, ms);
    let t_scene = t1.elapsed();
    println!(
        "loaded in {t_load:?}, scene in {t_scene:?}: {} batches, {} line vertices, {} fill vertices, {} instances, {} texts, {} segments and {} triangles drawn",
        scene.blocks.len(),
        scene.positions.len() / 2,
        scene.fill_positions.len() / 2,
        scene.instances.len() / 6,
        scene.texts.len(),
        scene.segments_drawn,
        scene.triangles_drawn
    );
    let payload = Arc::new(Payload {
        meta: scene.meta_json().into_bytes(),
        positions: f32_bytes(&scene.positions),
        colors: scene.colors.clone(),
        ids: scene.ids.iter().flat_map(|x| x.to_le_bytes()).collect(),
        fill_positions: f32_bytes(&scene.fill_positions),
        fill_colors: scene.fill_colors.clone(),
        instances: f32_bytes(&scene.instances),
        inst_colors: scene.inst_colors.clone(),
        layers: u16_bytes(&scene.layers),
        fill_layers: u16_bytes(&scene.fill_layers),
        inst_layers: u16_bytes(&scene.inst_layers),
        drawing: d,
    });
    drop(scene);
    let total: usize = [
        &payload.meta,
        &payload.positions,
        &payload.colors,
        &payload.ids,
        &payload.fill_positions,
        &payload.fill_colors,
        &payload.instances,
        &payload.inst_colors,
        &payload.layers,
        &payload.fill_layers,
        &payload.inst_layers,
    ]
    .iter()
    .map(|b| b.len())
    .sum();
    println!("payload {:.1} MB (meta {:.1} MB)", total as f64 / 1e6, payload.meta.len() as f64 / 1e6);

    // Loopback only: the drawing is confidential.
    let listener = TcpListener::bind(("127.0.0.1", port)).unwrap_or_else(|e| {
        eprintln!("cannot listen on 127.0.0.1:{port}: {e}");
        std::process::exit(1);
    });
    println!("viewer: http://127.0.0.1:{port}/");
    for stream in listener.incoming().flatten() {
        let p = Arc::clone(&payload);
        std::thread::spawn(move || {
            if let Err(e) = serve(stream, &p) {
                eprintln!("request failed: {e}");
            }
        });
    }
}

fn serve(mut s: TcpStream, p: &Payload) -> std::io::Result<()> {
    let mut line = String::new();
    let mut r = BufReader::new(s.try_clone()?);
    r.read_line(&mut line)?;
    // drain headers
    let mut h = String::new();
    while r.read_line(&mut h)? > 2 {
        h.clear();
    }
    let target = line.split_whitespace().nth(1).unwrap_or("/");
    let info;
    let (status, ctype, body): (&str, &str, &[u8]) = match target {
        "/" | "/index.html" => ("200 OK", "text/html; charset=utf-8", INDEX.as_bytes()),
        "/meta.json" => ("200 OK", "application/json; charset=utf-8", &p.meta),
        "/positions.bin" => ("200 OK", "application/octet-stream", &p.positions),
        "/colors.bin" => ("200 OK", "application/octet-stream", &p.colors),
        "/ids.bin" => ("200 OK", "application/octet-stream", &p.ids),
        "/fillpos.bin" => ("200 OK", "application/octet-stream", &p.fill_positions),
        "/fillcol.bin" => ("200 OK", "application/octet-stream", &p.fill_colors),
        "/instances.bin" => ("200 OK", "application/octet-stream", &p.instances),
        "/instcol.bin" => ("200 OK", "application/octet-stream", &p.inst_colors),
        "/layers.bin" => ("200 OK", "application/octet-stream", &p.layers),
        "/filllayers.bin" => ("200 OK", "application/octet-stream", &p.fill_layers),
        "/instlayers.bin" => ("200 OK", "application/octet-stream", &p.inst_layers),
        t if t.starts_with("/entity?h=") => {
            info = u64::from_str_radix(&t["/entity?h=".len()..], 16)
                .ok()
                .and_then(|h| entity_info(&p.drawing, h));
            match &info {
                Some(j) => ("200 OK", "application/json; charset=utf-8", j.as_bytes()),
                None => ("404 Not Found", "text/plain", b"entity not found"),
            }
        }
        _ => ("404 Not Found", "text/plain", b"not found"),
    };
    write!(
        s,
        "HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
        body.len()
    )?;
    s.write_all(body)?;
    s.flush()
}
