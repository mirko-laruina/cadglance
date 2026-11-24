//! CadGlance desktop: viewer for .dwg layouts (wgpu + egui). Same look and tools as the web viewer.
//!
//! Usage:
//!   cadglance [file.dwg] [--software]           open the viewer (optionally on the CPU adapter)
//!   cadglance --bench [--software] file.dwg     headless timings, no window
//!
//! `--software` selects the CPU adapter (lavapipe on Linux, WARP on Windows): works without a GPU.

use std::path::{Path, PathBuf};
use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant};

use cadglance::container::Version;
use cadglance::desktop::{GpuScene, LayerLook, Look, Target, TextBox, View, load_font};
use cadglance::drawing::Drawing;
use cadglance::entities::P2;
use cadglance::geom::Scene;
use cadglance::info::{EntityInfo, InfoValue, entity_details};
use cadglance::pick::{Picker, SnapKind, apply, entity_segments, inst_xf};
use eframe::egui_wgpu::{WgpuConfiguration, WgpuSetup, WgpuSetupCreateNew};
use eframe::{egui, wgpu};
use egui::{Align2, Color32, CornerRadius, FontId, Key, Pos2, Rect, Response, RichText, Sense, Stroke, Ui, pos2, vec2};

const SHARP_DELAY: Duration = Duration::from_millis(150);
const CACHE_MARGIN: f32 = 2.0;
const PICK_PX: f32 = 8.0;
const MAX_RECENT: usize = 8;
const SIDE_W: f32 = 330.0;

// ---------- loading ----------

struct LoadedCpu {
    path: PathBuf,
    drawing: Drawing,
    scene: Scene,
    t_load: Duration,
    t_scene: Duration,
}

enum LoadMsg {
    Phase(&'static str),
    Done(Box<Result<LoadedCpu, String>>),
}

fn load(path: &Path, tx: &mpsc::Sender<LoadMsg>, ctx: &egui::Context) -> Result<LoadedCpu, String> {
    let phase = |p| {
        let _ = tx.send(LoadMsg::Phase(p));
        ctx.request_repaint();
    };
    let t0 = Instant::now();
    phase("Reading the file");
    let raw = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    phase("Decoding the DWG");
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let drawing = Drawing::load(&raw, threads).map_err(|e| format!("{}: {e}", path.display()))?;
    drop(raw);
    let t_load = t0.elapsed();
    let ms = drawing
        .blocks
        .values()
        .find(|b| b.name == "*Model_Space")
        .map(|b| b.handle)
        .ok_or_else(|| "the file has no model space (*Model_Space)".to_string())?;
    phase("Building the scene");
    let t1 = Instant::now();
    let scene = Scene::build(&drawing, ms);
    Ok(LoadedCpu { path: path.to_path_buf(), drawing, scene, t_load, t_scene: t1.elapsed() })
}

/// Minimal executor: wgpu's native futures resolve without a reactor.
fn block_on<F: std::future::Future>(f: F) -> F::Output {
    let mut f = std::pin::pin!(f);
    let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
    loop {
        if let std::task::Poll::Ready(v) = f.as_mut().poll(&mut cx) {
            return v;
        }
        std::thread::yield_now();
    }
}

fn limits_for(adapter: &wgpu::Adapter) -> wgpu::Limits {
    let a = adapter.limits();
    let base = if adapter.get_info().backend == wgpu::Backend::Gl {
        wgpu::Limits::downlevel_webgl2_defaults()
    } else {
        wgpu::Limits::default()
    };
    // big drawings need big vertex buffers; the cache needs large textures
    wgpu::Limits { max_buffer_size: a.max_buffer_size, max_texture_dimension_2d: a.max_texture_dimension_2d, ..base }
}

fn fit_view(bbox: [f64; 4], w: f64, h: f64) -> View {
    View {
        cx: (bbox[0] + bbox[2]) / 2.0,
        cy: (bbox[1] + bbox[3]) / 2.0,
        ppu: 0.95 * (w / (bbox[2] - bbox[0]).max(1e-9)).min(h / (bbox[3] - bbox[1]).max(1e-9)),
    }
}

fn default_layers(scene: &Scene) -> Vec<LayerLook> {
    scene.layer_list.iter().map(|(_, c, on)| LayerLook { on: *on, color: None, base: *c }).collect()
}

// ---------- headless benchmark ----------

fn bench(path: &Path, software: bool) -> Result<(), String> {
    let (tx, _rx) = mpsc::channel();
    let l = load(path, &tx, &egui::Context::default())?;
    println!("file {}: decode {:?}, scene {:?}", path.display(), l.t_load, l.t_scene);
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: software,
        compatible_surface: None,
    }))
    .map_err(|e| format!("no adapter: {e}"))?;
    let info = adapter.get_info();
    println!("adapter: {} ({:?}, {:?})", info.name, info.device_type, info.backend);
    let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("bench"),
        required_limits: limits_for(&adapter),
        ..Default::default()
    }))
    .map_err(|e| format!("no device: {e}"))?;
    let samples = if info.device_type == wgpu::DeviceType::Cpu { 1 } else { 4 };

    let t = Instant::now();
    let gpu = GpuScene::new(&device, &queue, &l.scene, samples, load_font().0);
    gpu.set_layers(&queue, &default_layers(&l.scene));
    queue.submit([]);
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    println!("GPU upload incl. text {:?} (MSAA {samples}x)", t.elapsed());

    let (ww, wh) = (1920.0, 1080.0);
    let max = device.limits().max_texture_dimension_2d;
    let target = gpu.target(&device, (ww as u32 * 2).min(max), (wh as u32 * 2).min(max));
    let fit = fit_view(l.scene.bbox, ww, wh);
    let look = Look { bg: [17, 20, 24], mono: None, adapt: false, texts: true, fills: true };
    for (name, v) in [
        ("whole drawing", fit),
        ("zoom x30", View { ppu: fit.ppu * 30.0, ..fit }),
        ("zoom x300", View { ppu: fit.ppu * 300.0, ..fit }),
    ] {
        let mut times = Vec::new();
        for _ in 0..3 {
            let t = Instant::now();
            let idx = gpu.render(&device, &queue, &target, v, &look, l.scene.layer0);
            let _ = device.poll(wgpu::PollType::Wait { submission_index: Some(idx), timeout: None });
            times.push(t.elapsed());
        }
        println!("sharp redraw {name} ({}x{}): {:?}", target.width, target.height, times);
    }
    let picker = Picker::new(&l.scene);
    let t = Instant::now();
    let c = [(l.scene.bbox[0] + l.scene.bbox[2]) / 2.0, (l.scene.bbox[1] + l.scene.bbox[3]) / 2.0];
    let _ = picker.pick(&l.scene, c, 8.0 / fit.ppu, &|_| true, true);
    println!("pick at the center (whole view): {:?}", t.elapsed());
    Ok(())
}

// ---------- look: palette, fonts, icons (same tokens as the web viewer) ----------

#[derive(Clone, Copy)]
struct Pal {
    light: bool,
    panel: Color32,
    panel2: Color32,
    line: Color32,
    fg: Color32,
    muted: Color32,
    accent: Color32,
    input: Color32,
}

impl Pal {
    fn new(light: bool) -> Pal {
        if light {
            Pal {
                light,
                panel: Color32::WHITE,
                panel2: Color32::from_rgb(0xf2, 0xf4, 0xf7),
                line: Color32::from_rgb(0xd5, 0xdb, 0xe2),
                fg: Color32::from_rgb(0x1d, 0x23, 0x2a),
                muted: Color32::from_rgb(0x5d, 0x68, 0x74),
                accent: Color32::from_rgb(0x1f, 0x6f, 0xd6),
                input: Color32::from_rgb(0xf2, 0xf4, 0xf7),
            }
        } else {
            Pal {
                light,
                panel: Color32::from_rgb(0x17, 0x1b, 0x20),
                panel2: Color32::from_rgb(0x1e, 0x23, 0x29),
                line: Color32::from_rgb(0x2c, 0x33, 0x3b),
                fg: Color32::from_rgb(0xd8, 0xdd, 0xe3),
                muted: Color32::from_rgb(0x8b, 0x95, 0xa1),
                accent: Color32::from_rgb(0x4e, 0xa1, 0xff),
                input: Color32::from_rgb(0x0c, 0x0f, 0x12),
            }
        }
    }

    fn shadow(&self) -> egui::Shadow {
        egui::Shadow { offset: [0, 6], blur: 24, spread: 0, color: Color32::from_black_alpha(if self.light { 40 } else { 90 }) }
    }

    /// Floating group: panel color, thin border, rounded, shadow.
    fn group(&self) -> egui::Frame {
        egui::Frame::new()
            .fill(self.panel)
            .stroke(Stroke::new(1.0, self.line))
            .corner_radius(8)
            .inner_margin(egui::Margin::same(4))
            .shadow(self.shadow())
    }
}

/// The interface font: Inter or Noto Sans (Linux), Segoe UI (Windows); else egui's own.
fn setup_fonts(ctx: &egui::Context) {
    let pick = |paths: &[&str]| paths.iter().find_map(|p| std::fs::read(p).ok());
    let regular = pick(&[
        "/usr/share/fonts/rsms-inter-fonts/Inter-Regular.ttf",
        "/usr/share/fonts/truetype/inter/Inter-Regular.ttf",
        "/usr/share/fonts/google-noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "C:\\Windows\\Fonts\\segoeui.ttf",
        "/Library/Fonts/Arial.ttf",
    ]);
    let strong = pick(&[
        "/usr/share/fonts/rsms-inter-fonts/Inter-SemiBold.ttf",
        "/usr/share/fonts/truetype/inter/Inter-SemiBold.ttf",
        "/usr/share/fonts/google-noto/NotoSans-SemiBold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-SemiBold.ttf",
        "C:\\Windows\\Fonts\\seguisb.ttf",
    ]);
    let mut fonts = egui::FontDefinitions::default();
    if let Some(b) = regular {
        fonts.font_data.insert("ui".into(), Arc::new(egui::FontData::from_owned(b)));
        fonts.families.entry(egui::FontFamily::Proportional).or_default().insert(0, "ui".into());
    }
    let strong_name = if strong.is_some() { "ui-strong" } else { "ui" };
    if let Some(b) = strong {
        fonts.font_data.insert("ui-strong".into(), Arc::new(egui::FontData::from_owned(b)));
    }
    let mut strong_family = vec![strong_name.to_string()];
    strong_family.extend(fonts.families.get(&egui::FontFamily::Proportional).cloned().unwrap_or_default());
    fonts.families.insert(egui::FontFamily::Name("strong".into()), strong_family);
    ctx.set_fonts(fonts);
}

fn strong_font(size: f32) -> FontId {
    FontId::new(size, egui::FontFamily::Name("strong".into()))
}

fn apply_style(ctx: &egui::Context, p: &Pal) {
    ctx.style_mut(|s| {
        use egui::TextStyle as T;
        s.text_styles = [
            (T::Heading, strong_font(15.0)),
            (T::Body, FontId::proportional(13.0)),
            (T::Button, FontId::proportional(13.0)),
            (T::Small, FontId::proportional(11.0)),
            (T::Monospace, FontId::monospace(12.5)),
        ]
        .into();
        s.spacing.item_spacing = vec2(6.0, 6.0);
        s.spacing.button_padding = vec2(8.0, 4.0);
        s.spacing.interact_size.y = 26.0;
        s.spacing.window_margin = egui::Margin::same(14);
        let v = &mut s.visuals;
        *v = if p.light { egui::Visuals::light() } else { egui::Visuals::dark() };
        v.panel_fill = p.panel;
        v.window_fill = p.panel;
        v.window_stroke = Stroke::new(1.0, p.line);
        v.window_corner_radius = CornerRadius::same(10);
        v.menu_corner_radius = CornerRadius::same(8);
        v.window_shadow = p.shadow();
        v.popup_shadow = p.shadow();
        v.extreme_bg_color = p.input;
        v.faint_bg_color = p.panel2;
        v.selection.bg_fill = p.accent;
        v.selection.stroke = Stroke::new(1.0, Color32::WHITE);
        v.hyperlink_color = p.accent;
        let w = &mut v.widgets;
        for (st, fill, weak, stroke) in [
            (&mut w.noninteractive, p.panel, p.panel, p.line),
            (&mut w.inactive, p.panel2, Color32::TRANSPARENT, p.line),
            (&mut w.hovered, p.panel2, p.panel2, p.line),
            (&mut w.active, p.line, p.line, p.line),
            (&mut w.open, p.panel2, p.panel2, p.line),
        ] {
            st.bg_fill = fill;
            st.weak_bg_fill = weak;
            st.bg_stroke = Stroke::new(1.0, stroke);
            st.fg_stroke = Stroke::new(1.3, p.fg);
            st.corner_radius = CornerRadius::same(6);
            st.expansion = 0.0;
        }
        w.noninteractive.bg_stroke = Stroke::new(1.0, p.line);
    });
}

#[derive(Clone, Copy)]
enum Icon {
    Info,
    Measure,
    Area,
    Zoom,
    Fit,
    Clear,
    Open,
    Recent,
    Panel,
    Png,
    Help,
    Snap,
}

/// Same shapes as the web viewer's SVG icons (24x24 view box).
fn paint_icon(painter: &egui::Painter, r: Rect, icon: Icon, c: Color32) {
    let s = r.width() / 24.0;
    let p = |x: f32, y: f32| pos2(r.min.x + x * s, r.min.y + y * s);
    let st = Stroke::new(1.7, c);
    let line = |pts: &[(f32, f32)]| {
        painter.add(egui::Shape::line(pts.iter().map(|&(x, y)| p(x, y)).collect(), st));
    };
    let closed = |pts: &[(f32, f32)]| {
        painter.add(egui::Shape::closed_line(pts.iter().map(|&(x, y)| p(x, y)).collect(), st));
    };
    match icon {
        Icon::Info => closed(&[(4.0, 3.0), (11.0, 20.0), (13.5, 12.5), (21.0, 10.0)]),
        Icon::Measure => {
            closed(&[(3.0, 17.0), (17.0, 3.0), (21.0, 7.0), (7.0, 21.0)]);
            for (a, b) in [((7.0, 13.0), (9.0, 15.0)), ((10.0, 10.0), (12.0, 12.0)), ((13.0, 7.0), (15.0, 9.0))] {
                line(&[a, b]);
            }
        }
        Icon::Area => closed(&[(4.0, 19.0), (7.0, 6.0), (18.0, 8.0), (20.0, 17.0)]),
        Icon::Zoom => {
            line(&[(10.0, 3.0), (3.0, 3.0), (3.0, 10.0)]);
            line(&[(14.0, 21.0), (21.0, 21.0), (21.0, 14.0)]);
            line(&[(8.0, 8.0), (16.0, 16.0)]);
        }
        Icon::Fit => {
            line(&[(4.0, 9.0), (4.0, 4.0), (9.0, 4.0)]);
            line(&[(15.0, 4.0), (20.0, 4.0), (20.0, 9.0)]);
            line(&[(20.0, 15.0), (20.0, 20.0), (15.0, 20.0)]);
            line(&[(9.0, 20.0), (4.0, 20.0), (4.0, 15.0)]);
        }
        Icon::Clear => {
            line(&[(4.0, 7.0), (20.0, 7.0)]);
            line(&[(10.0, 11.0), (10.0, 17.0)]);
            line(&[(14.0, 11.0), (14.0, 17.0)]);
            line(&[(6.0, 7.0), (7.0, 20.0), (17.0, 20.0), (18.0, 7.0)]);
            line(&[(9.0, 7.0), (9.0, 4.0), (15.0, 4.0), (15.0, 7.0)]);
        }
        Icon::Open => closed(&[(3.0, 6.0), (9.0, 6.0), (11.0, 8.0), (21.0, 8.0), (21.0, 19.0), (3.0, 19.0)]),
        Icon::Recent => {
            painter.circle_stroke(p(12.0, 12.0), 9.0 * s, st);
            line(&[(12.0, 7.0), (12.0, 12.0), (15.0, 14.0)]);
        }
        Icon::Panel => {
            closed(&[(3.0, 4.0), (21.0, 4.0), (21.0, 20.0), (3.0, 20.0)]);
            line(&[(15.0, 4.0), (15.0, 20.0)]);
        }
        Icon::Png => {
            line(&[(12.0, 3.0), (12.0, 15.0)]);
            line(&[(7.0, 10.0), (12.0, 15.0), (17.0, 10.0)]);
            line(&[(4.0, 20.0), (20.0, 20.0)]);
        }
        Icon::Help => {
            painter.circle_stroke(p(12.0, 12.0), 9.0 * s, st);
            painter.text(p(12.0, 12.5), Align2::CENTER_CENTER, "?", strong_font(16.0 * s), c);
        }
        Icon::Snap => {
            closed(&[(6.0, 6.0), (18.0, 6.0), (18.0, 18.0), (6.0, 18.0)]);
            for (a, b) in [((12.0, 2.0), (12.0, 6.0)), ((12.0, 18.0), (12.0, 22.0)), ((2.0, 12.0), (6.0, 12.0)), ((18.0, 12.0), (22.0, 12.0))] {
                line(&[a, b]);
            }
        }
    }
}

/// Toolbar button as in the web viewer: transparent, hover tint, accent when pressed.
fn tool_button(ui: &mut Ui, p: &Pal, icon: Option<Icon>, label: Option<&str>, pressed: bool, tip: &str) -> Response {
    let fg = if pressed { Color32::WHITE } else { p.fg };
    let galley = label.map(|l| ui.painter().layout_no_wrap(l.to_string(), FontId::proportional(13.0), fg));
    let icon_w = if icon.is_some() { 16.0 } else { 0.0 };
    let gap = if icon.is_some() && galley.is_some() { 6.0 } else { 0.0 };
    let text_w = galley.as_ref().map_or(0.0, |g| g.size().x);
    let (rect, resp) = ui.allocate_exact_size(vec2(icon_w + gap + text_w + 16.0, 28.0), Sense::click());
    if ui.is_rect_visible(rect) {
        let bg = if pressed {
            p.accent
        } else if resp.hovered() {
            p.panel2
        } else {
            Color32::TRANSPARENT
        };
        ui.painter().rect_filled(rect, 6.0, bg);
        let mut x = rect.min.x + 8.0;
        if let Some(i) = icon {
            paint_icon(ui.painter(), Rect::from_min_size(pos2(x, rect.center().y - 8.0), vec2(16.0, 16.0)), i, fg);
            x += 16.0 + gap;
        }
        if let Some(g) = galley {
            ui.painter().galley(pos2(x, rect.center().y - g.size().y / 2.0), g, fg);
        }
    }
    if tip.is_empty() { resp } else { resp.on_hover_text(tip) }
}

fn separator(ui: &mut Ui, p: &Pal) {
    let (r, _) = ui.allocate_exact_size(vec2(9.0, 22.0), Sense::hover());
    ui.painter().line_segment([pos2(r.center().x, r.min.y), pos2(r.center().x, r.max.y)], Stroke::new(1.0, p.line));
}

fn section(ui: &mut Ui, p: &Pal, title: &str) {
    ui.add_space(10.0);
    ui.label(RichText::new(title.to_uppercase()).size(11.0).color(p.muted).extra_letter_spacing(0.5));
    ui.add_space(2.0);
}

fn hline(ui: &mut Ui, p: &Pal) {
    let w = ui.available_width();
    let (r, _) = ui.allocate_exact_size(vec2(w, 1.0), Sense::hover());
    ui.painter().rect_filled(r, 0.0, p.line);
}

/// Two-column table like the web viewer's: muted keys, right-aligned values, row separators.
/// Release year of a file format, from its version string.
fn format_year(v: &str) -> &'static str {
    Version::from_magic(v).map_or("unknown", Version::year)
}

fn kv_table(ui: &mut Ui, p: &Pal, rows: &[(String, String)]) {
    for (k, v) in rows {
        hline(ui, p);
        ui.horizontal_top(|ui| {
            ui.label(RichText::new(k).color(p.muted));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
                ui.add(egui::Label::new(RichText::new(v).color(p.fg)).wrap());
            });
        });
    }
}

// ---------- settings ----------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Theme {
    Dark,
    Light,
    MonoDark,
    MonoLight,
}

impl Theme {
    const ALL: [Theme; 4] = [Theme::Dark, Theme::Light, Theme::MonoDark, Theme::MonoLight];
    fn name(self) -> &'static str {
        match self {
            Theme::Dark => "Dark",
            Theme::Light => "Light",
            Theme::MonoDark => "Mono dark",
            Theme::MonoLight => "Mono light",
        }
    }
    fn key(self) -> &'static str {
        match self {
            Theme::Dark => "dark",
            Theme::Light => "light",
            Theme::MonoDark => "monoDark",
            Theme::MonoLight => "monoLight",
        }
    }
    fn bg(self) -> [u8; 3] {
        match self {
            Theme::Dark | Theme::MonoDark => [17, 20, 24],
            Theme::Light | Theme::MonoLight => [255, 255, 255],
        }
    }
    fn ink(self) -> Option<[u8; 3]> {
        match self {
            Theme::MonoDark => Some([232, 235, 239]),
            Theme::MonoLight => Some([26, 26, 26]),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tab {
    Props,
    Layers,
    Look,
}

#[derive(Clone)]
struct Settings {
    theme: Theme,
    bg: [u8; 3],
    mono: [u8; 3],
    sel: [u8; 3],
    adapt: bool,
    texts: bool,
    fills: bool,
    snap: bool,
    panel: bool,
    recent: Vec<PathBuf>,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            theme: Theme::Dark,
            bg: Theme::Dark.bg(),
            mono: [232, 235, 239],
            sel: [255, 176, 32],
            adapt: true,
            texts: true,
            fills: true,
            snap: true,
            panel: true,
            recent: Vec::new(),
        }
    }
}

fn config_path() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("APPDATA").map(PathBuf::from))
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))?;
    Some(base.join("cadglance").join("settings.txt"))
}

fn hex(c: [u8; 3]) -> String {
    format!("#{:02x}{:02x}{:02x}", c[0], c[1], c[2])
}

fn parse_hex(s: &str) -> Option<[u8; 3]> {
    let s = s.strip_prefix('#')?;
    let v = u32::from_str_radix(s, 16).ok()?;
    (s.len() == 6).then_some([(v >> 16) as u8, (v >> 8) as u8, v as u8])
}

impl Settings {
    fn load() -> Settings {
        let mut s = Settings::default();
        let Some(text) = config_path().and_then(|p| std::fs::read_to_string(p).ok()) else { return s };
        for line in text.lines() {
            let Some((k, v)) = line.split_once('=') else { continue };
            let b = v == "true";
            match k {
                "theme" => s.theme = Theme::ALL.into_iter().find(|t| t.key() == v).unwrap_or(Theme::Dark),
                "bg" => s.bg = parse_hex(v).unwrap_or(s.bg),
                "mono" => s.mono = parse_hex(v).unwrap_or(s.mono),
                "sel" => s.sel = parse_hex(v).unwrap_or(s.sel),
                "adapt" => s.adapt = b,
                "texts" => s.texts = b,
                "fills" => s.fills = b,
                "snap" => s.snap = b,
                "panel" => s.panel = b,
                "recent" => s.recent.push(PathBuf::from(v)),
                _ => {}
            }
        }
        s
    }

    fn save(&self) {
        let Some(p) = config_path() else { return };
        let mut out = format!(
            "theme={}\nbg={}\nmono={}\nsel={}\nadapt={}\ntexts={}\nfills={}\nsnap={}\npanel={}\n",
            self.theme.key(),
            hex(self.bg),
            hex(self.mono),
            hex(self.sel),
            self.adapt,
            self.texts,
            self.fills,
            self.snap,
            self.panel
        );
        for r in &self.recent {
            out.push_str(&format!("recent={}\n", r.display()));
        }
        if let Some(dir) = p.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let _ = std::fs::write(p, out);
    }

    fn bg_light(&self) -> bool {
        let c = self.bg;
        (0.2126 * c[0] as f32 + 0.7152 * c[1] as f32 + 0.0722 * c[2] as f32) / 255.0 > 0.5
    }

    fn look(&self) -> Look {
        Look {
            bg: self.bg,
            mono: matches!(self.theme, Theme::MonoDark | Theme::MonoLight).then_some(self.mono),
            adapt: self.adapt && self.bg_light(),
            texts: self.texts,
            fills: self.fills,
        }
    }

    fn sel_color(&self) -> Color32 {
        Color32::from_rgb(self.sel[0], self.sel[1], self.sel[2])
    }

    fn add_recent(&mut self, p: &Path) {
        self.recent.retain(|r| r != p);
        self.recent.insert(0, p.to_path_buf());
        self.recent.truncate(MAX_RECENT);
    }
}

// ---------- document ----------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tool {
    Info,
    Measure,
    Area,
    Zoom,
}

impl Tool {
    fn hint(self) -> &'static str {
        match self {
            Tool::Info => "Click an element to see its measurements.",
            Tool::Measure => "Click the first point, then the second. Shift = orthogonal.",
            Tool::Area => "Click the vertices. Enter or double click to close, ⌫ removes the last.",
            Tool::Zoom => "Drag a rectangle to zoom in.",
        }
    }
}

struct Selection {
    segs: Vec<[P2; 2]>,
    quad: Option<[P2; 4]>,
    info: Option<EntityInfo>,
    batch: usize,
    inst: usize,
}

struct LayerState {
    name: String,
    dwg_on: bool,
    look: LayerLook,
}

struct Doc {
    name: String,
    drawing: Drawing,
    scene: Scene,
    picker: Picker,
    gpu: GpuScene,
    target: Option<Target>,
    tex: Option<egui::TextureId>,
    view: View,
    /// View and content version of the cached image.
    cache: Option<(View, u64)>,
    version: u64,
    last_change: Instant,
    need_fit: bool,
    layers: Vec<LayerState>,
    layer_filter: String,
    /// Drawing info rows: file, format, contents, and the static rendering numbers.
    file_info: Vec<(String, String)>,
    render_info: Vec<(String, String)>,
    last_sharp: Duration,
    selection: Option<Selection>,
    measures: Vec<[P2; 2]>,
    areas: Vec<Vec<P2>>,
    pending: Vec<P2>,
    hover: Option<(P2, Option<SnapKind>)>,
    query: String,
    matches: Vec<usize>,
    match_index: Option<usize>,
    canvas: Rect,
}

impl Doc {
    fn upload_layers(&mut self, queue: &wgpu::Queue) {
        let looks: Vec<LayerLook> = self.layers.iter().map(|l| l.look).collect();
        self.gpu.set_layers(queue, &looks);
        self.version += 1;
    }

    fn layer_on(&self, l: u16) -> bool {
        self.layers.get(l as usize).is_none_or(|s| s.look.on)
    }

    fn text_visible(&self, b: &TextBox, texts: bool) -> bool {
        texts && self.layer_on(b.layer)
    }

    fn run_search(&mut self, texts: bool) {
        let q = self.query.trim().to_lowercase();
        self.matches = if q.is_empty() {
            Vec::new()
        } else {
            (0..self.gpu.text_boxes.len())
                .filter(|&i| {
                    let b = &self.gpu.text_boxes[i];
                    self.text_visible(b, texts) && self.scene.texts[b.text].value.to_lowercase().contains(&q)
                })
                .collect()
        };
        self.match_index = None;
    }

    fn go_to_match(&mut self, step: isize, ppp: f64) {
        let n = self.matches.len() as isize;
        if n == 0 {
            return;
        }
        let i = match self.match_index {
            Some(i) => (i as isize + step).rem_euclid(n),
            None if step > 0 => 0,
            None => n - 1,
        } as usize;
        self.match_index = Some(i);
        let c = self.gpu.text_boxes[self.matches[i]].corners;
        let (x0, x1) = c.iter().fold((f64::MAX, f64::MIN), |a, p| (a.0.min(p[0]), a.1.max(p[0])));
        let (y0, y1) = c.iter().fold((f64::MAX, f64::MIN), |a, p| (a.0.min(p[1]), a.1.max(p[1])));
        self.view.cx = (x0 + x1) / 2.0;
        self.view.cy = (y0 + y1) / 2.0;
        let (pw, ph) = (self.canvas.width() as f64 * ppp, self.canvas.height() as f64 * ppp);
        let target = (pw * 0.25 / (x1 - x0).max(1e-9)).min(ph * 0.25 / (y1 - y0).max(1e-9));
        if self.view.ppu < target * 0.5 || self.view.ppu > target * 8.0 {
            self.view.ppu = target;
        }
        self.last_change = Instant::now();
    }

    fn clear_measures(&mut self) {
        self.measures.clear();
        self.areas.clear();
        self.pending.clear();
    }
}

// ---------- app ----------

struct App {
    settings: Settings,
    doc: Option<Doc>,
    loading: Option<(mpsc::Receiver<LoadMsg>, &'static str, PathBuf)>,
    /// CPU data waiting for the GPU upload (done on the next frame, after showing the phase).
    upload: Option<LoadedCpu>,
    error: Option<String>,
    tool: Tool,
    tab: Tab,
    show_help: bool,
    /// Help dialog tab: false = usage and shortcuts, true = file and drawing info.
    help_info: bool,
    show_recent: Option<Pos2>,
    adapter: String,
    samples: u32,
    font: Option<ab_glyph::FontArc>,
    zoom_rect: Option<(Pos2, Pos2)>,
    screenshot_rect: Option<Rect>,
    /// PNG export: the UI around the drawing is hidden until the screenshot arrives.
    capturing: bool,
    focus_search: bool,
    styled_light: Option<bool>,
    /// Top of the bottom bar in the previous frame (0 = unknown).
    status_top: f32,
}

fn group_thousands(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, ch) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

fn fmt(v: f64, d: usize) -> String {
    let s = format!("{:.*}", d, v.abs());
    let (int, frac) = s.split_once('.').unwrap_or((&s, ""));
    let grouped = group_thousands(int.parse().unwrap_or(0));
    let sign = if v < 0.0 && v.abs() >= 0.5 * 10f64.powi(-(d as i32)) { "-" } else { "" };
    if frac.is_empty() { format!("{sign}{grouped}") } else { format!("{sign}{grouped}.{frac}") }
}

// world (relative to origin) <-> screen points
struct Xform {
    rect: Rect,
    view: View,
    ppp: f64,
}

impl Xform {
    fn to_world(&self, p: Pos2) -> P2 {
        [
            self.view.cx + (p.x - self.rect.center().x) as f64 * self.ppp / self.view.ppu,
            self.view.cy + (self.rect.center().y - p.y) as f64 * self.ppp / self.view.ppu,
        ]
    }
    fn to_screen(&self, w: P2) -> Pos2 {
        pos2(
            self.rect.center().x + ((w[0] - self.view.cx) * self.view.ppu / self.ppp) as f32,
            self.rect.center().y - ((w[1] - self.view.cy) * self.view.ppu / self.ppp) as f32,
        )
    }
    /// World units per screen point.
    fn unit(&self) -> f64 {
        self.ppp / self.view.ppu
    }
}

fn point_in_quad(q: &[P2; 4], x: f64, y: f64) -> bool {
    let mut inside = false;
    let mut j = 3;
    for i in 0..4 {
        let (xi, yi, xj, yj) = (q[i][0], q[i][1], q[j][0], q[j][1]);
        if (yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi {
            inside = !inside;
        }
        j = i;
    }
    inside
}

fn poly_area(p: &[P2]) -> f64 {
    let n = p.len();
    (0..n).map(|i| p[i][0] * p[(i + 1) % n][1] - p[(i + 1) % n][0] * p[i][1]).sum::<f64>().abs() / 2.0
}

fn poly_perimeter(p: &[P2]) -> f64 {
    let n = p.len();
    (0..n).map(|i| (p[(i + 1) % n][0] - p[i][0]).hypot(p[(i + 1) % n][1] - p[i][1])).sum()
}

fn save_png(img: &egui::ColorImage, r: Rect, ppp: f32, name: &str) {
    let [w, h] = img.size;
    let (x0, y0) = (((r.min.x * ppp) as usize).min(w), ((r.min.y * ppp) as usize).min(h));
    let (x1, y1) = (((r.max.x * ppp) as usize).min(w), ((r.max.y * ppp) as usize).min(h));
    if x1 <= x0 || y1 <= y0 {
        return;
    }
    let mut rgba = Vec::with_capacity((x1 - x0) * (y1 - y0) * 4);
    for y in y0..y1 {
        for x in x0..x1 {
            rgba.extend_from_slice(&img.pixels[y * w + x].to_srgba_unmultiplied());
        }
    }
    let stem = Path::new(name).file_stem().map_or_else(|| "view".into(), |s| s.to_string_lossy().into_owned());
    if let Some(p) = rfd::FileDialog::new().add_filter("PNG image", &["png"]).set_file_name(format!("{stem}-view.png")).save_file() {
        let _ = std::fs::write(p, cadglance::png::encode_rgba((x1 - x0) as u32, (y1 - y0) as u32, &rgba));
    }
}

impl App {
    fn open(&mut self, path: PathBuf, ctx: &egui::Context) {
        let (tx, rx) = mpsc::channel();
        let c = ctx.clone();
        let p = path.clone();
        std::thread::spawn(move || {
            let res = load(&p, &tx, &c);
            let _ = tx.send(LoadMsg::Done(Box::new(res)));
            c.request_repaint();
        });
        self.loading = Some((rx, "Reading the file", path));
        self.error = None;
        self.show_recent = None;
    }

    fn pick_file(&mut self, ctx: &egui::Context) {
        let mut dlg = rfd::FileDialog::new().add_filter("DWG drawings", &["dwg", "DWG"]);
        if let Some(dir) = self.settings.recent.first().and_then(|p| p.parent()) {
            dlg = dlg.set_directory(dir);
        }
        if let Some(p) = dlg.pick_file() {
            self.open(p, ctx);
        }
    }

    fn finish_upload(&mut self, l: LoadedCpu, rs: &eframe::egui_wgpu::RenderState, ctx: &egui::Context) {
        let t = Instant::now();
        let font = self.font.get_or_insert_with(|| load_font().0).clone();
        let gpu = GpuScene::new(&rs.device, &rs.queue, &l.scene, self.samples, font);
        let layers: Vec<LayerState> = l
            .scene
            .layer_list
            .iter()
            .map(|(n, c, on)| LayerState { name: n.clone(), dwg_on: *on, look: LayerLook { on: *on, color: None, base: *c } })
            .collect();
        let _ = rs.device.poll(wgpu::PollType::wait_indefinitely());
        let name = l.path.file_name().map_or_else(String::new, |n| n.to_string_lossy().into_owned());
        let d = &l.drawing;
        let mb = std::fs::metadata(&l.path).map_or(0.0, |m| m.len() as f64 / 1e6);
        let rows = |v: &[(&str, String)]| v.iter().map(|(k, v)| (k.to_string(), v.clone())).collect::<Vec<_>>();
        let file_info = rows(&[
            ("File", l.path.display().to_string()),
            ("Size", format!("{mb:.1} MB")),
            ("Format", format!("{} ({} format)", d.version, format_year(&d.version))),
            ("Objects", group_thousands(d.types.len() as u64)),
            ("Entities", group_thousands(d.entities.len() as u64)),
            ("Layers · blocks · layouts", format!("{} · {} · {}", d.layers.len(), d.blocks.len(), d.layouts.len())),
            ("Objects not decoded", group_thousands(d.errors.len() as u64)),
        ]);
        let render_info = rows(&[
            ("Segments", group_thousands(l.scene.segments_drawn)),
            ("Triangles", group_thousands(l.scene.triangles_drawn)),
            ("Texts", group_thousands(gpu.text_boxes.len() as u64)),
            ("Decode · scene · GPU", format!("{:.2} s · {:.2} s · {:.2} s", l.t_load.as_secs_f64(), l.t_scene.as_secs_f64(), t.elapsed().as_secs_f64())),
        ]);
        if let Some(old) = self.doc.take().and_then(|d| d.tex) {
            rs.renderer.write().free_texture(&old);
        }
        ctx.send_viewport_cmd(egui::ViewportCommand::Title(format!("{name} — CadGlance")));
        self.settings.add_recent(&l.path);
        self.settings.save();
        let picker = Picker::new(&l.scene);
        let mut doc = Doc {
            name,
            drawing: l.drawing,
            scene: l.scene,
            picker,
            gpu,
            target: None,
            tex: None,
            view: View { cx: 0.0, cy: 0.0, ppu: 1.0 },
            cache: None,
            version: 0,
            last_change: Instant::now() - SHARP_DELAY,
            need_fit: true,
            layers,
            layer_filter: String::new(),
            file_info,
            render_info,
            last_sharp: Duration::ZERO,
            selection: None,
            measures: Vec::new(),
            areas: Vec::new(),
            pending: Vec::new(),
            hover: None,
            query: String::new(),
            matches: Vec::new(),
            match_index: None,
            canvas: Rect::NOTHING,
        };
        doc.upload_layers(&rs.queue);
        self.doc = Some(doc);
    }

    fn set_tool(&mut self, t: Tool) {
        self.tool = t;
        if let Some(d) = &mut self.doc {
            d.pending.clear();
            d.hover = None;
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        let Some(rs) = frame.wgpu_render_state().cloned() else { return };
        let light = self.settings.bg_light();
        let pal = Pal::new(light);
        if self.styled_light != Some(light) {
            apply_style(ctx, &pal);
            self.styled_light = Some(light);
        }

        // loading: worker phases -> CPU data -> GPU upload on the next frame
        if let Some(l) = self.upload.take() {
            self.finish_upload(l, &rs, ctx);
            self.loading = None;
        }
        let mut done = None;
        if let Some((rx, phase, _)) = &mut self.loading {
            while let Ok(m) = rx.try_recv() {
                match m {
                    LoadMsg::Phase(p) => *phase = p,
                    LoadMsg::Done(r) => done = Some(*r),
                }
            }
        }
        match done {
            Some(Ok(l)) => {
                if let Some((_, phase, _)) = &mut self.loading {
                    *phase = "Uploading to the graphics card";
                }
                self.upload = Some(l);
                ctx.request_repaint();
            }
            Some(Err(e)) => {
                self.loading = None;
                self.error = Some(e);
            }
            None => {}
        }

        if let Some(p) = ctx.input(|i| i.raw.dropped_files.iter().find_map(|f| f.path.clone())) {
            self.open(p, ctx);
        }
        let shot = ctx.input(|i| {
            i.raw.events.iter().find_map(|e| match e {
                egui::Event::Screenshot { image, .. } => Some(image.clone()),
                _ => None,
            })
        });
        if let (Some(img), Some(r)) = (shot, self.screenshot_rect.take()) {
            self.capturing = false;
            save_png(&img, r, ctx.pixels_per_point(), self.doc.as_ref().map_or("view", |d| d.name.as_str()));
        }
        self.capturing &= self.doc.is_some();

        self.keyboard(ctx);
        let bg = Color32::from_rgb(self.settings.bg[0], self.settings.bg[1], self.settings.bg[2]);
        egui::CentralPanel::default().frame(egui::Frame::NONE.fill(bg)).show(ctx, |ui| self.canvas(ui, ctx, &rs, &pal));
        if self.capturing {
            // Ask for the screenshot only from a frame drawn without the UI; keep hiding it until the image arrives.
            if self.screenshot_rect.is_none() {
                ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
                self.screenshot_rect = self.doc.as_ref().map(|d| d.canvas);
            }
            ctx.request_repaint();
            return;
        }
        self.toolbars(ctx, &pal);
        if self.settings.panel && self.doc.is_some() {
            self.side_panel(ctx, &rs, &pal);
        }
        self.status(ctx, &pal);
        self.recent_menu(ctx, &pal);
        if self.show_help {
            self.help_window(ctx, &pal);
        }
        // With a drawing open the canvas keeps it on screen, so a failed open needs its own dialog.
        if self.doc.is_some() && self.error.is_some() {
            self.error_window(ctx, &pal);
        }
    }
}

impl App {
    fn keyboard(&mut self, ctx: &egui::Context) {
        if ctx.input_mut(|i| i.consume_shortcut(&egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, Key::O))) {
            self.pick_file(ctx);
        }
        if ctx.input_mut(|i| i.consume_shortcut(&egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, Key::F))) {
            self.focus_search = true;
        }
        if ctx.wants_keyboard_input() {
            return;
        }
        let pressed = |k: Key| ctx.input(|i| i.key_pressed(k) && !i.modifiers.command && !i.modifiers.alt);
        if pressed(Key::Slash) {
            self.focus_search = true;
        }
        for (k, t) in [(Key::I, Tool::Info), (Key::M, Tool::Measure), (Key::A, Tool::Area), (Key::Z, Tool::Zoom)] {
            if pressed(k) {
                self.set_tool(t);
            }
        }
        if pressed(Key::S) {
            self.settings.snap = !self.settings.snap;
            self.settings.save();
        }
        if pressed(Key::L) {
            self.settings.panel = !self.settings.panel;
            self.settings.save();
        }
        if pressed(Key::F1) || ctx.input(|i| i.events.iter().any(|e| matches!(e, egui::Event::Text(t) if t == "?"))) {
            self.show_help = !self.show_help;
        }
        if pressed(Key::Escape) && self.show_help {
            self.show_help = false;
            return;
        }
        if pressed(Key::Escape) && self.doc.is_some() && self.error.is_some() {
            self.error = None;
            return;
        }
        let Some(d) = &mut self.doc else { return };
        if pressed(Key::F) {
            d.need_fit = true;
        }
        if pressed(Key::Escape) {
            if !d.pending.is_empty() {
                d.pending.clear();
            } else if self.zoom_rect.is_some() {
                self.zoom_rect = None;
            } else {
                d.selection = None;
            }
        }
        if pressed(Key::Delete) {
            d.clear_measures();
        }
        if pressed(Key::Enter) && self.tool == Tool::Area && d.pending.len() > 2 {
            d.areas.push(std::mem::take(&mut d.pending));
        }
        if pressed(Key::Backspace) && self.tool == Tool::Area {
            d.pending.pop();
        }
    }

    fn toolbars(&mut self, ctx: &egui::Context, p: &Pal) {
        let ppp = ctx.pixels_per_point() as f64;
        egui::Area::new(egui::Id::new("top-left")).anchor(Align2::LEFT_TOP, vec2(12.0, 12.0)).movable(false).show(ctx, |ui| {
            ui.horizontal(|ui| {
                p.group().show(ui, |ui| {
                    ui.spacing_mut().item_spacing.x = 2.0;
                    ui.horizontal(|ui| {
                        if tool_button(ui, p, Some(Icon::Open), Some("Open"), false, "Open a DWG file (Ctrl+O)").clicked() {
                            self.pick_file(ctx);
                        }
                        let r = tool_button(ui, p, Some(Icon::Recent), None, self.show_recent.is_some(), "Recent files");
                        if r.clicked() {
                            self.show_recent = if self.show_recent.is_some() { None } else { Some(r.rect.left_bottom() + vec2(0.0, 8.0)) };
                        }
                        separator(ui, p);
                        for (t, icon, label, tip) in [
                            (Tool::Info, Icon::Info, "Info", "Select an element and read its measurements (I)"),
                            (Tool::Measure, Icon::Measure, "Distance", "Distance between two points (M)"),
                            (Tool::Area, Icon::Area, "Area", "Area and perimeter of a polygon (A)"),
                            (Tool::Zoom, Icon::Zoom, "Zoom", "Zoom to a rectangle (Z, or Shift+drag)"),
                        ] {
                            if tool_button(ui, p, Some(icon), Some(label), self.tool == t, tip).clicked() {
                                self.set_tool(t);
                            }
                        }
                        separator(ui, p);
                        if tool_button(ui, p, Some(Icon::Fit), Some("Fit"), false, "Fit the drawing to the window (F, double click)").clicked() {
                            if let Some(d) = &mut self.doc {
                                d.need_fit = true;
                            }
                        }
                        if tool_button(ui, p, Some(Icon::Clear), Some("Clear"), false, "Clear measurements and areas (Del)").clicked() {
                            if let Some(d) = &mut self.doc {
                                d.clear_measures();
                            }
                        }
                    });
                });
                let texts = self.settings.texts;
                if let Some(d) = &mut self.doc {
                    p.group().show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.add_space(4.0);
                            let r = ui.add(
                                egui::TextEdit::singleline(&mut d.query)
                                    .hint_text("Search text…  ( / )")
                                    .frame(false)
                                    .desired_width(200.0)
                                    .margin(vec2(4.0, 5.0)),
                            );
                            if self.focus_search {
                                r.request_focus();
                                self.focus_search = false;
                            }
                            if r.changed() {
                                d.run_search(texts);
                            }
                            if r.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter)) {
                                let back = ui.input(|i| i.modifiers.shift);
                                d.go_to_match(if back { -1 } else { 1 }, ppp);
                                r.request_focus();
                            }
                            if !d.query.is_empty() {
                                let count = match d.match_index {
                                    Some(i) => format!("{} / {}", i + 1, d.matches.len()),
                                    None => format!("{} results", d.matches.len()),
                                };
                                ui.label(RichText::new(count).color(p.muted));
                            }
                            if tool_button(ui, p, None, Some("◀"), false, "Previous result (Shift+Enter)").clicked() {
                                d.go_to_match(-1, ppp);
                            }
                            if tool_button(ui, p, None, Some("▶"), false, "Next result (Enter)").clicked() {
                                d.go_to_match(1, ppp);
                            }
                        });
                    });
                }
            });
        });
        egui::Area::new(egui::Id::new("top-right")).anchor(Align2::RIGHT_TOP, vec2(-12.0, 12.0)).movable(false).show(ctx, |ui| {
            p.group().show(ui, |ui| {
                ui.spacing_mut().item_spacing.x = 2.0;
                ui.horizontal(|ui| {
                    if tool_button(ui, p, Some(Icon::Panel), Some("Panel"), self.settings.panel, "Side panel (L)").clicked() {
                        self.settings.panel = !self.settings.panel;
                        self.settings.save();
                    }
                    if tool_button(ui, p, Some(Icon::Png), Some("PNG"), false, "Save the view as a PNG image").clicked() {
                        self.capturing = true;
                        ctx.request_repaint();
                    }
                    if tool_button(ui, p, Some(Icon::Help), None, self.show_help, "Commands and drawing info (F1, ?)").clicked() {
                        self.show_help = !self.show_help;
                    }
                });
            });
        });
    }

    fn recent_menu(&mut self, ctx: &egui::Context, p: &Pal) {
        let Some(at) = self.show_recent else { return };
        let mut open = None;
        let r = egui::Area::new(egui::Id::new("recent")).fixed_pos(at).order(egui::Order::Foreground).show(ctx, |ui| {
            p.group().inner_margin(egui::Margin::same(8)).show(ui, |ui| {
                ui.set_min_width(260.0);
                ui.label(RichText::new("RECENT FILES").size(11.0).color(p.muted));
                if self.settings.recent.is_empty() {
                    ui.label(RichText::new("No recent files").color(p.muted));
                }
                for path in &self.settings.recent {
                    let name = path.file_name().map_or_else(|| path.display().to_string(), |n| n.to_string_lossy().into_owned());
                    let dir = path.parent().map(|d| d.display().to_string()).unwrap_or_default();
                    hline(ui, p);
                    let resp = ui
                        .add(egui::Button::new(RichText::new(name).color(p.fg)).frame(false).min_size(vec2(250.0, 22.0)))
                        .on_hover_text(path.display().to_string());
                    ui.label(RichText::new(dir).size(11.0).color(p.muted));
                    if resp.clicked() {
                        open = Some(path.clone());
                    }
                }
            });
        });
        if let Some(path) = open {
            self.open(path, ctx);
        } else if ctx.input(|i| i.pointer.any_click()) && !r.response.contains_pointer() {
            // a click outside closes the menu (the toggle button handles its own click)
            let on_toggle = ctx.input(|i| i.pointer.interact_pos()).is_some_and(|q| (q - at).length() < 40.0);
            if !on_toggle {
                self.show_recent = None;
            }
        }
    }

    fn side_panel(&mut self, ctx: &egui::Context, rs: &eframe::egui_wgpu::RenderState, p: &Pal) {
        let screen = ctx.content_rect();
        // stop above the bottom bar (its top from the previous frame)
        let bottom = if self.status_top > 0.0 { self.status_top } else { screen.bottom() - 52.0 };
        let height = (bottom - 8.0 - 60.0).max(200.0);
        egui::Area::new(egui::Id::new("side")).anchor(Align2::RIGHT_TOP, vec2(-12.0, 60.0)).movable(false).show(ctx, |ui| {
            p.group().inner_margin(egui::Margin::same(0)).show(ui, |ui| {
                ui.set_width(SIDE_W);
                ui.set_height(height);
                ui.vertical(|ui| {
                    egui::Frame::new().inner_margin(egui::Margin::same(4)).show(ui, |ui| {
                        ui.spacing_mut().item_spacing.x = 2.0;
                        ui.horizontal(|ui| {
                            let w = (SIDE_W - 8.0 - 4.0) / 3.0;
                            for (t, l) in [(Tab::Props, "Properties"), (Tab::Layers, "Layers"), (Tab::Look, "Appearance")] {
                                let pressed = self.tab == t;
                                let (r, resp) = ui.allocate_exact_size(vec2(w, 28.0), Sense::click());
                                let bg = if pressed { p.accent } else if resp.hovered() { p.panel2 } else { Color32::TRANSPARENT };
                                ui.painter().rect_filled(r, 6.0, bg);
                                ui.painter().text(r.center(), Align2::CENTER_CENTER, l, FontId::proportional(13.0), if pressed { Color32::WHITE } else { p.fg });
                                if resp.clicked() {
                                    self.tab = t;
                                }
                            }
                        });
                    });
                    let (r, _) = ui.allocate_exact_size(vec2(SIDE_W, 1.0), Sense::hover());
                    ui.painter().rect_filled(r, 0.0, p.line);
                    egui::Frame::new().inner_margin(egui::Margin::symmetric(12, 10)).show(ui, |ui| {
                        ui.set_width(SIDE_W - 24.0);
                        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| match self.tab {
                            Tab::Props => self.props_tab(ui, rs, p),
                            Tab::Layers => self.layers_tab(ui, rs, p),
                            Tab::Look => self.look_tab(ui, p),
                        });
                    });
                });
            });
        });
    }

    fn props_tab(&mut self, ui: &mut Ui, rs: &eframe::egui_wgpu::RenderState, p: &Pal) {
        let Some(d) = &mut self.doc else { return };
        let mut layer_action: Option<(usize, bool)> = None; // (layer, solo)
        match &d.selection {
            Some(Selection { info: Some(info), batch, inst, segs, .. }) => {
                let m = inst_xf(&d.scene, *inst);
                let c = d.scene.centers[*batch];
                let o = d.scene.origin;
                let s = (m[0] * m[3] - m[1] * m[2]).abs().sqrt();
                let rot = m[1].atan2(m[0]).to_degrees();
                ui.label(RichText::new(&info.type_name).font(strong_font(15.0)).color(p.fg));
                ui.add_space(4.0);
                let mut rows: Vec<(String, String)> = vec![
                    ("Layer".into(), info.layer.clone()),
                    ("Color".into(), info.color.clone()),
                    ("Handle".into(), format!("{:X}", info.handle)),
                ];
                for (name, v) in &info.values {
                    let txt = match v {
                        InfoValue::Pt(q) => {
                            let w = apply(&m, q[0] - c[0], q[1] - c[1]);
                            format!("{}\n{}", fmt(w[0] + o[0], 3), fmt(w[1] + o[1], 3))
                        }
                        InfoValue::Len(x) => fmt(x * s, 3),
                        InfoValue::Area(x) => fmt(x * s * s, 3),
                        InfoValue::Ang(x) => format!("{}°", fmt(((x + rot) % 360.0 + 360.0) % 360.0, 2)),
                        InfoValue::Num(x) => fmt(*x, if x.fract() == 0.0 { 0 } else { 3 }),
                        InfoValue::Text(t) => t.clone(),
                    };
                    rows.push((name.clone(), txt));
                }
                if !segs.is_empty() {
                    let drawn: f64 = segs.iter().map(|[a, b]| (b[0] - a[0]).hypot(b[1] - a[1])).sum();
                    rows.push(("Drawn length".into(), fmt(drawn, 3)));
                }
                kv_table(ui, p, &rows);
                if (s - 1.0).abs() > 1e-9 || d.scene.blocks[*batch].inst_count > 1 {
                    ui.add_space(6.0);
                    ui.label(
                        RichText::new(format!("Element inside a block: values converted to drawing coordinates (scale {}).", fmt(s, 4)))
                            .size(12.0)
                            .color(p.muted),
                    );
                }
                if let Some(li) = d.layers.iter().position(|l| l.name == info.layer) {
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        if ui.button("Hide layer").clicked() {
                            layer_action = Some((li, false));
                        }
                        if ui.button("Show only this layer").clicked() {
                            layer_action = Some((li, true));
                        }
                    });
                }
            }
            _ => {
                ui.label(RichText::new("Click an element of the drawing to see its type, layer and measurements.").color(p.muted));
            }
        }
        if let Some((li, solo)) = layer_action {
            for (i, l) in d.layers.iter_mut().enumerate() {
                if solo {
                    l.look.on = i == li;
                } else if i == li {
                    l.look.on = false;
                }
            }
            if !solo {
                d.selection = None;
            }
            d.upload_layers(&rs.queue);
        }
        if !d.measures.is_empty() || !d.areas.is_empty() {
            section(ui, p, "Measurements");
            let mut remove: Option<(bool, usize)> = None;
            let items: Vec<(bool, usize, String)> = d
                .measures
                .iter()
                .enumerate()
                .map(|(i, m)| (false, i, format!("Distance {}", fmt((m[1][0] - m[0][0]).hypot(m[1][1] - m[0][1]), 3))))
                .chain(d.areas.iter().enumerate().map(|(i, a)| (true, i, format!("Area {} · perimeter {}", fmt(poly_area(a), 3), fmt(poly_perimeter(a), 3)))))
                .collect();
            for (is_area, i, text) in items {
                hline(ui, p);
                ui.horizontal(|ui| {
                    ui.label(text);
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if tool_button(ui, p, None, Some("✕"), false, "Delete measurement").clicked() {
                            remove = Some((is_area, i));
                        }
                    });
                });
            }
            match remove {
                Some((false, i)) => {
                    d.measures.remove(i);
                }
                Some((true, i)) => {
                    d.areas.remove(i);
                }
                None => {}
            }
        }
    }

    fn layers_tab(&mut self, ui: &mut Ui, rs: &eframe::egui_wgpu::RenderState, p: &Pal) {
        let texts = self.settings.texts;
        let Some(d) = &mut self.doc else { return };
        ui.add(egui::TextEdit::singleline(&mut d.layer_filter).hint_text("Filter layers…").desired_width(f32::INFINITY).margin(vec2(8.0, 5.0)));
        let mut changed = false;
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            if ui.button("All").clicked() {
                d.layers.iter_mut().for_each(|l| l.look.on = true);
                changed = true;
            }
            if ui.button("None").clicked() {
                d.layers.iter_mut().for_each(|l| l.look.on = false);
                changed = true;
            }
            if ui.button("Invert").clicked() {
                d.layers.iter_mut().for_each(|l| l.look.on = !l.look.on);
                changed = true;
            }
            let on = d.layers.iter().filter(|l| l.look.on).count();
            ui.label(RichText::new(format!("{on} / {} visible", d.layers.len())).color(p.muted));
        });
        ui.add_space(4.0);
        let filter = d.layer_filter.to_lowercase();
        let mut solo = None;
        for (i, l) in d.layers.iter_mut().enumerate() {
            if !filter.is_empty() && !l.name.to_lowercase().contains(&filter) {
                continue;
            }
            hline(ui, p);
            ui.horizontal(|ui| {
                changed |= ui.checkbox(&mut l.look.on, "").on_hover_text("Visible").changed();
                let mut c = l.look.color.unwrap_or(l.look.base);
                if ui.color_edit_button_srgb(&mut c).on_hover_text("Layer color").changed() {
                    l.look.color = Some(c);
                    changed = true;
                }
                let name = RichText::new(&l.name).color(if l.look.on { p.fg } else { p.muted });
                ui.add(egui::Label::new(if l.look.on { name } else { name.strikethrough() }).truncate());
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if tool_button(ui, p, None, Some("solo"), false, "Show only this layer").clicked() {
                        solo = Some(i);
                    }
                    if l.look.color.is_some() && tool_button(ui, p, None, Some("↺"), false, "Original color").clicked() {
                        l.look.color = None;
                        changed = true;
                    }
                    if l.look.on != l.dwg_on {
                        ui.label(RichText::new("•").color(p.accent)).on_hover_text("Different from the DWG");
                    }
                });
            });
        }
        if let Some(s) = solo {
            d.layers.iter_mut().enumerate().for_each(|(i, l)| l.look.on = i == s);
            changed = true;
        }
        if changed {
            d.upload_layers(&rs.queue);
            d.run_search(texts);
        }
    }

    fn look_tab(&mut self, ui: &mut Ui, p: &Pal) {
        let before = self.settings.look();
        let texts_before = self.settings.texts;
        let s = &mut self.settings;
        section(ui, p, "Theme");
        egui::Grid::new("themes").num_columns(2).spacing(vec2(6.0, 6.0)).show(ui, |ui| {
            for (k, t) in Theme::ALL.into_iter().enumerate() {
                let w = (SIDE_W - 24.0 - 6.0) / 2.0;
                let (r, resp) = ui.allocate_exact_size(vec2(w, 58.0), Sense::click());
                let pressed = s.theme == t;
                ui.painter().rect_filled(r, 6.0, p.panel2);
                let stroke = if pressed { Stroke::new(2.0, p.accent) } else { Stroke::new(1.0, p.line) };
                ui.painter().rect_stroke(r, 6.0, stroke, egui::StrokeKind::Inside);
                let sw = Rect::from_min_size(r.min + vec2(8.0, 8.0), vec2(w - 16.0, 22.0));
                let bgc = t.bg();
                ui.painter().rect_filled(sw, 4.0, Color32::from_rgb(bgc[0], bgc[1], bgc[2]));
                match t.ink() {
                    Some(ink) => {
                        let bar = Rect::from_min_max(pos2(sw.min.x + 1.0, sw.max.y - 6.0), pos2(sw.max.x - 1.0, sw.max.y - 1.0));
                        ui.painter().rect_filled(bar, 0.0, Color32::from_rgb(ink[0], ink[1], ink[2]));
                    }
                    None => {
                        for (j, c) in [Color32::from_rgb(255, 80, 80), Color32::from_rgb(80, 208, 80), Color32::from_rgb(80, 144, 255)].into_iter().enumerate() {
                            let x0 = sw.min.x + sw.width() * (0.55 + 0.15 * j as f32);
                            ui.painter().rect_filled(Rect::from_min_max(pos2(x0, sw.min.y + 1.0), pos2(x0 + sw.width() * 0.15, sw.max.y - 1.0)), 0.0, c);
                        }
                    }
                }
                ui.painter().rect_stroke(sw, 4.0, Stroke::new(1.0, p.line), egui::StrokeKind::Inside);
                ui.painter().text(pos2(r.min.x + 8.0, r.max.y - 15.0), Align2::LEFT_CENTER, t.name(), FontId::proportional(13.0), p.fg);
                if resp.clicked() {
                    s.theme = t;
                    s.bg = t.bg();
                    if let Some(ink) = t.ink() {
                        s.mono = ink;
                    }
                }
                if k % 2 == 1 {
                    ui.end_row();
                }
            }
        });
        section(ui, p, "Colors");
        let field = |ui: &mut Ui, label: &str, c: &mut [u8; 3]| {
            ui.horizontal(|ui| {
                ui.label(label);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.color_edit_button_srgb(c);
                });
            });
        };
        field(ui, "Background", &mut s.bg);
        field(ui, "Lines (monochrome themes)", &mut s.mono);
        field(ui, "Highlight and measurements", &mut s.sel);
        ui.horizontal(|ui| {
            ui.label("Darken light colors on a light background");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| ui.checkbox(&mut s.adapt, ""));
        });
        section(ui, p, "Elements");
        ui.horizontal(|ui| {
            ui.label("Texts");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| ui.checkbox(&mut s.texts, ""));
        });
        ui.horizontal(|ui| {
            ui.label("Fills (solid hatches, solids)");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| ui.checkbox(&mut s.fills, ""));
        });
        ui.add_space(8.0);
        ui.label(RichText::new("Colors of single layers are changed in the Layers tab.").size(12.0).color(p.muted));
        ui.add_space(6.0);
        if ui.button("Restore default settings").clicked() {
            let recent = std::mem::take(&mut s.recent);
            *s = Settings { recent, ..Settings::default() };
        }
        let texts_now = s.texts;
        if self.settings.look() != before || texts_now != texts_before {
            self.settings.save();
            if let Some(d) = &mut self.doc {
                d.version += 1;
                if texts_now != texts_before {
                    d.run_search(texts_now);
                }
            }
        }
    }

    fn status(&mut self, ctx: &egui::Context, p: &Pal) {
        // Left: coordinates and snap. The hint goes to their right only when it fits, so nothing
        // overlaps. Redraw time and graphics card are in the help dialog (File and drawing).
        let left = egui::Area::new(egui::Id::new("status")).anchor(Align2::LEFT_BOTTOM, vec2(12.0, -12.0)).movable(false).show(ctx, |ui| {
            ui.horizontal(|ui| {
                if let Some(d) = &self.doc {
                    p.group().inner_margin(egui::Margin::symmetric(10, 6)).show(ui, |ui| {
                        let txt = match d.hover.map(|h| h.0) {
                            Some(w) => {
                                let o = d.scene.origin;
                                let dec = (d.view.ppu.log10().ceil() as i32 + 1).clamp(0, 4) as usize;
                                let px = ctx.pixels_per_point() as f64 / d.view.ppu;
                                format!("X {}   Y {}   ·   1 px = {}", fmt(w[0] + o[0], dec), fmt(w[1] + o[1], dec), fmt(px, if px < 1.0 { 3 } else { 1 }))
                            }
                            None => "X —   Y —".into(),
                        };
                        ui.label(RichText::new(txt).monospace().color(p.fg));
                    });
                }
                p.group().show(ui, |ui| {
                    if tool_button(ui, p, Some(Icon::Snap), Some("Snap"), self.settings.snap, "Snap to endpoints and midpoints (S)").clicked() {
                        self.settings.snap = !self.settings.snap;
                        self.settings.save();
                    }
                });
            });
        });
        self.status_top = left.response.rect.top();
        let hint = self.tool.hint();
        let galley = ctx.fonts_mut(|f| f.layout_no_wrap(hint.to_string(), FontId::proportional(13.0), p.muted));
        let x = left.response.rect.right() + 6.0;
        let fits = x + galley.size().x + 20.0 + 6.0 <= ctx.content_rect().right() - 12.0;
        if fits {
            egui::Area::new(egui::Id::new("status-hint"))
                .fixed_pos(pos2(x, left.response.rect.bottom() - (galley.size().y + 12.0)))
                .movable(false)
                .interactable(false)
                .show(ctx, |ui| {
                    p.group().inner_margin(egui::Margin::symmetric(10, 6)).show(ui, |ui| {
                        ui.label(RichText::new(hint).color(p.muted));
                    });
                });
        }
    }

    fn error_window(&mut self, ctx: &egui::Context, p: &Pal) {
        let screen = ctx.content_rect();
        egui::Area::new(egui::Id::new("error-dim")).fixed_pos(screen.min).order(egui::Order::Foreground).show(ctx, |ui| {
            ui.allocate_exact_size(screen.size(), Sense::click());
            ui.painter().rect_filled(screen, 0.0, Color32::from_black_alpha(110));
        });
        let msg = self.error.clone().unwrap_or_default();
        egui::Area::new(egui::Id::new("error")).anchor(Align2::CENTER_CENTER, vec2(0.0, 0.0)).order(egui::Order::Tooltip).show(ctx, |ui| {
            p.group().inner_margin(egui::Margin::symmetric(18, 16)).corner_radius(10).show(ui, |ui| {
                ui.set_width(460.0);
                ui.label(RichText::new("Could not open the file").font(strong_font(16.0)).color(p.fg));
                ui.add_space(6.0);
                ui.label(RichText::new(msg).color(p.muted));
                ui.add_space(10.0);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
                    if ui.button("Close").clicked() {
                        self.error = None;
                    }
                });
            });
        });
    }

    fn help_window(&mut self, ctx: &egui::Context, p: &Pal) {
        // dim the drawing behind the dialog; a click outside closes it
        let screen = ctx.content_rect();
        egui::Area::new(egui::Id::new("help-dim")).fixed_pos(screen.min).order(egui::Order::Foreground).show(ctx, |ui| {
            let (_, resp) = ui.allocate_exact_size(screen.size(), Sense::click());
            ui.painter().rect_filled(screen, 0.0, Color32::from_black_alpha(110));
            if resp.clicked() {
                self.show_help = false;
            }
        });
        egui::Area::new(egui::Id::new("help")).anchor(Align2::CENTER_CENTER, vec2(0.0, 0.0)).order(egui::Order::Tooltip).show(ctx, |ui| {
            p.group().inner_margin(egui::Margin::symmetric(18, 16)).corner_radius(10).show(ui, |ui| {
                ui.set_width(540.0);
                ui.horizontal(|ui| {
                    for (tab, title) in [(false, "Usage"), (true, "File and drawing")] {
                        let on = self.help_info == tab;
                        let t = RichText::new(title).font(strong_font(15.0)).color(if on { p.fg } else { p.muted });
                        if ui.selectable_label(on, t).clicked() {
                            self.help_info = tab;
                        }
                    }
                });
                ui.add_space(6.0);
                if !self.help_info {
                    let rows: Vec<(String, String)> = [
                        ("Open a file", "Ctrl+O · drop a file on the window · Recent"),
                        ("Pan", "drag · middle or right button"),
                        ("Zoom", "wheel · Shift+drag a rectangle · Z"),
                        ("Fit to window", "double click · F"),
                        ("Element info", "I, then click"),
                        ("Distance", "M, two clicks · Shift = orthogonal"),
                        ("Area", "A, click the vertices · Enter or double click closes · ⌫ removes the last"),
                        ("Snap to endpoints and midpoints", "S"),
                        ("Search text", "/ or Ctrl+F · Enter next · Shift+Enter previous"),
                        ("Side panel", "L"),
                        ("Cancel", "Esc · Del clears all measurements"),
                    ]
                    .iter()
                    .map(|(a, b)| (a.to_string(), b.to_string()))
                    .collect();
                    kv_table(ui, p, &rows);
                } else {
                    if let Some(d) = &self.doc {
                        section(ui, p, "File");
                        kv_table(ui, p, &d.file_info);
                    }
                    section(ui, p, "Rendering");
                    let mut rows = self.doc.as_ref().map_or_else(Vec::new, |d| d.render_info.clone());
                    if let Some(d) = &self.doc {
                        rows.push(("Last sharp redraw".into(), format!("{:.0} ms", d.last_sharp.as_secs_f64() * 1000.0)));
                    }
                    rows.push(("Graphics".into(), self.adapter.clone()));
                    rows.push(("Antialiasing".into(), format!("MSAA {}x", self.samples)));
                    kv_table(ui, p, &rows);
                }
                ui.add_space(10.0);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
                    if ui.button("Close").clicked() {
                        self.show_help = false;
                    }
                });
            });
        });
    }

    fn canvas(&mut self, ui: &mut Ui, ctx: &egui::Context, rs: &eframe::egui_wgpu::RenderState, p: &Pal) {
        let rect = ui.max_rect();
        let resp = ui.allocate_rect(rect, Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        let hovering_files = ctx.input(|i| !i.raw.hovered_files.is_empty());
        let busy = self.loading.as_ref().map(|(_, ph, path)| (*ph, path.clone()));

        let Some(d) = &mut self.doc else {
            if busy.is_none() {
                let (title, body) = match &self.error {
                    Some(e) => ("Could not open the file", e.clone()),
                    None => ("CadGlance", "Open a DWG file (Ctrl+O) or drop it here.".to_string()),
                };
                painter.text(rect.center() - vec2(0.0, 14.0), Align2::CENTER_CENTER, title, strong_font(20.0), p.fg);
                painter.text(rect.center() + vec2(0.0, 14.0), Align2::CENTER_CENTER, body, FontId::proportional(14.0), p.muted);
            }
            self.overlay_busy(ui, rect, busy, hovering_files, p);
            return;
        };
        d.canvas = rect;
        let ppp = ctx.pixels_per_point() as f64;
        let (pw, ph) = (rect.width() as f64 * ppp, rect.height() as f64 * ppp);

        // cache target: twice the window, recreated on resize
        let max = rs.device.limits().max_texture_dimension_2d;
        let want = (((pw as f32) * CACHE_MARGIN) as u32).clamp(1, max);
        let want_h = (((ph as f32) * CACHE_MARGIN) as u32).clamp(1, max);
        if d.target.as_ref().is_none_or(|t| t.width != want || t.height != want_h) {
            let t = d.gpu.target(&rs.device, want, want_h);
            let mut r = rs.renderer.write();
            match d.tex {
                Some(id) => r.update_egui_texture_from_wgpu_texture(&rs.device, &t.resolve_view, wgpu::FilterMode::Linear, id),
                None => d.tex = Some(r.register_native_texture(&rs.device, &t.resolve_view, wgpu::FilterMode::Linear)),
            }
            d.target = Some(t);
            d.cache = None;
        }
        if d.need_fit {
            d.view = fit_view(d.scene.bbox, pw, ph);
            d.need_fit = false;
            d.last_change = Instant::now();
        }

        // ---- input ----
        let shift = ui.input(|i| i.modifiers.shift);
        let rect_drag = (self.tool == Tool::Zoom || (shift && self.tool == Tool::Info)) && resp.dragged_by(egui::PointerButton::Primary);
        if rect_drag {
            if let Some(b) = resp.interact_pointer_pos() {
                let start = self.zoom_rect.map_or(b - resp.drag_delta(), |r| r.0);
                self.zoom_rect = Some((start, b));
            }
        } else if resp.dragged() {
            let dd = resp.drag_delta();
            d.view.cx -= dd.x as f64 * ppp / d.view.ppu;
            d.view.cy += dd.y as f64 * ppp / d.view.ppu;
            d.last_change = Instant::now();
        }
        if resp.drag_stopped() {
            if let Some((a, b)) = self.zoom_rect.take() {
                let r = Rect::from_two_pos(a, b);
                if r.width() > 6.0 && r.height() > 6.0 {
                    let xf = Xform { rect, view: d.view, ppp };
                    let (p0, p1) = (xf.to_world(r.left_bottom()), xf.to_world(r.right_top()));
                    d.view = View {
                        cx: (p0[0] + p1[0]) / 2.0,
                        cy: (p0[1] + p1[1]) / 2.0,
                        ppu: (pw / (p1[0] - p0[0]).max(1e-12)).min(ph / (p1[1] - p0[1]).max(1e-12)),
                    };
                    d.last_change = Instant::now();
                }
            }
        }
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if let (true, Some(hp)) = (resp.hovered() && scroll != 0.0, resp.hover_pos()) {
            let ax = (hp.x - rect.center().x) as f64 * ppp;
            let ay = (rect.center().y - hp.y) as f64 * ppp;
            let (wx, wy) = (d.view.cx + ax / d.view.ppu, d.view.cy + ay / d.view.ppu);
            d.view.ppu *= (scroll as f64 * 0.004).exp();
            d.view.cx = wx - ax / d.view.ppu;
            d.view.cy = wy - ay / d.view.ppu;
            d.last_change = Instant::now();
        }

        let xf = Xform { rect, view: d.view, ppp };
        let tol = PICK_PX as f64 * xf.unit();
        let layers_on: Vec<bool> = d.layers.iter().map(|l| l.look.on).collect();
        let visible = move |l: u16| l == cadglance::geom::NO_LAYER || layers_on.get(l as usize).copied().unwrap_or(true);

        // hover: coordinates, and snap preview for the measuring tools (none in a PNG export)
        let measuring = matches!(self.tool, Tool::Measure | Tool::Area);
        let pointer = if self.capturing { None } else { resp.hover_pos() };
        d.hover = pointer.map(|hp| {
            let w = xf.to_world(hp);
            if measuring && self.settings.snap && !resp.dragged() {
                if let (_, Some(s)) = d.picker.pick(&d.scene, w, tol, &visible, false) {
                    return (s.p, Some(s.kind));
                }
            }
            if measuring && shift {
                if let Some(q) = d.pending.last().copied() {
                    return (if (w[0] - q[0]).abs() > (w[1] - q[1]).abs() { [w[0], q[1]] } else { [q[0], w[1]] }, None);
                }
            }
            (w, None)
        });

        if resp.double_clicked() {
            match self.tool {
                Tool::Area => {
                    // the double click also added its point once more
                    d.pending.pop();
                    if d.pending.len() > 2 {
                        d.areas.push(std::mem::take(&mut d.pending));
                    }
                }
                Tool::Measure => {}
                _ => d.need_fit = true,
            }
        } else if resp.clicked_by(egui::PointerButton::Primary) {
            self.show_recent = None;
            if let Some(cp) = resp.interact_pointer_pos() {
                let w = xf.to_world(cp);
                match self.tool {
                    Tool::Info => {
                        let texts = self.settings.texts;
                        let hit_text = d.gpu.text_boxes.iter().find(|b| d.text_visible(b, texts) && point_in_quad(&b.corners, w[0], w[1])).cloned();
                        d.selection = if let Some(b) = hit_text {
                            let batch = d.scene.texts[b.text].batch;
                            Some(Selection { segs: vec![], quad: Some(b.corners), info: entity_details(&d.drawing, b.id as u64), batch, inst: b.inst })
                        } else if let (Some(h), _) = d.picker.pick(&d.scene, w, tol, &visible, true) {
                            (h.id != 0).then(|| Selection {
                                segs: entity_segments(&d.scene, h.batch, h.inst, h.id),
                                quad: None,
                                info: entity_details(&d.drawing, h.id as u64),
                                batch: h.batch,
                                inst: h.inst,
                            })
                        } else {
                            None
                        };
                        if d.selection.is_some() {
                            self.tab = Tab::Props;
                        }
                    }
                    Tool::Measure => {
                        let q = d.hover.map_or(w, |h| h.0);
                        if d.pending.is_empty() {
                            d.pending.push(q);
                        } else {
                            d.measures.push([d.pending[0], q]);
                            d.pending.clear();
                        }
                    }
                    Tool::Area => {
                        let q = d.hover.map_or(w, |h| h.0);
                        if d.pending.len() > 2 && (xf.to_screen(d.pending[0]) - xf.to_screen(q)).length() < PICK_PX {
                            d.areas.push(std::mem::take(&mut d.pending));
                        } else {
                            d.pending.push(q);
                        }
                    }
                    Tool::Zoom => {}
                }
            }
        }

        // ---- sharp redraw once the view has been still for SHARP_DELAY ----
        let look = self.settings.look();
        if d.cache != Some((d.view, d.version)) {
            let idle = d.last_change.elapsed();
            if idle >= SHARP_DELAY || d.cache.is_none() {
                if let Some(t) = &d.target {
                    let t0 = Instant::now();
                    let idx = d.gpu.render(&rs.device, &rs.queue, t, d.view, &look, d.scene.layer0);
                    let _ = rs.device.poll(wgpu::PollType::Wait { submission_index: Some(idx), timeout: None });
                    d.last_sharp = t0.elapsed();
                    d.cache = Some((d.view, d.version));
                }
            } else {
                ctx.request_repaint_after(SHARP_DELAY - idle);
            }
        }

        // ---- present: cached image where its world rectangle falls on screen ----
        if let (Some((cv, _)), Some(tex), Some(t)) = (d.cache, d.tex, &d.target) {
            let (hw, hh) = (t.width as f64 / 2.0 / cv.ppu, t.height as f64 / 2.0 / cv.ppu);
            let r = Rect::from_min_max(xf.to_screen([cv.cx - hw, cv.cy + hh]), xf.to_screen([cv.cx + hw, cv.cy - hh]));
            painter.image(tex, r, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
        }

        // ---- overlay (same drawing rules as the web viewer) ----
        let sel = self.settings.sel_color();
        let (plate, ink) = if self.settings.bg_light() {
            (Color32::from_white_alpha(235), Color32::from_rgb(0x1a, 0x1f, 0x24))
        } else {
            (Color32::from_rgba_unmultiplied(17, 20, 24, 230), Color32::from_rgb(0xe8, 0xeb, 0xef))
        };
        let label = |at: Pos2, s: String| {
            let galley = painter.layout_no_wrap(s, FontId::proportional(12.5), ink);
            let r = Rect::from_center_size(at, galley.size() + vec2(12.0, 6.0));
            painter.rect_filled(r, 4.0, plate);
            painter.galley(r.min + vec2(6.0, 3.0), galley, ink);
        };
        let dot = |w: P2| painter.circle_filled(xf.to_screen(w), 3.5, sel);
        let quad = |q: &[P2; 4], stroke: Stroke| {
            painter.add(egui::Shape::closed_line(q.iter().map(|w| xf.to_screen(*w)).collect(), stroke));
        };
        for (k, &mi) in d.matches.iter().enumerate() {
            let b = &d.gpu.text_boxes[mi];
            let cur = d.match_index == Some(k);
            quad(&b.corners, Stroke::new(if cur { 2.5 } else { 1.2 }, Color32::from_rgba_unmultiplied(255, 213, 74, if cur { 255 } else { 130 })));
        }
        if let Some(s) = &d.selection {
            for [a, b] in &s.segs {
                painter.line_segment([xf.to_screen(*a), xf.to_screen(*b)], Stroke::new(3.0, sel));
            }
            if let Some(q) = &s.quad {
                quad(q, Stroke::new(2.5, sel));
            }
        }
        let area_shape = |pts: &[P2], done: bool| {
            let sp: Vec<Pos2> = pts.iter().map(|q| xf.to_screen(*q)).collect();
            if sp.len() > 2 {
                painter.add(egui::Shape::convex_polygon(sp.clone(), sel.gamma_multiply(0.18), Stroke::NONE));
            }
            let stroke = Stroke::new(2.0, sel);
            if done {
                painter.add(egui::Shape::closed_line(sp.clone(), stroke));
            } else {
                painter.add(egui::Shape::dashed_line(&[sp.clone(), vec![sp[0]]].concat(), stroke, 6.0, 4.0));
            }
            for q in pts {
                dot(*q);
            }
            if pts.len() > 2 {
                let c = pts.iter().fold([0.0, 0.0], |a, q| [a[0] + q[0], a[1] + q[1]]);
                let c = [c[0] / pts.len() as f64, c[1] / pts.len() as f64];
                label(xf.to_screen(c), format!("A {}   P {}", fmt(poly_area(pts), 3), fmt(poly_perimeter(pts), 3)));
            }
        };
        for a in &d.areas {
            area_shape(a, true);
        }
        if self.tool == Tool::Area && !d.pending.is_empty() {
            let mut pts = d.pending.clone();
            if let Some(h) = d.hover {
                pts.push(h.0);
            }
            area_shape(&pts, false);
        }
        let measure_shape = |a: P2, b: P2, done: bool| {
            let (sa, sb) = (xf.to_screen(a), xf.to_screen(b));
            let stroke = Stroke::new(2.0, sel);
            if done {
                painter.line_segment([sa, sb], stroke);
            } else {
                painter.add(egui::Shape::dashed_line(&[sa, sb], stroke, 6.0, 4.0));
            }
            dot(a);
            dot(b);
            let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
            label(
                pos2((sa.x + sb.x) / 2.0, (sa.y + sb.y) / 2.0 - 14.0),
                format!("{}   ΔX {}   ΔY {}   {}°", fmt(dx.hypot(dy), 3), fmt(dx, 3), fmt(dy, 3), fmt(dy.atan2(dx).to_degrees(), 2)),
            );
        };
        for m in &d.measures {
            measure_shape(m[0], m[1], true);
        }
        if self.tool == Tool::Measure {
            if let (Some(&a), Some(h)) = (d.pending.first(), d.hover) {
                measure_shape(a, h.0, false);
            } else if let Some(&a) = d.pending.first() {
                dot(a);
            }
        }
        if let (true, Some((q, Some(kind)))) = (measuring, d.hover) {
            let s = xf.to_screen(q);
            let k = 6.0;
            let stroke = Stroke::new(2.0, sel);
            match kind {
                SnapKind::End => {
                    painter.rect_stroke(Rect::from_center_size(s, vec2(2.0 * k, 2.0 * k)), 0.0, stroke, egui::StrokeKind::Middle);
                }
                SnapKind::Mid => {
                    painter.add(egui::Shape::closed_line(vec![s + vec2(0.0, -k), s + vec2(k, k), s + vec2(-k, k)], stroke));
                }
            }
        }
        if let Some((a, b)) = self.zoom_rect {
            painter.rect(Rect::from_two_pos(a, b), 0.0, sel.gamma_multiply(0.1), Stroke::new(1.5, sel), egui::StrokeKind::Middle);
        }
        if measuring || self.tool == Tool::Zoom {
            ctx.set_cursor_icon(egui::CursorIcon::Crosshair);
        } else if resp.dragged() {
            ctx.set_cursor_icon(egui::CursorIcon::Grabbing);
        }

        self.overlay_busy(ui, rect, busy, hovering_files, p);
    }

    /// Loading card with a spinner, and a drop hint while files are dragged over the window.
    fn overlay_busy(&self, ui: &mut Ui, rect: Rect, busy: Option<(&'static str, PathBuf)>, hovering: bool, p: &Pal) {
        let painter = ui.painter_at(rect);
        if hovering {
            let r = rect.shrink(16.0);
            painter.rect_filled(r, 12.0, p.accent.gamma_multiply(0.12));
            painter.rect_stroke(r, 12.0, Stroke::new(2.0, p.accent), egui::StrokeKind::Inside);
            painter.text(rect.center(), Align2::CENTER_CENTER, "Drop the DWG file to open it", strong_font(18.0), p.fg);
        }
        if let Some((phase, path)) = busy {
            let r = Rect::from_center_size(rect.center(), vec2(380.0, 128.0));
            painter.add(p.shadow().as_shape(r, CornerRadius::same(10)));
            painter.rect_filled(r, 10.0, p.panel);
            painter.rect_stroke(r, 10.0, Stroke::new(1.0, p.line), egui::StrokeKind::Inside);
            ui.put(Rect::from_center_size(r.center() - vec2(0.0, 26.0), vec2(30.0, 30.0)), egui::Spinner::new().size(26.0).color(p.accent));
            let name = path.file_name().map_or_else(String::new, |n| n.to_string_lossy().into_owned());
            painter.text(r.center() + vec2(0.0, 12.0), Align2::CENTER_CENTER, phase, strong_font(14.0), p.fg);
            painter.text(r.center() + vec2(0.0, 34.0), Align2::CENTER_CENTER, name, FontId::proportional(12.0), p.muted);
            ui.ctx().request_repaint_after(Duration::from_millis(30));
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let software = args.iter().any(|a| a == "--software");
    let path = args.iter().find(|a| !a.starts_with("--")).map(PathBuf::from);
    if args.iter().any(|a| a == "--bench") {
        let Some(p) = path else {
            eprintln!("usage: cadglance --bench [--software] <file.dwg>");
            std::process::exit(2);
        };
        if let Err(e) = bench(&p, software) {
            eprintln!("{e}");
            std::process::exit(1);
        }
        return;
    }

    let setup = WgpuSetupCreateNew {
        native_adapter_selector: software.then(|| {
            Arc::new(|adapters: &[wgpu::Adapter], _: Option<&wgpu::Surface<'_>>| {
                adapters
                    .iter()
                    .find(|a| a.get_info().device_type == wgpu::DeviceType::Cpu)
                    .cloned()
                    .ok_or_else(|| "no software (CPU) adapter available".to_string())
            }) as eframe::egui_wgpu::NativeAdapterSelectorMethod
        }),
        device_descriptor: Arc::new(|adapter| wgpu::DeviceDescriptor {
            label: Some("cadglance"),
            required_limits: limits_for(adapter),
            ..Default::default()
        }),
        ..Default::default()
    };
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1400.0, 900.0]).with_title("CadGlance").with_drag_and_drop(true),
        wgpu_options: WgpuConfiguration { wgpu_setup: WgpuSetup::CreateNew(setup), ..Default::default() },
        ..Default::default()
    };
    let result = eframe::run_native(
        "CadGlance",
        options,
        Box::new(move |cc| {
            setup_fonts(&cc.egui_ctx);
            let (adapter, samples) = cc.wgpu_render_state.as_ref().map_or((String::new(), 4), |rs| {
                let i = rs.adapter.get_info();
                let kind = match i.device_type {
                    wgpu::DeviceType::DiscreteGpu => "dedicated GPU",
                    wgpu::DeviceType::IntegratedGpu => "integrated GPU",
                    wgpu::DeviceType::Cpu => "no GPU (software)",
                    _ => "GPU",
                };
                (format!("{} · {kind}", i.name), if i.device_type == wgpu::DeviceType::Cpu { 1 } else { 4 })
            });
            let mut app = App {
                settings: Settings::load(),
                doc: None,
                loading: None,
                upload: None,
                error: None,
                tool: Tool::Info,
                tab: Tab::Props,
                show_help: false,
                help_info: false,
                show_recent: None,
                adapter,
                samples,
                font: None,
                zoom_rect: None,
                screenshot_rect: None,
                capturing: false,
                focus_search: false,
                styled_light: None,
                status_top: 0.0,
            };
            if let Some(p) = path {
                app.open(p, &cc.egui_ctx);
            }
            Ok(Box::new(app))
        }),
    );
    if let Err(e) = result {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
