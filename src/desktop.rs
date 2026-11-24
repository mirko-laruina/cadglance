//! GPU renderer for the desktop viewer (wgpu): the scene is drawn into an offscreen texture
//! ("cache", larger than the window) that the UI shows moved/scaled while panning and zooming.
//! Same data and rules as the web viewer: instanced block geometry, even-odd stencil fills,
//! layer table (visibility, color override), themes, text from a glyph atlas.

use std::collections::HashMap;

use ab_glyph::{Font, FontArc, PxScale, ScaleFont};
use eframe::wgpu;
use wgpu::util::DeviceExt;

use crate::entities::P2;
use crate::geom::{BlockDraw, Scene};
use crate::pick::{apply, eff_layer, inst_xf};
use crate::textlayout::{self, Metrics};

pub const CACHE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth24PlusStencil8;

/// View: center (world units relative to the scene origin) and device pixels per world unit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct View {
    pub cx: f64,
    pub cy: f64,
    pub ppu: f64,
}

/// Drawing options that change the rendered image.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Look {
    pub bg: [u8; 3],
    /// Monochrome: every line in this color.
    pub mono: Option<[u8; 3]>,
    /// Light background: white becomes dark ink, light colors darker.
    pub adapt: bool,
    pub texts: bool,
    pub fills: bool,
}

/// Per-layer state uploaded to the GPU.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayerLook {
    pub on: bool,
    pub color: Option<[u8; 3]>,
    pub base: [u8; 3],
}

const COMMON: &str = r#"
struct U {
  center: vec2<f32>, scale: vec2<f32>,
  mono_color: vec4<f32>,
  ink: vec4<f32>,
  flags: vec4<u32>,      // layer0, mono, adapt, -
};
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var layers: texture_2d<f32>;  // row 0: (visible, override, -, -), row 1: override rgb

fn layer_visible(l: u32) -> bool {
  return l == 65535u || textureLoad(layers, vec2<i32>(i32(l), 0), 0).r > 0.5;
}
fn themed(c0: vec3<f32>, l: u32) -> vec3<f32> {
  var c = c0;
  if (l != 65535u && textureLoad(layers, vec2<i32>(i32(l), 0), 0).g > 0.5) {
    c = textureLoad(layers, vec2<i32>(i32(l), 1), 0).rgb;
  }
  if (u.flags.y != 0u) { return u.mono_color.rgb; }
  if (u.flags.z != 0u) {
    if (min(c.r, min(c.g, c.b)) > 0.85) { return u.ink.rgb; }
    if (dot(c, vec3<f32>(0.2126, 0.7152, 0.0722)) > 0.7) { return c * 0.6; }
  }
  return c;
}
"#;

const SCENE_SHADER: &str = r#"
struct VIn {
  @location(0) pos: vec2<f32>,
  @location(1) col: vec4<f32>,
  @location(7) lay: u32,
  @location(2) ix: vec2<f32>,
  @location(3) iy: vec2<f32>,
  @location(4) it: vec2<f32>,
  @location(5) icb: vec4<f32>,
  @location(6) icl: vec4<f32>,
  @location(8) ilay: vec2<u32>,   // layer for layer-0 entities inside, top-level INSERT layer
};
struct VOut { @builtin(position) p: vec4<f32>, @location(0) c: vec4<f32> };

@vertex fn vs(v: VIn) -> VOut {
  var o: VOut;
  var eff = v.lay;
  if (v.lay == u.flags.x && v.ilay.x != 65535u) { eff = v.ilay.x; }
  if (!layer_visible(eff)) {
    o.p = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    o.c = vec4<f32>(0.0);
    return o;
  }
  let w = v.ix * v.pos.x + v.iy * v.pos.y + v.it;
  o.p = vec4<f32>((w - u.center) * u.scale, 0.0, 1.0);
  // vertex alpha = color mode: 255 fixed, 0 instance ByBlock color, 1 instance layer-0 color
  let m = v.col.a * 255.0;
  var c = v.col.rgb;
  if (m < 0.5) { c = v.icb.rgb; } else if (m < 2.0) { c = v.icl.rgb; }
  o.c = vec4<f32>(themed(c, eff), 1.0);
  return o;
}

@fragment fn fs(i: VOut) -> @location(0) vec4<f32> { return i.c; }
"#;

const TEXT_SHADER: &str = r#"
@group(1) @binding(0) var atlas: texture_2d<f32>;
@group(1) @binding(1) var samp: sampler;

struct VIn {
  @location(0) pos: vec2<f32>,
  @location(1) uv: vec2<f32>,
  @location(2) col: vec4<f32>,
  @location(3) lays: vec2<u32>,   // effective layer, top-level INSERT layer
};
struct VOut { @builtin(position) p: vec4<f32>, @location(0) uv: vec2<f32>, @location(1) c: vec4<f32> };

@vertex fn vs(v: VIn) -> VOut {
  var o: VOut;
  if (!layer_visible(v.lays.x)) {
    o.p = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    return o;
  }
  o.p = vec4<f32>((v.pos - u.center) * u.scale, 0.0, 1.0);
  o.uv = v.uv;
  o.c = vec4<f32>(themed(v.col.rgb, v.lays.x), 1.0);
  return o;
}

@fragment fn fs(i: VOut) -> @location(0) vec4<f32> {
  return vec4<f32>(i.c.rgb, textureSample(atlas, samp, i.uv).a);
}
"#;

pub struct Target {
    pub width: u32,
    pub height: u32,
    ms_view: Option<wgpu::TextureView>,
    depth_view: wgpu::TextureView,
    _resolve: wgpu::Texture,
    /// Single-sample result, shown by the UI.
    pub resolve_view: wgpu::TextureView,
}

/// A text placed in the world: for search and picking.
#[derive(Debug, Clone)]
pub struct TextBox {
    /// Index into `Scene::texts`.
    pub text: usize,
    pub inst: usize,
    pub layer: u16,
    pub top: u16,
    pub id: u32,
    /// World corners relative to the origin.
    pub corners: [P2; 4],
}

pub struct GpuScene {
    pub samples: u32,
    blocks: Vec<BlockDraw>,
    layer_count: u32,
    pos: wgpu::Buffer,
    col: wgpu::Buffer,
    lay: wgpu::Buffer,
    fill_pos: wgpu::Buffer,
    fill_col: wgpu::Buffer,
    fill_lay: wgpu::Buffer,
    inst: wgpu::Buffer,
    inst_col: wgpu::Buffer,
    inst_lay: wgpu::Buffer,
    text_vb: wgpu::Buffer,
    text_vertices: u32,
    uniforms: wgpu::Buffer,
    layers_tex: wgpu::Texture,
    bind: wgpu::BindGroup,
    atlas_bind: wgpu::BindGroup,
    lines: wgpu::RenderPipeline,
    fill_stencil: wgpu::RenderPipeline,
    fill_color: wgpu::RenderPipeline,
    text: wgpu::RenderPipeline,
    pub text_boxes: Vec<TextBox>,
}

fn buffer(device: &wgpu::Device, label: &str, bytes: &[u8]) -> wgpu::Buffer {
    // keep at least 16 bytes: some backends reject zero-sized vertex buffers
    let mut v = bytes.to_vec();
    if v.len() < 16 {
        v.resize(16, 0);
    }
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents: &v, usage: wgpu::BufferUsages::VERTEX })
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn u16_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Per-vertex layers widened to u32: vertex strides must be multiples of 4 bytes.
fn u16_as_u32_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| (*x as u32).to_le_bytes()).collect()
}

fn stencil(compare: wgpu::CompareFunction, pass: wgpu::StencilOperation, write: u32) -> wgpu::DepthStencilState {
    let face = wgpu::StencilFaceState {
        compare,
        fail_op: wgpu::StencilOperation::Keep,
        depth_fail_op: wgpu::StencilOperation::Keep,
        pass_op: pass,
    };
    wgpu::DepthStencilState {
        format: DEPTH_FORMAT,
        depth_write_enabled: false,
        depth_compare: wgpu::CompareFunction::Always,
        stencil: wgpu::StencilState { front: face, back: face, read_mask: 1, write_mask: write },
        bias: wgpu::DepthBiasState::default(),
    }
}

// ---------- text atlas ----------

const FONT_PX: f32 = 96.0;
const CELL: u32 = 128;
const BASE: f32 = 100.0;
const PADX: f32 = 16.0;

/// The viewer font: a common sans-serif TTF from the system, else egui's built-in font.
pub fn load_font() -> (FontArc, String) {
    let candidates = [
        "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ];
    for p in candidates {
        if let Some(f) = std::fs::read(p).ok().and_then(|b| FontArc::try_from_vec(b).ok()) {
            return (f, p.rsplit(['/', '\\']).next().unwrap_or(p).to_string());
        }
    }
    let defs = eframe::egui::FontDefinitions::default();
    let (name, data) = defs.font_data.iter().next().expect("egui has built-in fonts");
    (FontArc::try_from_vec(data.font.to_vec()).expect("egui font is valid"), format!("{name} (egui)"))
}

struct Atlas {
    font: FontArc,
    cap: f64,
    /// char -> (uv0, uv1, advance px)
    glyphs: HashMap<char, ([f32; 2], [f32; 2], f64)>,
    width: u32,
    height: u32,
    /// RGBA levels, largest first.
    levels: Vec<Vec<u8>>,
}

impl Metrics for Atlas {
    fn advance(&self, ch: char) -> Option<f64> {
        self.glyphs.get(&ch).map(|g| g.2).or_else(|| {
            let id = self.font.glyph_id(ch);
            (id.0 != 0 || ch == ' ').then(|| self.font.as_scaled(PxScale::from(FONT_PX)).h_advance(id) as f64)
        })
    }
    fn fallback_advance(&self) -> f64 {
        FONT_PX as f64 * 0.5
    }
    fn cap(&self) -> f64 {
        self.cap
    }
}

impl Atlas {
    fn build(font: FontArc, chars: &[char], max_tex: u32) -> Atlas {
        let scaled = font.as_scaled(PxScale::from(FONT_PX));
        let cap = font
            .outline_glyph(font.glyph_id('H').with_scale(PxScale::from(FONT_PX)))
            .map(|o| (o.px_bounds().max.y - o.px_bounds().min.y) as f64)
            .filter(|c| *c > 0.0)
            .unwrap_or(FONT_PX as f64 * 0.716);
        let drawable: Vec<char> = chars.iter().copied().filter(|c| font.glyph_id(*c).0 != 0).collect();
        let cols = (max_tex / CELL).clamp(1, 32);
        let rows = (drawable.len() as u32).div_ceil(cols).max(1);
        let (width, height) = (cols * CELL, rows * CELL);
        let mut px = vec![0u8; (width * height * 4) as usize];
        for p in px.chunks_exact_mut(4) {
            p[..3].copy_from_slice(&[255, 255, 255]);
        }
        let mut glyphs = HashMap::new();
        for (i, &ch) in drawable.iter().enumerate() {
            let (x, y) = ((i as u32 % cols) * CELL, (i as u32 / cols) * CELL);
            let id = font.glyph_id(ch);
            let g = id.with_scale_and_position(PxScale::from(FONT_PX), ab_glyph::point(x as f32 + PADX, y as f32 + BASE));
            if let Some(o) = font.outline_glyph(g) {
                let b = o.px_bounds();
                o.draw(|gx, gy, cov| {
                    let (px_x, px_y) = (b.min.x as i64 + gx as i64, b.min.y as i64 + gy as i64);
                    // stay inside this glyph's cell
                    if px_x >= x as i64 && px_x < (x + CELL) as i64 && px_y >= y as i64 && px_y < (y + CELL) as i64 {
                        let o = ((px_y as u32 * width + px_x as u32) * 4 + 3) as usize;
                        px[o] = px[o].max((cov.clamp(0.0, 1.0) * 255.0) as u8);
                    }
                });
            }
            let (w, h) = (width as f32, height as f32);
            glyphs.insert(
                ch,
                ([x as f32 / w, y as f32 / h], [(x + CELL) as f32 / w, (y + CELL) as f32 / h], scaled.h_advance(id) as f64),
            );
        }
        // mipmaps: 2x2 box filter on alpha
        let mut levels = vec![px];
        let (mut lw, mut lh) = (width, height);
        while lw > 1 || lh > 1 {
            let (nw, nh) = ((lw / 2).max(1), (lh / 2).max(1));
            let prev = levels.last().expect("level 0 exists");
            let mut next = vec![255u8; (nw * nh * 4) as usize];
            for y in 0..nh {
                for x in 0..nw {
                    let mut s = 0u32;
                    for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                        let (sx, sy) = ((x * 2 + dx).min(lw - 1), (y * 2 + dy).min(lh - 1));
                        s += prev[((sy * lw + sx) * 4 + 3) as usize] as u32;
                    }
                    next[((y * nw + x) * 4 + 3) as usize] = (s / 4) as u8;
                }
            }
            levels.push(next);
            (lw, lh) = (nw, nh);
        }
        Atlas { font, cap, glyphs, width, height, levels }
    }
}

/// Text vertices (24 bytes each: pos f32x2, uv f32x2, rgba u8x4, layers u16x2) and boxes.
fn text_mesh(scene: &Scene, atlas: &Atlas) -> (Vec<u8>, u32, Vec<TextBox>) {
    let ratios = textlayout::style_ratios(&scene.texts, atlas);
    let mut out = Vec::new();
    let mut count = 0u32;
    let mut boxes = Vec::new();
    for (ti, t) in scene.texts.iter().enumerate() {
        let scale = textlayout::text_scale(t, &ratios, atlas);
        let l = textlayout::layout(t, &scene.styles, atlas, scale);
        let b = &scene.blocks[t.batch];
        for i in b.inst_start..b.inst_start + b.inst_count {
            let m = inst_xf(scene, i);
            let c = if t.color[3] == 255 {
                t.color
            } else {
                let o = i * 8 + if t.color[3] == 0 { 0 } else { 4 };
                let ic = &scene.inst_colors;
                [ic[o], ic[o + 1], ic[o + 2], 255]
            };
            let (eff, top) = (eff_layer(scene, t.layer, i), scene.inst_layers[i * 2 + 1]);
            let w = |u: f64, v: f64| {
                let p = l.local(u, v);
                apply(&m, p[0], p[1])
            };
            for g in &l.glyphs {
                let Some(&(uv0, uv1, _)) = atlas.glyphs.get(&g.ch) else { continue };
                let (u0, u1) = (g.u - PADX as f64 * g.k * g.wf, g.u + (CELL as f64 - PADX as f64) * g.k * g.wf);
                let (v0, v1) = (g.v - (CELL as f64 - BASE as f64) * g.k, g.v + BASE as f64 * g.k);
                let (p00, p10, p01, p11) = (w(u0, v0), w(u1, v0), w(u0, v1), w(u1, v1));
                for (p, uv) in [
                    (p00, [uv0[0], uv1[1]]),
                    (p10, [uv1[0], uv1[1]]),
                    (p11, [uv1[0], uv0[1]]),
                    (p00, [uv0[0], uv1[1]]),
                    (p11, [uv1[0], uv0[1]]),
                    (p01, [uv0[0], uv0[1]]),
                ] {
                    out.extend_from_slice(&(p[0] as f32).to_le_bytes());
                    out.extend_from_slice(&(p[1] as f32).to_le_bytes());
                    out.extend_from_slice(&uv[0].to_le_bytes());
                    out.extend_from_slice(&uv[1].to_le_bytes());
                    out.extend_from_slice(&[c[0], c[1], c[2], 255]);
                    out.extend_from_slice(&eff.to_le_bytes());
                    out.extend_from_slice(&top.to_le_bytes());
                    count += 1;
                }
            }
            let [b0, b1, b2, b3] = l.bx;
            boxes.push(TextBox { text: ti, inst: i, layer: eff, top, id: t.id, corners: [w(b0, b1), w(b2, b1), w(b2, b3), w(b0, b3)] });
        }
    }
    (out, count, boxes)
}

// ---------- renderer ----------

impl GpuScene {
    /// `samples`: 4 for antialiasing, 1 on software adapters.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, scene: &Scene, samples: u32, font: FontArc) -> GpuScene {
        let layer_count = scene.layer_list.len().max(1) as u32;
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("view"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let layers_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("layers"),
            size: wgpu::Extent3d { width: layer_count, height: 2, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("view"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("view"),
            layout: &bind_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniforms.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&layers_tex.create_view(&Default::default())),
                },
            ],
        });

        // text atlas
        let mut chars: Vec<char> = scene.texts.iter().flat_map(|t| t.value.chars()).filter(|c| !c.is_whitespace()).collect();
        chars.sort_unstable();
        chars.dedup();
        let atlas = Atlas::build(font, &chars, device.limits().max_texture_dimension_2d);
        let atlas_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("atlas"),
            size: wgpu::Extent3d { width: atlas.width, height: atlas.height, depth_or_array_layers: 1 },
            mip_level_count: atlas.levels.len() as u32,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let (mut lw, mut lh) = (atlas.width, atlas.height);
        for (lvl, data) in atlas.levels.iter().enumerate() {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo { texture: &atlas_tex, mip_level: lvl as u32, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                data,
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(lw * 4), rows_per_image: Some(lh) },
                wgpu::Extent3d { width: lw, height: lh, depth_or_array_layers: 1 },
            );
            (lw, lh) = ((lw / 2).max(1), (lh / 2).max(1));
        }
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("atlas"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let atlas_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("atlas"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let atlas_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("atlas"),
            layout: &atlas_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&atlas_tex.create_view(&Default::default())) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) },
            ],
        });
        let (text_bytes, text_vertices, text_boxes) = text_mesh(scene, &atlas);

        // pipelines
        let scene_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scene"),
            source: wgpu::ShaderSource::Wgsl(format!("{COMMON}{SCENE_SHADER}").into()),
        });
        let text_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("text"),
            source: wgpu::ShaderSource::Wgsl(format!("{COMMON}{TEXT_SHADER}").into()),
        });
        let scene_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scene"),
            bind_group_layouts: &[&bind_layout],
            push_constant_ranges: &[],
        });
        let text_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("text"),
            bind_group_layouts: &[&bind_layout, &atlas_layout],
            push_constant_ranges: &[],
        });
        let a_pos = wgpu::vertex_attr_array![0 => Float32x2];
        let a_col = wgpu::vertex_attr_array![1 => Unorm8x4];
        let a_lay = wgpu::vertex_attr_array![7 => Uint32];
        let a_inst = wgpu::vertex_attr_array![2 => Float32x2, 3 => Float32x2, 4 => Float32x2];
        let a_inst_col = wgpu::vertex_attr_array![5 => Unorm8x4, 6 => Unorm8x4];
        let a_inst_lay = wgpu::vertex_attr_array![8 => Uint16x2];
        let vtx = wgpu::VertexStepMode::Vertex;
        let ins = wgpu::VertexStepMode::Instance;
        let scene_buffers = [
            wgpu::VertexBufferLayout { array_stride: 8, step_mode: vtx, attributes: &a_pos },
            wgpu::VertexBufferLayout { array_stride: 4, step_mode: vtx, attributes: &a_col },
            wgpu::VertexBufferLayout { array_stride: 4, step_mode: vtx, attributes: &a_lay },
            wgpu::VertexBufferLayout { array_stride: 24, step_mode: ins, attributes: &a_inst },
            wgpu::VertexBufferLayout { array_stride: 8, step_mode: ins, attributes: &a_inst_col },
            wgpu::VertexBufferLayout { array_stride: 4, step_mode: ins, attributes: &a_inst_lay },
        ];
        let a_text = wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Unorm8x4, 3 => Uint16x2];
        let text_buffers = [wgpu::VertexBufferLayout { array_stride: 24, step_mode: vtx, attributes: &a_text }];
        let make = |label: &str,
                    module: &wgpu::ShaderModule,
                    layout: &wgpu::PipelineLayout,
                    buffers: &[wgpu::VertexBufferLayout],
                    topology,
                    ds: wgpu::DepthStencilState,
                    write: wgpu::ColorWrites,
                    blend: Option<wgpu::BlendState>| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                vertex: wgpu::VertexState { module, entry_point: Some("vs"), compilation_options: Default::default(), buffers },
                primitive: wgpu::PrimitiveState { topology, ..Default::default() },
                depth_stencil: Some(ds),
                multisample: wgpu::MultisampleState { count: samples, ..Default::default() },
                fragment: Some(wgpu::FragmentState {
                    module,
                    entry_point: Some("fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState { format: CACHE_FORMAT, blend, write_mask: write })],
                }),
                multiview: None,
                cache: None,
            })
        };
        use wgpu::{ColorWrites as W, CompareFunction as C, PrimitiveTopology as T, StencilOperation as S};
        let (sm, sl, sb) = (&scene_module, &scene_layout, &scene_buffers[..]);
        let lines = make("lines", sm, sl, sb, T::LineList, stencil(C::Always, S::Keep, 0), W::ALL, None);
        // fills: even-odd parity in the stencil, then color where it is odd (and reset it)
        let fill_stencil = make("fill parity", sm, sl, sb, T::TriangleList, stencil(C::Always, S::Invert, 1), W::empty(), None);
        let fill_color = make("fill color", sm, sl, sb, T::TriangleList, stencil(C::Equal, S::Zero, 1), W::ALL, None);
        let text = make(
            "text",
            &text_module,
            &text_layout,
            &text_buffers,
            T::TriangleList,
            stencil(C::Always, S::Keep, 0),
            W::ALL,
            Some(wgpu::BlendState::ALPHA_BLENDING),
        );

        GpuScene {
            samples,
            blocks: scene.blocks.clone(),
            layer_count,
            pos: buffer(device, "positions", &f32_bytes(&scene.positions)),
            col: buffer(device, "colors", &scene.colors),
            lay: buffer(device, "layers", &u16_as_u32_bytes(&scene.layers)),
            fill_pos: buffer(device, "fill positions", &f32_bytes(&scene.fill_positions)),
            fill_col: buffer(device, "fill colors", &scene.fill_colors),
            fill_lay: buffer(device, "fill layers", &u16_as_u32_bytes(&scene.fill_layers)),
            inst: buffer(device, "instances", &f32_bytes(&scene.instances)),
            inst_col: buffer(device, "instance colors", &scene.inst_colors),
            inst_lay: buffer(device, "instance layers", &u16_bytes(&scene.inst_layers)),
            text_vb: buffer(device, "texts", &text_bytes),
            text_vertices,
            uniforms,
            layers_tex,
            bind,
            atlas_bind,
            lines,
            fill_stencil,
            fill_color,
            text,
            text_boxes,
        }
    }

    pub fn set_layers(&self, queue: &wgpu::Queue, layers: &[LayerLook]) {
        let n = self.layer_count as usize;
        let mut px = vec![0u8; n * 2 * 4];
        for (i, l) in layers.iter().take(n).enumerate() {
            px[i * 4] = if l.on { 255 } else { 0 };
            px[i * 4 + 1] = if l.color.is_some() { 255 } else { 0 };
            px[i * 4 + 3] = 255;
            let c = l.color.unwrap_or(l.base);
            px[(n + i) * 4..(n + i) * 4 + 4].copy_from_slice(&[c[0], c[1], c[2], 255]);
        }
        queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &self.layers_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            &px,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(self.layer_count * 4), rows_per_image: Some(2) },
            wgpu::Extent3d { width: self.layer_count, height: 2, depth_or_array_layers: 1 },
        );
    }

    pub fn target(&self, device: &wgpu::Device, width: u32, height: u32) -> Target {
        let tex = |label: &str, format, samples, usage| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: samples,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let ra = wgpu::TextureUsages::RENDER_ATTACHMENT;
        let ms = (self.samples > 1).then(|| tex("cache msaa", CACHE_FORMAT, self.samples, ra).create_view(&Default::default()));
        let depth = tex("cache stencil", DEPTH_FORMAT, self.samples, ra);
        let resolve = tex("cache", CACHE_FORMAT, 1, ra | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC);
        Target {
            width,
            height,
            ms_view: ms,
            depth_view: depth.create_view(&Default::default()),
            resolve_view: resolve.create_view(&Default::default()),
            _resolve: resolve,
        }
    }

    /// Draw the whole scene into `target`, centered on `view`. Returns the submission index.
    pub fn render(&self, device: &wgpu::Device, queue: &wgpu::Queue, target: &Target, view: View, look: &Look, layer0: u16) -> wgpu::SubmissionIndex {
        let f = |c: [u8; 3]| [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0, 1.0];
        let mono = look.mono.map_or([0.0; 4], f);
        let mut u: Vec<u8> = f32_bytes(&[
            view.cx as f32,
            view.cy as f32,
            (2.0 * view.ppu / target.width as f64) as f32,
            (2.0 * view.ppu / target.height as f64) as f32,
        ]);
        u.extend(f32_bytes(&mono));
        u.extend(f32_bytes(&[0.08, 0.09, 0.1, 1.0]));
        for v in [layer0 as u32, look.mono.is_some() as u32, look.adapt as u32, 0] {
            u.extend_from_slice(&v.to_le_bytes());
        }
        queue.write_buffer(&self.uniforms, 0, &u);
        let bg = look.bg.map(|c| c as f64 / 255.0);
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("scene") });
        {
            let (view_tex, resolve) = match &target.ms_view {
                Some(ms) => (ms, Some(&target.resolve_view)),
                None => (&target.resolve_view, None),
            };
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: view_tex,
                    depth_slice: None,
                    resolve_target: resolve,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: bg[0], g: bg[1], b: bg[2], a: 1.0 }),
                        store: if resolve.is_some() { wgpu::StoreOp::Discard } else { wgpu::StoreOp::Store },
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &target.depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Discard }),
                    stencil_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(0), store: wgpu::StoreOp::Discard }),
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_bind_group(0, &self.bind, &[]);
            pass.set_stencil_reference(1);
            let bind = |pass: &mut wgpu::RenderPass, pos: &wgpu::Buffer, col: &wgpu::Buffer, lay: &wgpu::Buffer, v: usize, i: usize| {
                pass.set_vertex_buffer(0, pos.slice((v * 8) as u64..));
                pass.set_vertex_buffer(1, col.slice((v * 4) as u64..));
                pass.set_vertex_buffer(2, lay.slice((v * 4) as u64..));
                pass.set_vertex_buffer(3, self.inst.slice((i * 24) as u64..));
                pass.set_vertex_buffer(4, self.inst_col.slice((i * 8) as u64..));
                pass.set_vertex_buffer(5, self.inst_lay.slice((i * 4) as u64..));
            };
            if look.fills {
                for b in self.blocks.iter().filter(|b| b.fill_count > 0) {
                    bind(&mut pass, &self.fill_pos, &self.fill_col, &self.fill_lay, b.fill_start, b.inst_start);
                    let (n, k) = (b.fill_count as u32, b.inst_count as u32);
                    pass.set_pipeline(&self.fill_stencil);
                    pass.draw(0..n, 0..k);
                    pass.set_pipeline(&self.fill_color);
                    pass.draw(0..n, 0..k);
                }
            }
            pass.set_pipeline(&self.lines);
            for b in self.blocks.iter().filter(|b| b.vertex_count > 0) {
                bind(&mut pass, &self.pos, &self.col, &self.lay, b.vertex_start, b.inst_start);
                pass.draw(0..b.vertex_count as u32, 0..b.inst_count as u32);
            }
            if look.texts && self.text_vertices > 0 {
                pass.set_pipeline(&self.text);
                pass.set_bind_group(1, &self.atlas_bind, &[]);
                pass.set_vertex_buffer(0, self.text_vb.slice(..));
                pass.draw(0..self.text_vertices, 0..1);
            }
        }
        queue.submit([enc.finish()])
    }
}

