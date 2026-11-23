//! Reader for DWG files: R2000 (AC1015), R2007 (AC1021) and R2010+ (AC1024, AC1027, AC1032), 2D subset.
//! Based only on the Open Design Alliance "Open Design Specification for .dwg files".

pub mod bits;
pub mod compress;
pub mod compress21;
pub mod container;
pub mod crc;
pub mod handles;
pub mod objects;
pub mod classes;
pub mod drawing;
pub mod entities;
pub mod geom;
pub mod aci;
pub mod curves;
pub mod hatch;
pub mod text;
pub mod info;
#[cfg(feature = "desktop")]
pub mod desktop;
pub mod textlayout;
pub mod pick;
pub mod png;
