//! Reader for the `.mf` model bundle format (`moonfile.py`).
//!
//! Layout: `b"MOON"`, `u8` version (1), then repeated
//! `{u32 BE name_len, name, u64 BE content_len, content}` until EOF.

use std::{fs::File, io::Read, path::Path};

use memmap2::Mmap;

use crate::{Error, Result};

const MAGIC: &[u8; 4] = b"MOON";
const VERSION: u8 = 1;

/// Raw bundle bytes: memory-mapped when uncompressed, decompressed otherwise.
pub(crate) enum ModelBytes {
    Mapped(Mmap),
    Owned(Vec<u8>),
}

impl ModelBytes {
    pub(crate) fn as_slice(&self) -> &[u8] {
        match self {
            ModelBytes::Mapped(m) => m,
            ModelBytes::Owned(v) => v,
        }
    }
}

/// Opens a `.mf` file; a `.gz` suffix is decompressed into memory.
pub(crate) fn open(path: &Path) -> Result<ModelBytes> {
    let file = File::open(path)?;
    if path.extension().is_some_and(|e| e == "gz") {
        let mut out = Vec::new();
        flate2::read::GzDecoder::new(file).read_to_end(&mut out)?;
        return Ok(ModelBytes::Owned(out));
    }
    // SAFETY: the model file is opened read-only and is not modified by this
    // process while mapped; concurrent external modification is unsupported.
    let map = unsafe { Mmap::map(&file)? };
    Ok(ModelBytes::Mapped(map))
}

fn take<'a>(bytes: &mut &'a [u8], n: usize) -> Result<&'a [u8]> {
    if bytes.len() < n {
        return Err(Error::InvalidModel("Truncated .mf file".into()));
    }
    let (head, rest) = bytes.split_at(n);
    *bytes = rest;
    Ok(head)
}

/// Parses all `(file name, content)` entries without copying.
pub(crate) fn entries(bytes: &[u8]) -> Result<Vec<(&str, &[u8])>> {
    let mut rest = bytes;
    let magic = take(&mut rest, 4)?;
    if magic != MAGIC {
        return Err(Error::InvalidModel(format!(
            "Invalid magic bytes: {magic:?}"
        )));
    }
    let version = take(&mut rest, 1)?[0];
    if version != VERSION {
        return Err(Error::InvalidModel(format!(
            "Unsupported version: {version}"
        )));
    }
    let mut out = Vec::new();
    while !rest.is_empty() {
        let name_len = u32::from_be_bytes(take(&mut rest, 4)?.try_into().map_err(|_| truncated())?);
        let name = std::str::from_utf8(take(&mut rest, name_len as usize)?)
            .map_err(|e| Error::InvalidModel(format!("Invalid file name in .mf: {e}")))?;
        let len = u64::from_be_bytes(take(&mut rest, 8)?.try_into().map_err(|_| truncated())?);
        let len = usize::try_from(len).map_err(|_| truncated())?;
        out.push((name, take(&mut rest, len)?));
    }
    Ok(out)
}

fn truncated() -> Error {
    Error::InvalidModel("Truncated .mf file".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(name: &str, content: &[u8]) -> Vec<u8> {
        let mut v = (name.len() as u32).to_be_bytes().to_vec();
        v.extend_from_slice(name.as_bytes());
        v.extend_from_slice(&(content.len() as u64).to_be_bytes());
        v.extend_from_slice(content);
        v
    }

    fn archive(version: u8) -> Vec<u8> {
        let mut v = b"MOON".to_vec();
        v.push(version);
        v.extend(entry("config.json", b"{}"));
        v.extend(entry("dir/vision_encoder.onnx", &[1, 2, 3]));
        v
    }

    #[test]
    fn entries_should_round_trip_archive() {
        let bytes = archive(1);
        let e = entries(&bytes).unwrap();
        assert_eq!(
            e,
            vec![
                ("config.json", &b"{}"[..]),
                ("dir/vision_encoder.onnx", &[1u8, 2, 3][..])
            ]
        );
    }

    #[test]
    fn entries_should_reject_bad_magic() {
        let mut bytes = archive(1);
        bytes[0] = b'X';
        let err = entries(&bytes).unwrap_err().to_string();
        assert!(err.contains("Invalid magic bytes"), "{err}");
    }

    #[test]
    fn entries_should_reject_unknown_version() {
        let err = entries(&archive(2)).unwrap_err().to_string();
        assert_eq!(err, "Unsupported version: 2");
    }

    #[test]
    fn entries_should_reject_truncated_content() {
        let bytes = archive(1);
        let err = entries(&bytes[..bytes.len() - 1]).unwrap_err();
        assert!(matches!(err, Error::InvalidModel(m) if m == "Truncated .mf file"));
    }
}
