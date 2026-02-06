//! File format parsers for standard ANN benchmark datasets.
//!
//! Supports fvecs, ivecs, and bvecs formats from corpus-texmex.irisa.fr.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

/// Reads vectors from a file in fvecs format.
///
/// Fvecs format: `[4 bytes: dim as u32 LE][dim*4 bytes: f32 values LE]` per vector.
///
/// # Example
/// ```no_run
/// use ami_diskann::benchmark::read_fvecs;
/// let vectors = read_fvecs("sift_base.fvecs").unwrap();
/// ```
pub fn read_fvecs(path: impl AsRef<Path>) -> std::io::Result<Vec<Vec<f32>>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    read_fvecs_from_reader(&mut reader)
}

/// Reads fvecs from a reader.
pub fn read_fvecs_from_reader<R: Read>(reader: &mut R) -> std::io::Result<Vec<Vec<f32>>> {
    let mut vectors = Vec::new();
    let mut dim_buf = [0u8; 4];

    loop {
        // Read dimension
        match reader.read_exact(&mut dim_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        }

        let dim = u32::from_le_bytes(dim_buf) as usize;
        if dim == 0 || dim > 100_000 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("invalid vector dimension: {}", dim),
            ));
        }

        // Read vector values
        let mut vec = vec![0.0f32; dim];
        let vec_bytes = unsafe {
            std::slice::from_raw_parts_mut(vec.as_mut_ptr() as *mut u8, dim * 4)
        };
        reader.read_exact(vec_bytes)?;

        // Convert from little-endian if needed (no-op on little-endian systems)
        #[cfg(target_endian = "big")]
        for v in &mut vec {
            *v = f32::from_le_bytes(v.to_ne_bytes());
        }

        vectors.push(vec);
    }

    if vectors.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "no vectors found in file",
        ));
    }

    Ok(vectors)
}

/// Writes vectors to a file in fvecs format.
pub fn write_fvecs(path: impl AsRef<Path>, vectors: &[Vec<f32>]) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    write_fvecs_to_writer(&mut writer, vectors)
}

/// Writes fvecs to a writer.
pub fn write_fvecs_to_writer<W: Write>(writer: &mut W, vectors: &[Vec<f32>]) -> std::io::Result<()> {
    for vec in vectors {
        let dim = vec.len() as u32;
        writer.write_all(&dim.to_le_bytes())?;

        let vec_bytes = unsafe {
            std::slice::from_raw_parts(vec.as_ptr() as *const u8, vec.len() * 4)
        };
        writer.write_all(vec_bytes)?;
    }
    Ok(())
}

/// Reads integer vectors from a file in ivecs format.
///
/// Ivecs format: `[4 bytes: dim as i32 LE][dim*4 bytes: i32 values LE]` per vector.
/// Used primarily for ground truth files (nearest neighbor IDs).
pub fn read_ivecs(path: impl AsRef<Path>) -> std::io::Result<Vec<Vec<i32>>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    read_ivecs_from_reader(&mut reader)
}

/// Reads ivecs from a reader.
pub fn read_ivecs_from_reader<R: Read>(reader: &mut R) -> std::io::Result<Vec<Vec<i32>>> {
    let mut vectors = Vec::new();
    let mut dim_buf = [0u8; 4];

    loop {
        match reader.read_exact(&mut dim_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        }

        let dim = i32::from_le_bytes(dim_buf) as usize;
        if dim == 0 || dim > 100_000 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("invalid vector dimension: {}", dim),
            ));
        }

        let mut vec = vec![0i32; dim];
        let vec_bytes = unsafe {
            std::slice::from_raw_parts_mut(vec.as_mut_ptr() as *mut u8, dim * 4)
        };
        reader.read_exact(vec_bytes)?;

        #[cfg(target_endian = "big")]
        for v in &mut vec {
            *v = i32::from_le_bytes(v.to_ne_bytes());
        }

        vectors.push(vec);
    }

    if vectors.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "no vectors found in file",
        ));
    }

    Ok(vectors)
}

/// Writes integer vectors to a file in ivecs format.
pub fn write_ivecs(path: impl AsRef<Path>, vectors: &[Vec<i32>]) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    write_ivecs_to_writer(&mut writer, vectors)
}

/// Writes ivecs to a writer.
pub fn write_ivecs_to_writer<W: Write>(writer: &mut W, vectors: &[Vec<i32>]) -> std::io::Result<()> {
    for vec in vectors {
        let dim = vec.len() as i32;
        writer.write_all(&dim.to_le_bytes())?;

        let vec_bytes = unsafe {
            std::slice::from_raw_parts(vec.as_ptr() as *const u8, vec.len() * 4)
        };
        writer.write_all(vec_bytes)?;
    }
    Ok(())
}

/// Reads byte vectors from a file in bvecs format.
///
/// Bvecs format: `[4 bytes: dim as u32 LE][dim bytes: u8 values]` per vector.
pub fn read_bvecs(path: impl AsRef<Path>) -> std::io::Result<Vec<Vec<u8>>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    read_bvecs_from_reader(&mut reader)
}

/// Reads bvecs from a reader.
pub fn read_bvecs_from_reader<R: Read>(reader: &mut R) -> std::io::Result<Vec<Vec<u8>>> {
    let mut vectors = Vec::new();
    let mut dim_buf = [0u8; 4];

    loop {
        match reader.read_exact(&mut dim_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        }

        let dim = u32::from_le_bytes(dim_buf) as usize;
        if dim == 0 || dim > 100_000 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("invalid vector dimension: {}", dim),
            ));
        }

        let mut vec = vec![0u8; dim];
        reader.read_exact(&mut vec)?;
        vectors.push(vec);
    }

    if vectors.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "no vectors found in file",
        ));
    }

    Ok(vectors)
}

/// Converts byte vectors to f32 vectors.
pub fn bvecs_to_f32(bvecs: &[Vec<u8>]) -> Vec<Vec<f32>> {
    bvecs
        .iter()
        .map(|v| v.iter().map(|&b| b as f32).collect())
        .collect()
}

/// Converts i32 ground truth to u32.
pub fn ivecs_to_u32(ivecs: &[Vec<i32>]) -> Vec<Vec<u32>> {
    ivecs
        .iter()
        .map(|v| v.iter().map(|&x| x as u32).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_fvecs_roundtrip() {
        let vectors = vec![
            vec![1.0f32, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
        ];

        let mut buf = Vec::new();
        write_fvecs_to_writer(&mut buf, &vectors).unwrap();

        let mut cursor = Cursor::new(buf);
        let read_back = read_fvecs_from_reader(&mut cursor).unwrap();

        assert_eq!(vectors, read_back);
    }

    #[test]
    fn test_ivecs_roundtrip() {
        let vectors = vec![
            vec![0i32, 1, 2, 3, 4],
            vec![10, 11, 12, 13, 14],
        ];

        let mut buf = Vec::new();
        write_ivecs_to_writer(&mut buf, &vectors).unwrap();

        let mut cursor = Cursor::new(buf);
        let read_back = read_ivecs_from_reader(&mut cursor).unwrap();

        assert_eq!(vectors, read_back);
    }

    #[test]
    fn test_bvecs_read() {
        // Create bvecs data: dim=4, values=[1,2,3,4] and [255,128,64,0]
        let data = vec![
            4, 0, 0, 0, // dim=4 (little-endian u32)
            1, 2, 3, 4, // values
            4, 0, 0, 0, // dim=4
            255, 128, 64, 0, // values
        ];

        let mut cursor = Cursor::new(data);
        let vectors = read_bvecs_from_reader(&mut cursor).unwrap();

        assert_eq!(vectors.len(), 2);
        assert_eq!(vectors[0], vec![1, 2, 3, 4]);
        assert_eq!(vectors[1], vec![255, 128, 64, 0]);
    }

    #[test]
    fn test_bvecs_to_f32() {
        let bvecs = vec![vec![0u8, 128, 255]];
        let fvecs = bvecs_to_f32(&bvecs);
        assert_eq!(fvecs[0], vec![0.0f32, 128.0, 255.0]);
    }
}
