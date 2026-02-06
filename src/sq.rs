//! # Scalar Quantization Trait and Implementations
//!
//! Provides a `VectorQuantizer` trait for composable vector quantization,
//! with implementations for F16 and Int8 scalar quantization.

use half::f16;

/// Trait for vector quantization strategies.
///
/// Implementations encode high-dimensional float vectors into compact byte
/// representations and provide distance computation on the compressed form.
pub trait VectorQuantizer: Send + Sync {
    /// Encode a float vector into quantized bytes.
    fn encode(&self, vector: &[f32]) -> Vec<u8>;

    /// Decode quantized bytes back to approximate float vector.
    fn decode(&self, codes: &[u8]) -> Vec<f32>;

    /// Compute L2 squared distance between a float query and quantized codes.
    fn asymmetric_distance(&self, query: &[f32], codes: &[u8]) -> f32;

    /// Compression ratio vs F32 (e.g. 2.0 for F16, 4.0 for Int8).
    fn compression_ratio(&self, dim: usize) -> f32;
}

/// F16 (half-precision) scalar quantizer.
///
/// Stateless — simply converts each f32 element to IEEE 754 half-precision.
/// Provides 2x compression with very low distortion.
pub struct F16Quantizer;

impl VectorQuantizer for F16Quantizer {
    fn encode(&self, vector: &[f32]) -> Vec<u8> {
        let f16_values: Vec<f16> = vector.iter().map(|&v| f16::from_f32(v)).collect();
        let bytes: &[u8] = bytemuck::cast_slice(&f16_values);
        bytes.to_vec()
    }

    fn decode(&self, codes: &[u8]) -> Vec<f32> {
        let f16_values: &[f16] = bytemuck::cast_slice(codes);
        f16_values.iter().map(|v| v.to_f32()).collect()
    }

    fn asymmetric_distance(&self, query: &[f32], codes: &[u8]) -> f32 {
        let decoded = self.decode(codes);
        query
            .iter()
            .zip(decoded.iter())
            .map(|(q, d)| {
                let diff = q - d;
                diff * diff
            })
            .sum()
    }

    fn compression_ratio(&self, _dim: usize) -> f32 {
        2.0
    }
}

/// Int8 scalar quantizer with per-dimension scaling.
///
/// Trained on a dataset to compute per-dimension min/max values.
/// Each dimension is independently mapped to [0, 255].
pub struct Int8Quantizer {
    mins: Vec<f32>,
    ranges: Vec<f32>,
}

impl Int8Quantizer {
    /// Train an Int8 quantizer by computing per-dimension min/max from training vectors.
    pub fn train(vectors: &[impl AsRef<[f32]>], dim: usize) -> Self {
        let mut mins = vec![f32::INFINITY; dim];
        let mut maxs = vec![f32::NEG_INFINITY; dim];

        for v in vectors {
            let v = v.as_ref();
            for (d, &val) in v.iter().enumerate().take(dim) {
                if val < mins[d] {
                    mins[d] = val;
                }
                if val > maxs[d] {
                    maxs[d] = val;
                }
            }
        }

        let ranges: Vec<f32> = mins
            .iter()
            .zip(maxs.iter())
            .map(|(&min, &max)| {
                let min = if min.is_finite() { min } else { 0.0 };
                let max = if max.is_finite() { max } else { 0.0 };
                if max > min { max - min } else { 1.0 }
            })
            .collect();

        // Fix non-finite mins
        for m in &mut mins {
            if !m.is_finite() {
                *m = 0.0;
            }
        }

        Self { mins, ranges }
    }

    /// Construct from pre-computed per-dimension scales.
    pub fn from_scales(mins: Vec<f32>, ranges: Vec<f32>) -> Self {
        Self { mins, ranges }
    }

    /// Get the per-dimension min values.
    pub fn mins(&self) -> &[f32] {
        &self.mins
    }

    /// Get the per-dimension range values.
    pub fn ranges(&self) -> &[f32] {
        &self.ranges
    }
}

impl VectorQuantizer for Int8Quantizer {
    fn encode(&self, vector: &[f32]) -> Vec<u8> {
        vector
            .iter()
            .enumerate()
            .map(|(d, &v)| {
                ((v - self.mins[d]) / self.ranges[d] * 255.0)
                    .round()
                    .clamp(0.0, 255.0) as u8
            })
            .collect()
    }

    fn decode(&self, codes: &[u8]) -> Vec<f32> {
        codes
            .iter()
            .enumerate()
            .map(|(d, &c)| c as f32 / 255.0 * self.ranges[d] + self.mins[d])
            .collect()
    }

    fn asymmetric_distance(&self, query: &[f32], codes: &[u8]) -> f32 {
        let decoded = self.decode(codes);
        query
            .iter()
            .zip(decoded.iter())
            .map(|(q, d)| {
                let diff = q - d;
                diff * diff
            })
            .sum()
    }

    fn compression_ratio(&self, _dim: usize) -> f32 {
        4.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_encode_decode_roundtrip() {
        let q = F16Quantizer;
        let vector = vec![1.0, -2.5, 3.14, 0.0, 100.0];
        let codes = q.encode(&vector);
        let decoded = q.decode(&codes);

        assert_eq!(decoded.len(), vector.len());
        for (orig, dec) in vector.iter().zip(decoded.iter()) {
            assert!(
                (orig - dec).abs() < 0.1,
                "F16 roundtrip: {} -> {}",
                orig,
                dec
            );
        }
    }

    #[test]
    fn test_f16_compression_ratio() {
        let q = F16Quantizer;
        assert!((q.compression_ratio(128) - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_int8_train_encode_decode() {
        let vectors = vec![
            vec![0.0, 0.0],
            vec![1.0, 1000.0],
            vec![0.5, 500.0],
        ];
        let q = Int8Quantizer::train(&vectors, 2);

        let codes = q.encode(&vectors[2]);
        let decoded = q.decode(&codes);

        assert_eq!(decoded.len(), 2);
        // Per-dimension: dim 0 error ~= 1/255, dim 1 error ~= 1000/255
        assert!(
            (decoded[0] - 0.5).abs() < 0.01,
            "dim 0: {}",
            decoded[0]
        );
        assert!(
            (decoded[1] - 500.0).abs() < 5.0,
            "dim 1: {}",
            decoded[1]
        );
    }

    #[test]
    fn test_int8_per_dimension_precision() {
        let vectors = vec![
            vec![0.0, 0.0],
            vec![1.0, 1000.0],
        ];
        let q = Int8Quantizer::train(&vectors, 2);

        // Encode [0.5, 500.0]
        let test = vec![0.5, 500.0];
        let codes = q.encode(&test);
        let decoded = q.decode(&codes);

        // dim 0 range is 1.0, so error ~= 1/255 ~= 0.004
        let dim0_err = (decoded[0] - 0.5).abs();
        assert!(
            dim0_err < 0.01,
            "dim 0 error should be ~0.004, got {}",
            dim0_err
        );

        // dim 1 range is 1000, so error ~= 1000/255 ~= 3.9
        let dim1_err = (decoded[1] - 500.0).abs();
        assert!(
            dim1_err < 5.0,
            "dim 1 error should be ~2.0, got {}",
            dim1_err
        );
    }

    #[test]
    fn test_int8_compression_ratio() {
        let vectors = vec![vec![0.0; 128], vec![1.0; 128]];
        let q = Int8Quantizer::train(&vectors, 128);
        assert!((q.compression_ratio(128) - 4.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_int8_from_scales() {
        let mins = vec![0.0, -10.0];
        let ranges = vec![1.0, 20.0];
        let q = Int8Quantizer::from_scales(mins.clone(), ranges.clone());

        assert_eq!(q.mins(), &mins[..]);
        assert_eq!(q.ranges(), &ranges[..]);

        let codes = q.encode(&[0.5, 0.0]);
        let decoded = q.decode(&codes);
        assert!((decoded[0] - 0.5).abs() < 0.01);
        assert!((decoded[1] - 0.0).abs() < 0.2);
    }

    #[test]
    fn test_f16_asymmetric_distance() {
        let q = F16Quantizer;
        let query = vec![1.0, 2.0, 3.0];
        let target = vec![4.0, 5.0, 6.0];
        let codes = q.encode(&target);

        let dist = q.asymmetric_distance(&query, &codes);
        // L2 squared: (1-4)^2 + (2-5)^2 + (3-6)^2 = 9 + 9 + 9 = 27
        assert!(
            (dist - 27.0).abs() < 0.1,
            "Expected ~27.0, got {}",
            dist
        );
    }

    #[test]
    fn test_int8_asymmetric_distance() {
        let vectors = vec![
            vec![0.0, 0.0, 0.0],
            vec![10.0, 10.0, 10.0],
        ];
        let q = Int8Quantizer::train(&vectors, 3);

        let query = vec![1.0, 2.0, 3.0];
        let target = vec![4.0, 5.0, 6.0];
        let codes = q.encode(&target);

        let dist = q.asymmetric_distance(&query, &codes);
        // Should be close to L2 squared: 9+9+9 = 27
        assert!(
            (dist - 27.0).abs() < 2.0,
            "Expected ~27.0, got {}",
            dist
        );
    }
}
