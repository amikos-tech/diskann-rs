//! Synthetic benchmark data generation.

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

use super::ground_truth::{compute_ground_truth_parallel, l2_distance, cosine_distance, DistanceFunc};
use super::BenchmarkData;

/// Configuration for synthetic benchmark generation.
#[derive(Debug, Clone)]
pub struct SyntheticConfig {
    /// Number of base vectors
    pub num_base: usize,
    /// Number of query vectors
    pub num_query: usize,
    /// Vector dimensionality
    pub dimension: usize,
    /// Number of clusters (0 for uniform random)
    pub num_clusters: usize,
    /// Cluster spread (smaller = tighter clusters)
    pub cluster_spread: f32,
    /// Number of ground truth neighbors to compute
    pub k: usize,
    /// Random seed for reproducibility
    pub seed: u64,
    /// Distance function for ground truth
    pub distance_func: DistanceFunc,
}

impl Default for SyntheticConfig {
    fn default() -> Self {
        Self {
            num_base: 10_000,
            num_query: 100,
            dimension: 128,
            num_clusters: 100,
            cluster_spread: 0.1,
            k: 100,
            seed: 42,
            distance_func: l2_distance,
        }
    }
}

/// Generates a synthetic benchmark dataset.
pub fn generate_synthetic_benchmark(config: &SyntheticConfig) -> BenchmarkData {
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);

    let (base_vectors, query_vectors) = if config.num_clusters > 0 {
        generate_clustered_data(&mut rng, config)
    } else {
        let base = generate_uniform_vectors(&mut rng, config.num_base, config.dimension);
        let query = generate_uniform_vectors(&mut rng, config.num_query, config.dimension);
        (base, query)
    };

    let ground_truth = compute_ground_truth_parallel(
        &base_vectors,
        &query_vectors,
        config.k,
        config.distance_func,
    );

    BenchmarkData::new(base_vectors, query_vectors, ground_truth, "synthetic")
}

/// Generates clustered data for more realistic benchmarking.
pub fn generate_clustered_data(
    rng: &mut impl Rng,
    config: &SyntheticConfig,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    // Generate cluster centers
    let centers: Vec<Vec<f32>> = (0..config.num_clusters)
        .map(|_| {
            let mut center: Vec<f32> = (0..config.dimension)
                .map(|_| rng.gen::<f32>() * 2.0 - 1.0)
                .collect();
            normalize(&mut center);
            center
        })
        .collect();

    // Generate base vectors around cluster centers
    let base_vectors: Vec<Vec<f32>> = (0..config.num_base)
        .map(|_| {
            let cluster = rng.gen_range(0..config.num_clusters);
            let center = &centers[cluster];
            center
                .iter()
                .map(|&c| c + rng.gen::<f32>() * config.cluster_spread * 2.0 - config.cluster_spread)
                .collect()
        })
        .collect();

    // Generate query vectors - 80% from clusters, 20% random
    let query_vectors: Vec<Vec<f32>> = (0..config.num_query)
        .map(|_| {
            if rng.gen::<f32>() < 0.8 {
                let cluster = rng.gen_range(0..config.num_clusters);
                let center = &centers[cluster];
                center
                    .iter()
                    .map(|&c| c + rng.gen::<f32>() * config.cluster_spread - config.cluster_spread * 0.5)
                    .collect()
            } else {
                (0..config.dimension)
                    .map(|_| rng.gen::<f32>() * 2.0 - 1.0)
                    .collect()
            }
        })
        .collect();

    (base_vectors, query_vectors)
}

fn generate_uniform_vectors(rng: &mut impl Rng, n: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect())
        .collect()
}

fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Generates a small benchmark for quick tests.
/// 1000 base vectors, 50 queries, 64 dimensions.
pub fn generate_small_benchmark(seed: u64) -> BenchmarkData {
    generate_synthetic_benchmark(&SyntheticConfig {
        num_base: 1_000,
        num_query: 50,
        dimension: 64,
        num_clusters: 20,
        cluster_spread: 0.15,
        k: 50,
        seed,
        distance_func: l2_distance,
    })
}

/// Generates a medium-sized benchmark.
/// 10000 base vectors, 100 queries, 128 dimensions.
pub fn generate_medium_benchmark(seed: u64) -> BenchmarkData {
    generate_synthetic_benchmark(&SyntheticConfig {
        num_base: 10_000,
        num_query: 100,
        dimension: 128,
        num_clusters: 100,
        cluster_spread: 0.1,
        k: 100,
        seed,
        distance_func: l2_distance,
    })
}

/// Generates a cosine similarity benchmark with normalized vectors.
pub fn generate_cosine_benchmark(seed: u64) -> BenchmarkData {
    let mut data = generate_synthetic_benchmark(&SyntheticConfig {
        num_base: 10_000,
        num_query: 100,
        dimension: 128,
        num_clusters: 100,
        cluster_spread: 0.1,
        k: 100,
        seed,
        distance_func: cosine_distance,
    });

    // Normalize all vectors
    for v in &mut data.base_vectors {
        normalize(v);
    }
    for v in &mut data.query_vectors {
        normalize(v);
    }

    // Recompute ground truth with normalized vectors
    data.ground_truth = compute_ground_truth_parallel(
        &data.base_vectors,
        &data.query_vectors,
        100,
        cosine_distance,
    );

    data.name = "synthetic-cosine".to_string();
    data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_small_benchmark() {
        let data = generate_small_benchmark(42);

        assert_eq!(data.base_vectors.len(), 1000);
        assert_eq!(data.query_vectors.len(), 50);
        assert_eq!(data.dimension, 64);
        assert_eq!(data.ground_truth.len(), 50);
        assert!(data.ground_truth[0].len() <= 50);
    }

    #[test]
    fn test_generate_medium_benchmark() {
        let data = generate_medium_benchmark(42);

        assert_eq!(data.base_vectors.len(), 10_000);
        assert_eq!(data.query_vectors.len(), 100);
        assert_eq!(data.dimension, 128);
    }

    #[test]
    fn test_reproducibility() {
        let data1 = generate_small_benchmark(42);
        let data2 = generate_small_benchmark(42);

        assert_eq!(data1.base_vectors, data2.base_vectors);
        assert_eq!(data1.query_vectors, data2.query_vectors);
        assert_eq!(data1.ground_truth, data2.ground_truth);
    }

    #[test]
    fn test_different_seeds() {
        let data1 = generate_small_benchmark(42);
        let data2 = generate_small_benchmark(123);

        assert_ne!(data1.base_vectors, data2.base_vectors);
    }

    #[test]
    fn test_ground_truth_validity() {
        let data = generate_small_benchmark(42);

        // First result should be a valid index
        for gt in &data.ground_truth {
            for &id in gt {
                assert!((id as usize) < data.base_vectors.len());
            }
        }
    }
}
