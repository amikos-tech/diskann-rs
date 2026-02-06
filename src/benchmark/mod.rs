//! Benchmark utilities for evaluating DiskANN accuracy and performance.
//!
//! This module provides tools for:
//! - Loading standard ANN benchmark datasets (SIFT, GIST) in fvecs/ivecs format
//! - Computing ground truth nearest neighbors via brute force
//! - Calculating recall metrics (recall@k)
//! - Generating synthetic benchmark data for testing
//!
//! # Example
//!
//! ```no_run
//! use ami_diskann::benchmark::{generate_small_benchmark, calculate_recall};
//!
//! // Generate synthetic data with ground truth
//! let data = generate_small_benchmark(42);
//!
//! // ... build and search your index ...
//! // let results = search_results;
//!
//! // Calculate recall
//! // let stats = calculate_recall(&results, &data.ground_truth, 10);
//! // println!("Recall@10: {:.4}", stats.mean);
//! ```
//!
//! # File Formats
//!
//! Supports the standard corpus-texmex file formats:
//! - **fvecs**: Float32 vectors `[dim: u32][values: f32 * dim]`
//! - **ivecs**: Int32 vectors `[dim: i32][values: i32 * dim]`
//! - **bvecs**: Byte vectors `[dim: u32][values: u8 * dim]`

mod formats;
mod ground_truth;
mod recall;
mod synthetic;

pub use formats::{
    bvecs_to_f32, ivecs_to_u32, read_bvecs, read_fvecs, read_ivecs, write_fvecs, write_ivecs,
};
pub use ground_truth::{
    compute_ground_truth, compute_ground_truth_parallel, compute_ground_truth_with_distances,
    cosine_distance, inner_product_distance, l2_distance, l2_distance_sqrt, DistanceFunc,
};
pub use recall::{calculate_recall, calculate_recall_at_k, format_recall_report, RecallStats};
pub use synthetic::{
    generate_clustered_data, generate_cosine_benchmark, generate_medium_benchmark,
    generate_small_benchmark, generate_synthetic_benchmark, SyntheticConfig,
};

/// Standard ANN benchmark datasets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dataset {
    /// SIFT10K (siftsmall): 10K vectors, 128 dimensions, ~2MB
    Sift10K,
    /// SIFT1M: 1M vectors, 128 dimensions, ~168MB
    Sift1M,
    /// GIST1M: 1M vectors, 960 dimensions, ~2.6GB
    Gist1M,
}

impl Dataset {
    /// Returns the name of the dataset (used for file prefixes).
    pub fn name(&self) -> &'static str {
        match self {
            Dataset::Sift10K => "siftsmall",
            Dataset::Sift1M => "sift",
            Dataset::Gist1M => "gist",
        }
    }

    /// Returns the FTP download URL for the dataset.
    pub fn url(&self) -> &'static str {
        match self {
            Dataset::Sift10K => "ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz",
            Dataset::Sift1M => "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz",
            Dataset::Gist1M => "ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz",
        }
    }

    /// Returns the expected number of base vectors.
    pub fn num_base(&self) -> usize {
        match self {
            Dataset::Sift10K => 10_000,
            Dataset::Sift1M => 1_000_000,
            Dataset::Gist1M => 1_000_000,
        }
    }

    /// Returns the expected dimension.
    pub fn dimension(&self) -> usize {
        match self {
            Dataset::Sift10K | Dataset::Sift1M => 128,
            Dataset::Gist1M => 960,
        }
    }
}

/// Loaded benchmark data containing vectors and ground truth.
#[derive(Debug, Clone)]
pub struct BenchmarkData {
    /// Base vectors to index and search
    pub base_vectors: Vec<Vec<f32>>,
    /// Query vectors for search
    pub query_vectors: Vec<Vec<f32>>,
    /// Ground truth nearest neighbor IDs per query (ordered by distance)
    pub ground_truth: Vec<Vec<u32>>,
    /// Dimensionality of vectors
    pub dimension: usize,
    /// Name of the dataset
    pub name: String,
}

impl BenchmarkData {
    /// Creates new benchmark data.
    pub fn new(
        base_vectors: Vec<Vec<f32>>,
        query_vectors: Vec<Vec<f32>>,
        ground_truth: Vec<Vec<u32>>,
        name: impl Into<String>,
    ) -> Self {
        let dimension = base_vectors.first().map(|v| v.len()).unwrap_or(0);
        Self {
            base_vectors,
            query_vectors,
            ground_truth,
            dimension,
            name: name.into(),
        }
    }

    /// Loads a dataset from a directory containing fvecs/ivecs files.
    ///
    /// Expects files named:
    /// - `{name}_base.fvecs` - base vectors
    /// - `{name}_query.fvecs` - query vectors
    /// - `{name}_groundtruth.ivecs` - ground truth neighbors
    pub fn load_from_directory(dir: &std::path::Path, dataset: Dataset) -> std::io::Result<Self> {
        let name = dataset.name();
        let base_path = dir.join(format!("{}_base.fvecs", name));
        let query_path = dir.join(format!("{}_query.fvecs", name));
        let gt_path = dir.join(format!("{}_groundtruth.ivecs", name));

        let base_vectors = read_fvecs(&base_path)?;
        let query_vectors = read_fvecs(&query_path)?;
        let gt_ivecs = read_ivecs(&gt_path)?;

        let ground_truth = ivecs_to_u32(&gt_ivecs);
        let dimension = base_vectors.first().map(|v| v.len()).unwrap_or(0);

        Ok(Self {
            base_vectors,
            query_vectors,
            ground_truth,
            dimension,
            name: name.to_string(),
        })
    }

    /// Returns the number of base vectors.
    pub fn num_base(&self) -> usize {
        self.base_vectors.len()
    }

    /// Returns the number of query vectors.
    pub fn num_queries(&self) -> usize {
        self.query_vectors.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_names() {
        assert_eq!(Dataset::Sift10K.name(), "siftsmall");
        assert_eq!(Dataset::Sift1M.name(), "sift");
        assert_eq!(Dataset::Gist1M.name(), "gist");
    }

    #[test]
    fn test_dataset_dimensions() {
        assert_eq!(Dataset::Sift10K.dimension(), 128);
        assert_eq!(Dataset::Sift1M.dimension(), 128);
        assert_eq!(Dataset::Gist1M.dimension(), 960);
    }

    #[test]
    fn test_benchmark_data_new() {
        let base = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let query = vec![vec![1.5, 2.5, 3.5]];
        let gt = vec![vec![0, 1]];

        let data = BenchmarkData::new(base, query, gt, "test");

        assert_eq!(data.dimension, 3);
        assert_eq!(data.name, "test");
        assert_eq!(data.num_base(), 2);
        assert_eq!(data.num_queries(), 1);
    }
}
