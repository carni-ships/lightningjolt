//! Setup caching with memory-mapped files, compression, and lazy loading
//!
//! This module provides high-performance setup caching with:
//! - SHA256 hash-based cache filenames to avoid collisions
//! - Memory-mapped file support for large setups
//! - LZ4 compression for fast disk I/O
//! - Versioned cache format for forward compatibility

use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::PathBuf;

#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

/// Current cache version for forward compatibility
const CACHE_VERSION: u32 = 1;

/// Magic bytes to identify valid cache files
const CACHE_MAGIC: [u8; 4] = [b'D', b'O', b'R', b'Y'];

/// Cache file header structure (32 bytes)
#[repr(C)]
struct CacheHeader {
    magic: [u8; 4],       // 4 bytes: "DORY"
    version: u32,         // 4 bytes: version number
    params_hash: [u8; 32], // 32 bytes: SHA256 of parameters
    compressed_size: u64,  // 8 bytes: compressed data size
    decompressed_size: u64, // 8 bytes: original data size
    reserved: [u8; 4],    // 4 bytes: reserved for future use
}

impl CacheHeader {
    fn new(params_hash: [u8; 32], compressed_size: u64, decompressed_size: u64) -> Self {
        Self {
            magic: CACHE_MAGIC,
            version: CACHE_VERSION,
            params_hash,
            compressed_size,
            decompressed_size,
            reserved: [0u8; 4],
        }
    }

    fn to_bytes(&self) -> [u8; 32] {
        let mut bytes = [0u8; 32];
        bytes[0..4].copy_from_slice(&self.magic);
        bytes[4..8].copy_from_slice(&self.version.to_le_bytes());
        bytes[8..40].copy_from_slice(&self.params_hash);
        bytes[40..48].copy_from_slice(&self.compressed_size.to_le_bytes());
        bytes[48..56].copy_from_slice(&self.decompressed_size.to_le_bytes());
        bytes
    }

    fn from_bytes(bytes: &[u8; 32]) -> Option<Self> {
        if &bytes[0..4] != &CACHE_MAGIC {
            return None;
        }
        let version = u32::from_le_bytes(bytes[4..8].try_into().ok()?);
        if version > CACHE_VERSION {
            return None; // Future version - incompatible
        }
        let mut params_hash = [0u8; 32];
        params_hash.copy_from_slice(&bytes[8..40]);
        let compressed_size = u64::from_le_bytes(bytes[40..48].try_into().ok()?);
        let decompressed_size = u64::from_le_bytes(bytes[48..56].try_into().ok()?);
        Some(Self {
            magic: CACHE_MAGIC,
            version,
            params_hash,
            compressed_size,
            decompressed_size,
            reserved: [0u8; 4],
        })
    }

    fn validate(&self) -> bool {
        self.magic == CACHE_MAGIC && self.version <= CACHE_VERSION
    }
}

/// Compute SHA256 hash of setup parameters
pub fn compute_params_hash(max_log_n: usize) -> [u8; 32] {
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    // Include version in hash to invalidate old caches
    hasher.update(b"DORY-v1:");
    hasher.update(&(max_log_n as u64).to_le_bytes());
    hasher.update(b":setup");
    hasher.finalize().into()
}

/// Get the cache directory path
pub fn get_cache_dir() -> Option<PathBuf> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let home = std::env::var("HOME").ok()?;
        let mut path = PathBuf::from(home);

        // Detect macOS vs Linux
        let macos_cache = {
            let mut test = PathBuf::from(&home);
            test.push("Library");
            test.push("Caches");
            test.exists()
        };

        if macos_cache {
            path.push("Library");
            path.push("Caches");
        } else {
            path.push(".cache");
        }

        path.push("dory");
        Some(path)
    }
    #[cfg(target_arch = "wasm32")]
    None
}

/// Get cache file path for given parameters
pub fn get_cache_path(max_log_n: usize, params_hash: [u8; 32]) -> Option<PathBuf> {
    get_cache_dir().map(|mut p| {
        let hash_hex = hex::encode(params_hash);
        p.push(format!("dory_{max_log_n}_{}.cache", hash_hex));
        p
    })
}

/// Compress data using LZ4
pub fn compress(data: &[u8]) -> Vec<u8> {
    lz4_flex::compress_prepend_size(data)
}

/// Decompress data using LZ4
pub fn decompress(compressed: &[u8], expected_size: usize) -> Option<Vec<u8>> {
    lz4_flex::decompress_size_prepended(compressed)
        .ok()
        .and_then(|d| if d.len() == expected_size { Some(d) } else { None })
}

/// Save setup to cache with compression
pub fn save_to_cache(max_log_n: usize, data: &[u8]) -> std::io::Result<PathBuf> {
    let params_hash = compute_params_hash(max_log_n);
    let cache_path = get_cache_path(max_log_n, params_hash)
        .ok_or_else(|| std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "Failed to determine cache directory"
        ))?;

    if let Some(parent) = cache_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Compress data
    let compressed = compress(data);

    // Create header
    let header = CacheHeader::new(params_hash, compressed.len() as u64, data.len() as u64);

    // Write header + compressed data
    let mut file = File::create(&cache_path)?;
    file.write_all(&header.to_bytes())?;
    file.write_all(&compressed)?;

    tracing::debug!(
        "Saved setup cache: {} bytes -> {} bytes (lz4)",
        data.len(),
        compressed.len()
    );

    Ok(cache_path)
}

/// Load setup from cache with verification
pub fn load_from_cache(max_log_n: usize) -> Option<Vec<u8>> {
    let params_hash = compute_params_hash(max_log_n);
    let cache_path = get_cache_path(max_log_n, params_hash)?;

    if !cache_path.exists() {
        return None;
    }

    // Read header
    let mut header_bytes = [0u8; 32];
    let mut file = OpenOptions::new().read(true).open(&cache_path).ok()?;
    file.read_exact(&mut header_bytes).ok()?;

    let header = CacheHeader::from_bytes(&header_bytes)?;
    if !header.validate() {
        tracing::warn!("Invalid cache header, ignoring");
        return None;
    }

    // Verify hash matches
    if header.params_hash != params_hash {
        tracing::warn!("Cache hash mismatch, ignoring");
        return None;
    }

    // Read compressed data
    let mut compressed = vec![0u8; header.compressed_size as usize];
    file.read_exact(&mut compressed).ok()?;

    // Decompress
    let decompressed = decompress(&compressed, header.decompressed_size as usize)?;

    tracing::debug!(
        "Loaded setup cache: {} bytes -> {} bytes (lz4)",
        compressed.len(),
        decompressed.len()
    );

    Some(decompressed)
}

/// Memory-mapped setup for lazy loading
pub struct MmapSetup {
    _file: File,
    data: memmap2::Mmap,
    size: usize,
}

impl MmapSetup {
    /// Load setup using memory-mapped file for lazy access
    pub fn from_cache(max_log_n: usize) -> Option<Self> {
        let params_hash = compute_params_hash(max_log_n);
        let cache_path = get_cache_path(max_log_n, params_hash)?;

        if !cache_path.exists() {
            return None;
        }

        // Read header to get decompressed size
        let mut header_bytes = [0u8; 32];
        let mut header_file = OpenOptions::new().read(true).open(&cache_path).ok()?;
        header_file.read_exact(&mut header_bytes).ok()?;

        let header = CacheHeader::from_bytes(&header_bytes)?;
        if !header.validate() || header.params_hash != params_hash {
            return None;
        }

        // Open file for memory mapping
        let file = OpenOptions::new()
            .read(true)
            .open(&cache_path)
            .ok()?;

        // Memory map the file (header is 32 bytes)
        let mmap = unsafe { memmap2::Mmap::map(&file) }.ok()?;

        Some(Self {
            _file: file,
            data: mmap,
            size: header.decompressed_size as usize,
        })
    }

    /// Get the underlying decompressed size
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Slice a portion of the setup data (for lazy loading specific parts)
    pub fn slice(&self, offset: usize, len: usize) -> Option<&[u8]> {
        if offset + len > self.size {
            return None;
        }
        // Data starts after 32-byte header
        let start = 32 + offset;
        self.data.get(start..start + len)
    }

    /// Get full decompressed data (reads and decompresses on demand)
    pub fn decompress_all(&self) -> Option<Vec<u8>> {
        // This is used when we need the full data at once
        let compressed_start = 32;
        let compressed_data = &self.data[compressed_start..];
        decompress(compressed_data, self.size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_roundtrip() {
        let params_hash = compute_params_hash(20);
        let header = CacheHeader::new(params_hash, 1000, 5000);
        let bytes = header.to_bytes();
        let recovered = CacheHeader::from_bytes(&bytes).unwrap();

        assert_eq!(recovered.magic, CACHE_MAGIC);
        assert_eq!(recovered.version, CACHE_VERSION);
        assert_eq!(recovered.params_hash, params_hash);
        assert_eq!(recovered.compressed_size, 1000);
        assert_eq!(recovered.decompressed_size, 5000);
    }

    #[test]
    fn test_compression() {
        let original: Vec<u8> = (0..10000).map(|i| i as u8).collect();
        let compressed = compress(&original);
        let decompressed = decompress(&compressed, original.len()).unwrap();
        assert_eq!(original, decompressed);
    }

    #[test]
    fn test_params_hash() {
        let h1 = compute_params_hash(20);
        let h2 = compute_params_hash(20);
        let h3 = compute_params_hash(21);

        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
    }
}