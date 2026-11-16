//! Real-time convolution engine (placeholder until M4).

/// Crate version, for the CLI and the Python bindings.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    #[test]
    fn version_is_set() {
        assert!(!super::VERSION.is_empty());
    }
}
