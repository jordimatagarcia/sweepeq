//! Python bindings: `sweepeq_engine.Convolver`.
//!
//! ```python
//! import numpy as np, sweepeq_engine
//! c = sweepeq_engine.Convolver(taps, method="partitioned", block=256)
//! y = c.process(x)          # float32 in, float32 out, same length; continues across calls
//! c.latency                 # samples
//! ```

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::Mutex;

use sweepeq_core::{Convolver, Method, build};

/// A streaming convolver running in Rust.
#[pyclass(name = "Convolver")]
struct PyConvolver {
    // Python objects may be shared between threads; the lock is uncontended in normal use
    // and is not on any real-time path.
    inner: Mutex<Box<dyn Convolver>>,
    method: String,
}

impl PyConvolver {
    fn with<R>(&self, f: impl FnOnce(&mut dyn Convolver) -> R) -> R {
        let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        f(guard.as_mut())
    }
}

#[pymethods]
impl PyConvolver {
    #[new]
    #[pyo3(signature = (taps, method = "partitioned", block = 256))]
    fn new(taps: PyReadonlyArray1<'_, f32>, method: &str, block: usize) -> PyResult<Self> {
        let taps = taps.as_slice()?;
        if taps.is_empty() {
            return Err(PyValueError::new_err("taps must not be empty"));
        }
        let m = match method {
            "direct" => Method::Direct,
            "partitioned" => {
                if block < 2 || !block.is_power_of_two() {
                    return Err(PyValueError::new_err("block must be a power of two >= 2"));
                }
                Method::Partitioned { block }
            }
            other => return Err(PyValueError::new_err(format!("unknown method {other:?}"))),
        };
        Ok(Self {
            inner: Mutex::new(build(taps, m, 4096)),
            method: method.to_owned(),
        })
    }

    /// Filter a block of float32 samples; returns a new array of the same length.
    fn process<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let input = x.as_slice()?.to_vec();
        let mut out = vec![0.0f32; input.len()];
        // the GIL is released while Rust runs
        py.detach(|| self.with(|c| c.process(&input, &mut out)));
        Ok(out.into_pyarray(py))
    }

    /// Forget all history.
    fn reset(&self) {
        self.with(|c| c.reset());
    }

    #[getter]
    fn latency(&self) -> usize {
        self.with(|c| c.latency())
    }

    #[getter]
    fn taps(&self) -> usize {
        self.with(|c| c.taps())
    }

    fn __repr__(&self) -> String {
        format!(
            "Convolver({} taps, method={:?}, latency={})",
            self.taps(),
            self.method,
            self.latency()
        )
    }
}

#[pymodule]
fn sweepeq_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyConvolver>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
