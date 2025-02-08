use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use reqwest;

/// A simple blocking GET written in Rust via reqwest.
/// It fully downloads (and discards) the response body.
#[pyfunction]
fn get(url: &str) -> PyResult<()> {
    // Create a Tokio runtime on the fly, blocking for the GET request.
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let resp = reqwest::get(url)
            .await
            .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))?;

        // Download the entire response into memory, then discard it
        resp.bytes()
            .await
            .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))?;

        // Help the compiler out by specifying the success and error types
        Ok::<(), pyo3::PyErr>(())
    })?;

    // If we reach here, everything succeeded
    Ok(())
}

/// This is the module initialiser. The name here (`fastr6`) must match
/// the library name in Cargo.toml ([lib] name = "fastr6").
#[pymodule]
fn fastr6(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get, m)?)?;
    Ok(())
}
