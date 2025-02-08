use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

/// A simple blocking GET written in Rust via reqwest.
/// It returns the response body as a Python string.
#[pyfunction]
fn get(url: &str) -> PyResult<String> {
    // Create a Tokio runtime on the fly, blocking for the GET request.
    let rt = tokio::runtime::Runtime::new().unwrap();
    let body = rt.block_on(async {
        let resp = reqwest::get(url).await.map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))?;
        let text = resp.text().await.map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))?;
        Ok::<String, pyo3::PyErr>(text)
    })?;
    Ok(body)
}

/// This is the module initialiser. The name here (`fastr`) must match the
/// library name you declared in Cargo.toml (lib.name).
#[pymodule]
fn fastr5(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get, m)?)?;
    Ok(())
}
