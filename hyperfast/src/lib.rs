use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::wrap_pyfunction;
use hyper::Client;
use hyper::body::to_bytes;
use hyper::Uri;

/// A simple blocking GET written in Rust via Hyper.
/// It returns the response body as Python bytes.
#[pyfunction]
fn get(py: Python, url: &str) -> PyResult<Py<PyBytes>> {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let data = rt.block_on(async {
        let client = Client::new();
        let uri = url.parse::<Uri>().map_err(...)?;
        let resp = client.get(uri).await.map_err(...)?;
        to_bytes(resp.into_body()).await.map_err(...)
    })?;
    Ok(PyBytes::new(py, &data).into())
}


/// This is the module initialiser. The name here (`hyperfast`) must match the
/// library name you declared in Cargo.toml (lib.name).
#[pymodule]
fn hyperfast(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get, m)?)?;
    Ok(())
}
