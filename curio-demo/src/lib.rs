use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use hyper::Client;
use hyper_tls::HttpsConnector;
use hyper::http::uri::InvalidUri;

/// A simple blocking GET written in Rust via hyper.
/// It returns the response body as a Python string.
#[pyfunction]
fn get(url: &str) -> PyResult<String> {
    // Create a Tokio runtime on the fly, blocking for the GET request.
    let rt = tokio::runtime::Runtime::new().unwrap();
    let body = rt.block_on(async {
        let https = HttpsConnector::new();
        let client = Client::builder().build(https);
        let uri = url.parse::<hyper::Uri>().map_err(|e: InvalidUri| pyo3::exceptions::PyOSError::new_err(e.to_string()))?;
        let resp = client.get(uri).await.map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))?;
        let body_bytes = hyper::body::to_bytes(resp.into_body()).await.map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))?;
        let body = String::from_utf8(body_bytes.to_vec()).map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))?;
        Ok::<String, pyo3::PyErr>(body)
    })?;
    Ok(body)
}

/// This is the module initialiser. The name here (`hyperfast`) must match the
/// library name you declared in Cargo.toml (lib.name).
#[pymodule]
fn hyperfast(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get, m)?)?;
    Ok(())
}
