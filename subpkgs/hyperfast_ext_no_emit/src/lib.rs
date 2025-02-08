use hyper::{Client, Uri, body::to_bytes};
use hyper_tls::HttpsConnector;
use once_cell::sync::Lazy;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

// Build a single global Client at module load time
static GLOBAL_CLIENT: Lazy<Client<HttpsConnector<hyper::client::connect::HttpConnector>>> = Lazy::new(|| {
    let https = HttpsConnector::new();
    Client::builder().build::<_, hyper::Body>(https)
});

/// A simple blocking GET using Hyper and TLS,
/// reusing a single global Client, and discarding the response body.
#[pyfunction]
fn get(url: &str) -> PyResult<()> {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        // Parse the URL
        let uri = url
            .parse::<Uri>()
            .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))?;

        // Make the GET request using our global client
        let resp = GLOBAL_CLIENT
            .get(uri)
            .await
            .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))?;

        // Download the response bytes and discard them
        to_bytes(resp.into_body())
            .await
            .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))?;

        Ok::<(), pyo3::PyErr>(())
    })?;

    Ok(())
}

/// This is the module initialiser. The name here (`hyperfast5`) must match
/// the library name in your Cargo.toml ([lib] name="hyperfast5").
#[pymodule]
fn hyperfast5(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get, m)?)?;
    Ok(())
}
