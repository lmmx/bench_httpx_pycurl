use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use hyper::{Client, Uri, body::to_bytes};
use hyper_tls::HttpsConnector;
use once_cell::sync::Lazy;
use futures::future::try_join_all;

/// A global Tokio runtime you can reuse for all calls rather than spinning up a new one.
static RUNTIME: Lazy<tokio::runtime::Runtime> = Lazy::new(|| {
    tokio::runtime::Runtime::new().expect("Failed to create Tokio runtime")
});

/// A single, global HTTPS client. Reused across all requests.
static GLOBAL_CLIENT: Lazy<Client<HttpsConnector<hyper::client::HttpConnector>>> = Lazy::new(|| {
    let https = HttpsConnector::new();
    Client::builder().build::<_, hyper::Body>(https)
});

/// Download multiple URLs in parallel, discarding the response bodies.
#[pyfunction]
fn download_all(urls: Vec<String>) -> PyResult<()> {
    // Use the global Tokio runtime
    RUNTIME.block_on(async {
        // Convert each URL into an async task
        let tasks = urls.into_iter().map(|url| async move {
            let uri = url
                .parse::<Uri>()
                .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))?;

            let resp = GLOBAL_CLIENT
                .get(uri)
                .await
                .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))?;

            to_bytes(resp.into_body())
                .await
                .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))?;

            Ok::<(), pyo3::PyErr>(())
        });

        // Run all tasks in parallel
        try_join_all(tasks).await?;

        Ok::<(), pyo3::PyErr>(())
    })?;

    Ok(())
}

/// The module initialiser. The name here (`hyperfastparallel`) must match
/// the lib name in Cargo.toml ([lib] name="hyperfastparallel").
#[pymodule]
fn hyperfastparallel(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(download_all, m)?)?;
    Ok(())
}
