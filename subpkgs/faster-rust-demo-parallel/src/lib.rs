use once_cell::sync::Lazy;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use reqwest::Client;
use futures::future::try_join_all;

/// A single global Tokio runtime to avoid spinning one up per call.
static RUNTIME: Lazy<tokio::runtime::Runtime> = Lazy::new(|| {
    tokio::runtime::Runtime::new().expect("Failed to create Tokio runtime")
});

/// A single global Reqwest client, reused across requests.
static GLOBAL_CLIENT: Lazy<Client> = Lazy::new(|| {
    Client::new()
});

/// Download multiple URLs concurrently, discarding the response bodies.
#[pyfunction]
fn download_all(urls: Vec<String>) -> PyResult<()> {
    RUNTIME.block_on(async {
        // For each URL, create a future that GETs it and discards the body
        let tasks = urls.into_iter().map(|url| async move {
            let resp = GLOBAL_CLIENT
                .get(&url)
                .send()
                .await
                .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))?;

            // Download the body fully, then discard
            resp.bytes()
                .await
                .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))?;

            Ok::<(), pyo3::PyErr>(())
        });

        // Try to run all requests in parallel
        try_join_all(tasks).await?;

        Ok::<(), pyo3::PyErr>(())
    })?;

    // Return None to Python
    Ok(())
}

/// The module initialiser for "fastrparallel".
/// This must match the [lib] name = "fastrparallel" in Cargo.toml.
#[pymodule]
fn fastrparallel(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(download_all, m)?)?;
    Ok(())
}
