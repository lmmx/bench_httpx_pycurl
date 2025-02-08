use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::wrap_pyfunction;
use hyper::Client;
use hyper::body::to_bytes;
use hyper::Uri;
use hyper_tls::HttpsConnector;

#[pyfunction]
fn get(py: Python, url: &str) -> PyResult<Py<PyBytes>> {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let data = rt.block_on(async {
        // Build an HTTPS connector
        let https = HttpsConnector::new();
        let client = Client::builder().build::<_, hyper::Body>(https);

        // Convert the &str URL into a Uri
        let uri = url
            .parse::<Uri>()
            .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))?;

        // Perform the GET request
        let resp = client
            .get(uri)
            .await
            .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))?;

        // Read the response body into bytes
        to_bytes(resp.into_body())
            .await
            .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))
    })?;

    // Return the raw bytes to Python
    Ok(PyBytes::new(py, &data).into())
}

#[pymodule]
fn hyperfast3(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get, m)?)?;
    Ok(())
}
