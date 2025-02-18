use crate::error::{Error, Result};
use std::str::FromStr;
use std::time::Duration;
use tonic::transport::Uri;

#[derive(Clone, Debug)]
pub struct TritonClientConfig {
    /// Triton server URI to connect to
    pub grpc_uri: Uri,

    /// Triton server HTTP URI to connect to
    pub http_uri: Uri,

    /// Timeout for API requests
    pub timeout: Duration,

    /// Secure connection
    pub tls: bool,

    /// Timeout for connecting to the Triton server
    pub connect_timeout: Duration,

    /// Whether to keep idle connections active
    pub keep_alive_while_idle: bool,

    /// Duration an idle connection remains open before closing
    pub keep_alive_timeout: Duration,

}

impl TritonClientConfig {
    pub fn from_uri<S: AsRef<str>>(grpc_uri: S, http_uri: S) -> Result<Self> {
        Ok(Self {
            grpc_uri: Uri::from_str(grpc_uri.as_ref()).map_err(|e| Error::InvalidUri("Invalid Triton GRPC URI".to_string(), grpc_uri.as_ref().to_string()))?,
            http_uri: Uri::from_str(http_uri.as_ref()).map_err(|e| Error::InvalidUri("Invalid Triton HTTP URI".to_string(), http_uri.as_ref().to_string()))?,
            ..Self::default()
        })
    }

    pub fn with_tls(mut self, tls: bool) -> Self {
        self.tls = tls;
        self
    }

    pub fn with_connection_timeout<T: AsTimeout>(mut self, timeout: T) -> Self {
        self.connect_timeout = AsTimeout::timeout(timeout);
        self
    }

    pub fn with_keep_alive_while_idle(mut self, keep_alive_while_idle: bool) -> Self {
        self.keep_alive_while_idle = keep_alive_while_idle;
        self
    }

    pub fn with_keep_alive_timeout<T: AsTimeout>(mut self, keep_alive_timeout: T) -> Self {
        self.keep_alive_timeout = keep_alive_timeout.timeout();
        self
    }
}

impl Default for TritonClientConfig {
    fn default() -> Self {
        Self {
            grpc_uri: Uri::from_str("localhost:8001").unwrap(),
            http_uri: Uri::from_str("localhost:8000").unwrap(),
            timeout: Duration::from_secs(30),
            tls: false,
            connect_timeout: Duration::from_secs(5),
            keep_alive_while_idle: true,
            keep_alive_timeout: Duration::from_secs(20),
        }
    }
}

pub trait AsTimeout {
    fn timeout(self) -> Duration;
}

impl AsTimeout for Duration {
    fn timeout(self) -> Duration {
        self
    }
}

impl AsTimeout for u64 {
    fn timeout(self) -> Duration {
        Duration::from_secs(self)
    }
}