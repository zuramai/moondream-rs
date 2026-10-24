//! HTTP inference server (`server.py`).
//!
//! `POST /caption`, `/query`, `/detect`, `/point` with a JSON body holding a
//! base64 `image_url`; `stream: true` on caption/query returns server-sent events.

use std::{convert::Infallible, sync::Arc};

use axum::{
    body::Bytes,
    extract::{DefaultBodyLimit, State},
    http::{header, HeaderMap, HeaderValue, Method, StatusCode, Uri},
    middleware::map_response,
    response::{
        sse::{Event, Sse},
        Html, IntoResponse, Response,
    },
    Json, Router,
};
use base64::Engine;
use image::DynamicImage;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio_stream::{wrappers::ReceiverStream, StreamExt};

use crate::{CaptionLength, ImageInput, Result, Vlm};

const INDEX_HTML: &str = r#"
            <!DOCTYPE html>
            <html>
            <head>
                <title>Moondream Local Inference Server</title>
                <link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🌙</text></svg>">
                <style>
                    body { font-family: system-ui, sans-serif; max-width: 1200px; margin: 40px auto; padding: 0 20px; }
                    a { color: #0066cc; }
                </style>
            </head>
            <body>
                <h1>Moondream Local Inference Server is Running!</h1>
                <p>Visit the <a href="https://docs.moondream.ai">Moondream documentation</a> to learn more.</p>
            </body>
            </html>
            "#;

const UNEXPECTED: &str = "An unexpected error occurred.";

type Model = Arc<dyn Vlm>;

/// Builds the server router for `model`.
pub fn router(model: Model) -> Router {
    Router::new()
        .fallback(dispatch)
        .layer(DefaultBodyLimit::disable())
        .layer(map_response(|mut res: Response| async move {
            res.headers_mut()
                .insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, HeaderValue::from_static("*"));
            res
        }))
        .with_state(model)
}

/// Serves `model` on `host:port` until Ctrl-C.
///
/// Like Python's `HTTPServer`, an IPv4 address is preferred when `host`
/// resolves to several (e.g. `localhost` → `127.0.0.1` rather than `::1`).
pub async fn serve(model: Model, host: &str, port: u16) -> Result<()> {
    let addrs: Vec<_> = tokio::net::lookup_host((host, port)).await?.collect();
    let addr = addrs
        .iter()
        .find(|a| a.is_ipv4())
        .or(addrs.first())
        .copied()
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, format!("cannot resolve {host}")))?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    println!("Starting Moondream server on http://{host}:{port}");
    axum::serve(listener, router(model))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
            println!("\nShutting down server...");
        })
        .await?;
    Ok(())
}

fn error(status: StatusCode, message: &str) -> Response {
    (status, Json(json!({ "error": message }))).into_response()
}

/// Python truthiness of a JSON value.
fn truthy(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64().is_some_and(|f| f != 0.0),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
    }
}

/// Non-empty string field.
fn text_field<'a>(payload: &'a serde_json::Map<String, Value>, key: &str) -> Option<&'a str> {
    payload.get(key).and_then(Value::as_str).filter(|s| !s.is_empty())
}

/// Decodes a base64 image, optionally prefixed with a data URI header.
/// Like Python's `b64decode`, characters outside the alphabet are ignored.
fn decode_image(url: &str) -> Option<DynamicImage> {
    let data = url.split_once("base64,").map_or(url, |(_, d)| d);
    let cleaned: Vec<u8> = data
        .bytes()
        .filter(|b| b.is_ascii_alphanumeric() || matches!(b, b'+' | b'/' | b'='))
        .collect();
    let bytes = base64::engine::general_purpose::STANDARD.decode(cleaned).ok()?;
    image::load_from_memory(&bytes).ok()
}

async fn dispatch(
    State(model): State<Model>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    match method {
        Method::GET if uri.path_and_query().is_some_and(|p| p.as_str() == "/") => {
            Html(INDEX_HTML).into_response()
        }
        Method::GET => error(StatusCode::METHOD_NOT_ALLOWED, "Method not allowed"),
        Method::POST => post(model, uri.path(), &headers, &body).await,
        _ => error(StatusCode::NOT_IMPLEMENTED, "Unsupported method"),
    }
}

async fn post(model: Model, path: &str, headers: &HeaderMap, body: &[u8]) -> Response {
    let json_content = headers
        .get(header::CONTENT_TYPE)
        .is_some_and(|v| v.as_bytes() == b"application/json");
    if !json_content || body.is_empty() {
        return error(StatusCode::BAD_REQUEST, UNEXPECTED);
    }
    let Ok(Value::Object(payload)) = serde_json::from_slice::<Value>(body) else {
        return error(StatusCode::BAD_REQUEST, UNEXPECTED);
    };
    let Some(image) = text_field(&payload, "image_url").and_then(decode_image) else {
        tracing::error!("invalid or missing image_url");
        return error(StatusCode::BAD_REQUEST, UNEXPECTED);
    };
    let stream = truthy(payload.get("stream"));

    match path {
        "/caption" => {
            const FAILED: &str = "Caption generation failed.";
            let length = match payload.get("length") {
                None => Some(CaptionLength::Normal),
                Some(Value::String(s)) if s == "normal" => Some(CaptionLength::Normal),
                Some(Value::String(s)) if s == "short" => Some(CaptionLength::Short),
                _ => None,
            };
            let Some(length) = length else {
                return error(StatusCode::BAD_REQUEST, FAILED);
            };
            if stream {
                return sse(model, "An error occurred during caption generation.", move |m| {
                    m.caption_stream(ImageInput::Image(&image), length, None)
                });
            }
            blocking(FAILED, move || model.caption(ImageInput::Image(&image), length, None)).await
        }
        "/query" => {
            const FAILED: &str = "Query processing failed.";
            let Some(question) = text_field(&payload, "question").map(str::to_owned) else {
                return error(StatusCode::BAD_REQUEST, FAILED);
            };
            if stream {
                return sse(model, "An error occurred during query processing.", move |m| {
                    m.query_stream(ImageInput::Image(&image), &question, None)
                });
            }
            blocking(FAILED, move || model.query(ImageInput::Image(&image), &question, None)).await
        }
        "/detect" | "/point" => {
            let detect = path == "/detect";
            let failed = if detect { "Object detection failed." } else { "Object pointing failed." };
            let Some(object) = text_field(&payload, "object").map(str::to_owned) else {
                return error(StatusCode::BAD_REQUEST, failed);
            };
            if detect {
                blocking(failed, move || model.detect(ImageInput::Image(&image), &object)).await
            } else {
                blocking(failed, move || model.point(ImageInput::Image(&image), &object)).await
            }
        }
        _ => error(StatusCode::NOT_FOUND, "Not found"),
    }
}

/// Runs a model call off the async runtime and serializes its output.
async fn blocking<T, F>(failed: &'static str, f: F) -> Response
where
    T: serde::Serialize + Send + 'static,
    F: FnOnce() -> Result<T> + Send + 'static,
{
    match tokio::task::spawn_blocking(f).await {
        Ok(Ok(out)) => Json(out).into_response(),
        Ok(Err(e)) => {
            tracing::error!("{failed} {e}");
            error(StatusCode::BAD_REQUEST, failed)
        }
        Err(e) => {
            tracing::error!("{failed} task panicked: {e}");
            error(StatusCode::BAD_REQUEST, failed)
        }
    }
}

/// Streams chunks as `data: {"chunk": .., "completed": ..}` events. Errors,
/// eager or mid-stream, end the stream with `error_chunk`.
fn sse<F>(model: Model, error_chunk: &'static str, start: F) -> Response
where
    F: for<'a> FnOnce(&'a dyn Vlm) -> Result<crate::TokenStream<'a>> + Send + 'static,
{
    let (tx, rx) = mpsc::channel::<String>(32);
    tokio::task::spawn_blocking(move || {
        let send = |chunk: &str, completed: bool| {
            tx.blocking_send(json!({ "chunk": chunk, "completed": completed }).to_string())
                .is_ok()
        };
        let tokens = match start(model.as_ref()) {
            Ok(t) => t,
            Err(e) => {
                tracing::error!("streaming error: {e}");
                send(error_chunk, true);
                return;
            }
        };
        for item in tokens {
            match item {
                Ok(chunk) => {
                    if !send(&chunk, false) {
                        return; // client disconnected
                    }
                }
                Err(e) => {
                    tracing::error!("streaming error: {e}");
                    send(error_chunk, true);
                    return;
                }
            }
        }
        send("", true);
    });
    Sse::new(ReceiverStream::new(rx).map(|s| Ok::<_, Infallible>(Event::default().data(s))))
        .into_response()
}
