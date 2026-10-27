# moondream-rs

Rust port of the [Moondream](https://github.com/vikhyat/moondream) Python client:
the `Vlm` API (`encode_image`, `caption`, `query`, `detect`, `point`), two local
backends, and the `moondream serve` HTTP server.

| Backend | Model file | Runtime |
|---|---|---|
| `OnnxVL` | `.mf` / `.mf.gz` bundle, or an unpacked model directory | ONNX Runtime (`ort`) |
| `TorchVL` | `.safetensors` (`vikhyatk/moondream2`, revision `2025-04-14`) | candle (CPU f32, Metal/CUDA f16) |

`moondream_rs::vl(path)` picks the backend from the path.

## Install

```toml
[dependencies]
moondream-rs = "0.1"
```

Server binary: `cargo install moondream-rs` (installs `moondream`).

On macOS, the prebuilt ONNX Runtime needs clang's compiler-rt. Add this to your
project's `.cargo/config.toml` (or pass it via `RUSTFLAGS`):

```toml
[target.'cfg(target_os = "macos")']
rustflags = ["-C", "link-arg=-fapple-link-rtlib"]
```

## Library

```rust
use moondream_rs::{vl, CaptionLength, ImageInput};

let model = vl("model")?; // directory, .mf, .mf.gz, or .safetensors
let image = image::open("demo-1.jpg")?;
let encoded = model.encode_image(ImageInput::Image(&image))?; // reuse across calls
println!("{}", model.caption((&encoded).into(), CaptionLength::Normal, None)?.caption);
for chunk in model.query_stream((&encoded).into(), "What is in the image?", None)? {
    print!("{}", chunk?);
}
println!("{:?}", model.detect((&image).into(), "a face")?.objects);
```

The API is blocking; `*_stream` methods validate and encode eagerly, then
generate lazily. ONNX decoding is greedy (like the Python client); `TorchVL`
honors `SamplingSettings::temperature` / `top_p` (defaults 0.5 / 0.3, 768 tokens).

## Server

```sh
cargo run --release --bin moondream -- serve --model ./model [--host localhost] [--port 3475]
```

Same endpoints, request validation, and error strings as `moondream serve` in
Python: `POST /caption` (`length`: `normal`|`short`, `stream`), `/query`
(`question`, `stream`), `/detect` and `/point` (`object`), each with a base64
`image_url`. Streaming responses are server-sent events
`data: {"chunk": "...", "completed": false}`. Unknown POST paths return 404.

## Features

| Cargo feature | Effect |
|---|---|
| `metal` | candle on Apple GPUs (`TorchVL`) |
| `cuda` | CUDA for both backends |
| `coreml` | CoreML execution provider for `OnnxVL` |
| `accelerate` | Apple Accelerate BLAS for candle CPU |

## Differences from the Python client

- `OnnxVL` assembles local crop features from indices `1..` (the global patch is
  index 0); `onnx_vl.py` starts at 0, dropping the last crop. This follows the
  reference torch model and gives tighter boxes.
- `OnnxVL` also accepts an unpacked model directory.
- CloudVL is not ported.

## Tests

```sh
cargo test                                                     # unit tests
cargo test --release --test onnx_e2e -- --ignored              # needs ./model
MOONDREAM_SAFETENSORS=model-torch/model.safetensors \
  cargo test --release --features metal --test torch_e2e -- --ignored
```

## License

Apache-2.0, same as upstream [Moondream](https://github.com/vikhyat/moondream). See [LICENSE](LICENSE).
