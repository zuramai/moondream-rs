//! End-to-end tests against the unpacked 0.5b ONNX model in `./model`.
//! Run with `cargo test --release --test onnx_e2e -- --ignored --nocapture`.

use std::{sync::LazyLock, time::Instant};

use moondream_rs::{CaptionLength, Error, ImageInput, OnnxVL, Vlm};

static MODEL: LazyLock<OnnxVL> = LazyLock::new(|| {
    let t = Instant::now();
    let m = OnnxVL::from_path("model").expect("load ./model");
    println!("load: {:?}", t.elapsed());
    m
});

fn model() -> &'static OnnxVL {
    &MODEL
}

fn image(name: &str) -> image::DynamicImage {
    image::open(name).expect("open demo image")
}

#[test]
#[ignore = "needs ./model"]
fn caption_should_return_nonempty_text_for_demo1() {
    let img = image("demo-1.jpg");
    let m = model();
    let t = Instant::now();
    let enc = m.encode_image(ImageInput::Image(&img)).unwrap();
    println!("encode_image: {:?}", t.elapsed());
    let t = Instant::now();
    let out = m.caption((&enc).into(), CaptionLength::Normal, None).unwrap();
    println!("caption (encoded): {:?} {:?}", t.elapsed(), out.caption);
    assert!(!out.caption.trim().is_empty());
}

#[test]
#[ignore = "needs ./model"]
fn reused_encoded_image_should_produce_identical_captions() {
    let img = image("demo-1.jpg");
    let m = model();
    let enc = m.encode_image(ImageInput::Image(&img)).unwrap();
    let a = m.caption((&enc).into(), CaptionLength::Short, None).unwrap();
    let b = m.caption((&enc).into(), CaptionLength::Short, None).unwrap();
    let c = m.caption((&img).into(), CaptionLength::Short, None).unwrap();
    assert_eq!(a, b);
    assert_eq!(a, c);
}

#[test]
#[ignore = "needs ./model"]
fn detect_should_find_id_card_in_demo4() {
    let img = image("demo-4.jpeg");
    let t = Instant::now();
    let out = model().detect((&img).into(), "an id card").unwrap();
    println!("detect: {:?} {:?}", t.elapsed(), out.objects);
    assert!(!out.objects.is_empty());
    for r in &out.objects {
        assert!(r.x_min < r.x_max && r.y_min < r.y_max, "{r:?}");
    }
}

#[test]
#[ignore = "needs ./model"]
fn point_should_error_unsupported_without_point_template() {
    // The 0.5b config has no `point` template, as in the Python client.
    let img = image("demo-4.jpeg");
    let err = model().point((&img).into(), "an id card").unwrap_err();
    assert!(matches!(&err, Error::Unsupported(m) if m == "Model does not support 'point'."), "{err}");
}

#[test]
#[ignore = "needs ./model"]
fn caption_long_should_error_unsupported() {
    let img = image("demo-1.jpg");
    let err = model()
        .caption((&img).into(), CaptionLength::Long, None)
        .unwrap_err();
    assert_eq!(err.to_string(), "Model does not support caption length 'long'.");
}

#[test]
#[ignore = "needs ./model"]
fn query_stream_should_concatenate_to_query_answer() {
    let img = image("demo-1.jpg");
    let m = model();
    let enc = m.encode_image((&img).into()).unwrap();
    let chunks: Vec<String> = m
        .query_stream((&enc).into(), "What is in the image?", None)
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    let full = m.query((&enc).into(), "What is in the image?", None).unwrap();
    println!("query: {:?}", full.answer);
    assert!(chunks.len() > 1);
    assert_eq!(chunks.concat(), full.answer);
}
