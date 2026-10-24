//! End-to-end tests for the candle backend. Needs `MOONDREAM_SAFETENSORS`
//! pointing at `vikhyatk/moondream2` weights (revision 2025-04-14):
//! `cargo test --release [--features metal] --test torch_e2e -- --ignored --nocapture`.

use std::{sync::LazyLock, time::Instant};

use moondream_rs::{CaptionLength, ImageInput, SamplingSettings, TorchVL, Vlm};

static MODEL: LazyLock<TorchVL> = LazyLock::new(|| {
    let path = std::env::var("MOONDREAM_SAFETENSORS").expect("set MOONDREAM_SAFETENSORS");
    let t = Instant::now();
    let m = TorchVL::new(path).expect("load weights");
    println!("load: {:?}", t.elapsed());
    m
});

fn image(name: &str) -> image::DynamicImage {
    image::open(name).expect("open demo image")
}

#[test]
#[ignore = "needs MOONDREAM_SAFETENSORS"]
fn caption_should_return_nonempty_text_for_demo1() {
    let img = image("demo-1.jpg");
    let t = Instant::now();
    let out = MODEL.caption((&img).into(), CaptionLength::Normal, None).unwrap();
    println!("caption: {:?} {:?}", t.elapsed(), out.caption);
    assert!(!out.caption.trim().is_empty());
}

#[test]
#[ignore = "needs MOONDREAM_SAFETENSORS"]
fn greedy_query_should_be_deterministic_across_calls() {
    let img = image("demo-1.jpg");
    let enc = MODEL.encode_image(ImageInput::Image(&img)).unwrap();
    let settings = Some(SamplingSettings {
        temperature: Some(0.0),
        ..Default::default()
    });
    let q = "What is the girl holding?";
    let a = MODEL.query((&enc).into(), q, settings).unwrap();
    let b = MODEL.query((&img).into(), q, settings).unwrap();
    println!("query: {:?}", a.answer);
    assert!(!a.answer.trim().is_empty());
    assert_eq!(a, b);
}

#[test]
#[ignore = "needs MOONDREAM_SAFETENSORS"]
fn detect_should_find_id_card_in_demo4() {
    let img = image("demo-4.jpeg");
    let t = Instant::now();
    let out = MODEL.detect((&img).into(), "an id card").unwrap();
    println!("detect: {:?} {:?}", t.elapsed(), out.objects);
    assert!(!out.objects.is_empty());
    for r in &out.objects {
        for v in [r.x_min, r.y_min, r.x_max, r.y_max] {
            assert!((0.0..=1.0).contains(&v), "{r:?}");
        }
    }
}

#[test]
#[ignore = "needs MOONDREAM_SAFETENSORS"]
fn point_should_locate_face_in_demo4() {
    let img = image("demo-4.jpeg");
    let out = MODEL.point((&img).into(), "the face").unwrap();
    println!("point: {:?}", out.points);
    assert!(!out.points.is_empty());
    // The photo sits in the right part of the card.
    assert!(out.points.iter().any(|p| p.x > 0.6 && p.y > 0.2 && p.y < 0.7), "{out:?}");
}
