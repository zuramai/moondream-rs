use moondream_rs::{moondream::moondream::Moondream, triton::{client::TritonGRPCClient, config::{self, TritonClientConfig}}};


#[tokio::main]
async fn main() {
    dotenvy::dotenv().ok();
    // init triton client
    let mut triton_client = TritonGRPCClient::new(
        TritonClientConfig::from_uri(std::env::var("TRITON_GRPC_URL").expect("TRITON_GRPC_URL must be set"))
            .expect("Invalid triton GRPC uri")
    ).await.unwrap();

    let models = triton_client.get_list_models().await.unwrap();
    println!("Models: {:?}", models);

    let tokenizer = tokenizers::Tokenizer::from_file("files/tokenizer.json").unwrap();
    let config = serde_json::from_reader(
        std::fs::File::open("files/config.json").unwrap()
    ).unwrap();
    let m = Moondream::new(triton_client, tokenizer, config);

    let image = image::open("test/images/ktp.jpeg").expect("Failed to load image");
    let object = "person";

    let result = m.detect(&image, object).await.unwrap();
    println!("Result: {:?}", result);
}
