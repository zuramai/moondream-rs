//! `moondream` CLI (`cli.py`).

use std::{path::PathBuf, sync::Arc};

use clap::{CommandFactory, Parser, Subcommand};
use tracing_subscriber::{filter::Targets, layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Parser)]
#[command(name = "moondream", about = "Moondream CLI")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Start the Moondream server
    Serve {
        /// Path to the model file
        #[arg(long)]
        model: Option<PathBuf>,
        /// Host to bind to
        #[arg(long, default_value = "localhost")]
        host: String,
        /// Port to listen on
        #[arg(long, default_value_t = 3475)]
        port: u16,
    },
}

#[tokio::main]
async fn main() {
    // ONNX Runtime logs allocator details at INFO; the Python CLI shows errors only.
    let filter = Targets::new()
        .with_default(tracing::Level::INFO)
        .with_target("ort", tracing::Level::ERROR);
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(filter)
        .init();
    let cli = Cli::parse();
    let Some(Command::Serve { model, host, port }) = cli.command else {
        let _ = Cli::command().print_help();
        return;
    };
    let Some(model) = model else {
        Cli::command()
            .error(clap::error::ErrorKind::MissingRequiredArgument, "Model path is required")
            .exit();
    };
    let result = async {
        let vl = tokio::task::spawn_blocking(move || moondream_rs::vl(model))
            .await
            .map_err(|e| moondream_rs::Error::InvalidModel(format!("model loading panicked: {e}")))??;
        moondream_rs::server::serve(Arc::from(vl), &host, port).await
    }
    .await;
    if let Err(e) = result {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
