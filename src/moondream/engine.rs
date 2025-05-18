use std::path::{Path, PathBuf};

use ort::session::Session;
use crate::error::Result;

pub struct Engine {
    pub session: Session
}

impl Engine {
    pub fn new(model_path: PathBuf) -> Result<Self> {
        let builder = Session::builder()?
            .commit_from_file(model_path)?;
        
        Ok(Self {
            session: builder
        })
    }

    pub fn run(){

    } 
}