use std::{collections::HashMap, path::{Path, PathBuf}};

use half::f16;
use ndarray::ArrayD;
use ort::session::Session;
use crate::error::{Error, Result};

pub struct Engine {
    session: Session
}

impl Engine {
    pub fn new(model_path: PathBuf) -> Result<Self> {
        let builder = Session::builder()?
            .commit_from_file(model_path)?;
        
        Ok(Self {
            session: builder
        })
    }

    pub fn run(&self, inputs: ArrayD<f16>) -> Result<ArrayD<f32>> {
        dbg!(&self.session.inputs);
        dbg!(&self.session.outputs);

        let inference_output =self.session.run(ort::inputs![inputs]?)?;
        let output = inference_output.get("output").ok_or(Error::Error("Output key does not exists".into()))?;
        let extracted = output.try_extract_tensor::<f32>()?;
        return Ok(extracted.into_dyn().to_owned());
    } 
}