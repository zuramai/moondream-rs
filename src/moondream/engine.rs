use std::{collections::HashMap, path::{Path, PathBuf}};

use half::f16;
use ndarray::ArrayD;
use ort::{session::{Session, SessionInputValue}, value::{DynTensor, Tensor, Value}};
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

    pub fn run(&self, inputs: HashMap<&str, ArrayD<f16>>, output_key: &str) -> Result<ArrayD<f16>> {
        // can I map this input values to a whole different type? 
        let inputs: HashMap<&str, Tensor<f16>> = inputs.into_iter().map(|v| (v.0, Tensor::from_array(v.1).unwrap())).collect();
        let inference_output =self.session.run(inputs)?;
        let output = inference_output.get(output_key).ok_or(Error::Error("Output key does not exists".into()))?;
        let extracted = output.try_extract_tensor::<f16>()?;
        return Ok(extracted.into_dyn().to_owned());
    } 
}