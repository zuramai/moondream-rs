use std::{collections::HashMap, fmt::Debug, path::{Path, PathBuf}};

use half::f16;
use ndarray::ArrayD;
use ort::{session::{Session, SessionInputValue}, tensor::{IntoTensorElementType, PrimitiveTensorElementType}, value::{DynTensor, Tensor, Value}};
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
    pub fn run<T, O>(&self, inputs: HashMap<&str, ArrayD<T>>, output_key: &str) -> Result<ArrayD<O>>
    where
        T: Copy + IntoTensorElementType + Debug + PrimitiveTensorElementType + 'static,
        O: Copy + IntoTensorElementType + Debug + PrimitiveTensorElementType + 'static,
    {
        dbg!(&self.session);
        // can I map this input values to a whole different type? 
        let inputs: HashMap<&str, Tensor<T>> = inputs.into_iter().map(|v| (v.0, Tensor::from_array(v.1).unwrap())).collect();
        let inference_output =self.session.run(inputs)?;
        let output = inference_output.get(output_key).ok_or(Error::Error("Output key does not exists".into()))?;
        let extracted = output.try_extract_tensor::<O>()?;
        return Ok(extracted.into_dyn().to_owned());
    } 
}