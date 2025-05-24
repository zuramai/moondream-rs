use std::{collections::HashMap, fmt::Debug, path::{Path, PathBuf}};

use half::f16;
use ndarray::{ArrayD, ArrayViewD};
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
    pub fn run<T, O>(&self, inputs: HashMap<&str, ArrayViewD<T>>, output_key: Vec<&str>) -> Result<HashMap<&str, ArrayD<O>>>
    where
        T: Copy + IntoTensorElementType + Debug + PrimitiveTensorElementType + 'static,
        O: Copy + IntoTensorElementType + Debug + PrimitiveTensorElementType + 'static,
    {
        dbg!(&self.session);
        // can I map this input values to a whole different type? 
        let inputs: HashMap<&str, Tensor<T>> = inputs.into_iter().map(|v| (v.0, Tensor::from_array(v.1).unwrap())).collect();
        let inference_output =self.session.run(inputs)?;

        let mut outputs = HashMap::with_capacity(2);
        for key in output_key {
            let output = inference_output.get(key).ok_or(Error::Error(format!("Output key {} does not exists", key)))?;
            let extracted = output.try_extract_tensor::<O>()?;
            outputs.insert(key, extracted.to_owned().into_dyn());
        }
        return Ok(outputs);
    } 
}