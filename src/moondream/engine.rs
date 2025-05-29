use std::{collections::HashMap, fmt::Debug, path::{Path, PathBuf}};

use half::f16;
use ndarray::{ArrayD, ArrayViewD};
use ort::{execution_providers::{CPUExecutionProvider, CoreMLExecutionProvider}, session::{builder::GraphOptimizationLevel, Session, SessionInputValue}, sys::OrtSessionOptions, tensor::{IntoTensorElementType, PrimitiveTensorElementType}, value::{DynTensor, Tensor, TensorRef, Value}};
use crate::error::{Error, Result};

pub struct Engine {
    session: Session
}

impl Engine {
    pub fn new(model_path: PathBuf) -> Result<Self> {
        let builder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers([
                // CoreMLExecutionProvider::default().build(),
                CPUExecutionProvider::default().build()
            ])?;
        
        let session = builder.commit_from_file(model_path)?;
        Ok(Self {
            session
        })
    }
    pub fn run<T, O>(&mut self, inputs: HashMap<&str, ArrayViewD<T>>, output_key: Vec<&str>) -> Result<HashMap<String, ArrayD<O>>>
    where
        T: Copy + IntoTensorElementType + Debug + PrimitiveTensorElementType + 'static,
        O: Copy + IntoTensorElementType + Debug + PrimitiveTensorElementType + 'static,
    {
        
        // Pre-allocate with known capacity to avoid reallocations
        let mut ort_inputs = HashMap::with_capacity(inputs.len());
        for (key, array_view) in inputs {
            ort_inputs.insert(key, TensorRef::from_array_view(array_view)?);
        }
        
        let inference_output = self.session.run(ort_inputs)?;
        
        // Pre-allocate output HashMap with known capacity
        let mut outputs = HashMap::with_capacity(output_key.len());
        for key in output_key {
            let output = inference_output.get(key).ok_or(Error::Error(format!("Output key {} does not exists", key)))?;
            let extracted = output.try_extract_array::<O>()?;
            outputs.insert(key.to_string(), extracted.to_owned());
        }
        
        Ok(outputs)
    } 
}