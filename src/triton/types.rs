#[derive(Serialize)]
pub enum TypedInferRequest {
    F32(InferRequest<f32>),
    I64(InferRequest<i64>)
}

#[derive(Serialize)]
pub struct InferRequest<T> {
    pub inputs: Vec<InferInputRequest<T>>,
    pub outputs: Vec<InferOutputTensor>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct InferInputRequest<T> {
    pub name: String,
    pub shape: Vec<usize>,
    pub datatype: String,
    pub data: Vec<T>,
}

impl<T> Into<InferInputTensor> for InferInputRequest<T> {
    fn into(self) -> InferInputTensor {
        let contents = match self.datatype.as_str() {
            "FP32" => Some(InferTensorContents {
                fp32_contents: unsafe { std::mem::transmute(self.data) },
                ..Default::default()
            }),
            "INT64" => Some(InferTensorContents {
                int64_contents: unsafe { std::mem::transmute(self.data) },
                ..Default::default()
            }),
            _ => None,
        };

        InferInputTensor {
            name: self.name,
            shape: self.shape.iter().map(|x| *x as i64).collect(),
            datatype: self.datatype,
            parameters: HashMap::new(),
            contents,
            ..Default::default()
        }
    }
}

impl<T> InferInputRequest<T> {
    pub fn new(name: String, shape: Vec<usize>, datatype: String, data: Vec<T>) -> Self {
        Self { name, shape, datatype, data }
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct InferOutputTensor {
    pub name: String,
}
use ndarray::ArrayD;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::grpc::inference::{model_infer_request::InferInputTensor, InferTensorContents};

#[derive(Debug, Deserialize, Serialize)]
pub struct InferResponse {
    pub model_name: String,
    pub model_version: String,
    pub id: String,
    pub parameters: HashMap<String, String>,
    pub outputs: Vec<InferResponseOutput>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct InferResponseOutput {
    pub name: String,
    pub shape: Vec<usize>,
    pub datatype: String,
    pub parameters: HashMap<String, String>,
    pub data: InferResponseData
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum InferResponseData {
    String(Vec<String>),
    F32(Vec<f32>),
    I64(Vec<i64>),
}

impl InferResponseOutput {
    pub fn get_f32(self) -> ArrayD<f32> {
        if let InferResponseData::F32(data) = self.data {
            return ArrayD::from_shape_vec(self.shape, data).unwrap()
        }
        panic!("Expected F32 data");
    }

    pub fn get_i64(self) -> ArrayD<i64> {
        if let InferResponseData::I64(data) = self.data {
            return ArrayD::from_shape_vec(self.shape, data).unwrap()
        }
        panic!("Expected I64 data");
    }

}