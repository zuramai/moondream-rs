use std::{borrow::BorrowMut, collections::HashMap};

use ndarray::ArrayD;
use tonic::{transport::{channel::ClientTlsConfig, Channel}, Status};
use super::{config::TritonClientConfig, grpc::inference::{grpc_inference_service_client, ModelInferRequest}, types::{self, InferInputRequest, InferOutputTensor, InferRequest, InferResponse, InferResponseData, InferResponseOutput, TypedInferRequest}};
use crate::{triton::grpc::{self, inference::{grpc_inference_service_client::GrpcInferenceServiceClient, model_infer_request::{InferInputTensor, InferRequestedOutputTensor}, repository_index_response::ModelIndex, InferTensorContents}}, utils};
use crate::{error::{Error, Result}};


#[derive(Clone)]
pub struct TritonGRPCClient {
    client: GrpcInferenceServiceClient<Channel>,
    http_client: reqwest::Client,
    config: TritonClientConfig
}

pub enum InferInputs {
    F32(Vec<InferInputRequest<f32>>),
    I64(Vec<InferInputRequest<i64>>),
    F16(Vec<InferInputRequest<f32>>),
}

// Convert into GRPC supported input type
impl Into<Vec<InferInputTensor>> for InferInputs {
    fn into(self) -> Vec<InferInputTensor> {
        match self {
            InferInputs::F32(input) => input.into_iter().map(|i| i.into()).collect(),
            InferInputs::I64(input) => input.into_iter().map(|i| i.into()).collect(),
            InferInputs::F16(input) => input.into_iter().map(|i| i.into()).collect(),
        }
    }
}

impl TritonGRPCClient {
    pub async fn new(config: TritonClientConfig) -> Result<Self> {
        // Create channel with TLS
        let client = Self::create_client(&config).await?;
        let http_client = reqwest::Client::new();
        Ok(Self {
            client,
            http_client,
            config
        })
    }

    async fn create_client(config: &TritonClientConfig) -> Result<GrpcInferenceServiceClient<Channel>> {
        let mut tls = config.tls;
        let mut uri = config.grpc_uri.clone();
        match uri.scheme_str() {
            Some(scheme) => {
                let scheme = scheme.to_lowercase();
                if scheme == "https" {
                    tls = true;
                } else if scheme == "http" {
                    tls = false;
                } else {
                    return Err(Status::invalid_argument(format!(
                        "Invalid scheme `{scheme}`."
                    )))?;
                }
            }
            None => {
                uri = format!("http://{}", uri).parse().map_err(|_| Error::InvalidUri("Invalid Triton GRPC URI".to_string(), uri.to_string()))?;
            }
        };
        let endpoint = Channel::builder(uri)
            .timeout(config.timeout)
            .connect_timeout(config.connect_timeout)
            .keep_alive_while_idle(config.keep_alive_while_idle)
            .keep_alive_timeout(config.keep_alive_timeout);

        let endpoint = if tls {
            endpoint
                .tls_config(ClientTlsConfig::new().with_enabled_roots())
                .map_err(|e| Status::internal(format!("Failed to create TLS config: {}", e)))?
        } else {
            endpoint
        };

        let channel = endpoint
            .connect()
            .await
            .map_err(|e| Status::internal(format!("Failed to connect to {}: {}", config.grpc_uri, e)))?;
        let client = grpc_inference_service_client::GrpcInferenceServiceClient::new(channel);


        Ok(client)
    }


    pub async fn is_server_live(&mut self) -> Result<bool> {
        let request = tonic::Request::new(grpc::inference::ServerLiveRequest {});
        let result = self.client.server_live(request).await?.into_inner().live;
        Ok(result)
    }

    pub async fn is_model_ready(&mut self, model_name: String, version: String) -> Result<bool> {
        let request = tonic::Request::new(grpc::inference::ModelReadyRequest {
            name: model_name,
            version,
        });
        let result = self.client.model_ready(request).await?.into_inner().ready;
        Ok(result)
    }

    pub async fn get_list_models(&mut self) -> Result<Vec<ModelIndex>> {
        let request = tonic::Request::new(grpc::inference::RepositoryIndexRequest {
            repository_name: "".to_string(),
            ready: true,
        });
        let result = self.client.repository_index(request).await?.into_inner().models;
        Ok(result)
    }

    pub async fn infer(&self, model_name: String, model_version: String, mut inputs: InferInputs, outputs: Vec<&str>, with_http: bool) -> Result<HashMap<String, InferResponseOutput>> {
        if with_http {
            self.infer_http(model_name, model_version, inputs, outputs).await
        } else {
            self.infer_grpc(model_name, model_version, inputs, outputs).await
        }
    }
    pub async fn infer_http(&self, model_name: String, model_version: String, inputs: InferInputs, outputs: Vec<&str>) -> Result<HashMap<String, InferResponseOutput>> {
        let request_outputs: Vec<InferOutputTensor> = outputs.iter().map(|o| 
            InferOutputTensor {
                name: o.to_string(), 
            }).collect();

        let request: TypedInferRequest = match inputs {
            InferInputs::F32(inputs) => {
                TypedInferRequest::F32(InferRequest { outputs: request_outputs, inputs })
            },
            InferInputs::F16(inputs) => {
                TypedInferRequest::F32(InferRequest { outputs: request_outputs, inputs })
            },
            InferInputs::I64(inputs) => {
                TypedInferRequest::I64(InferRequest { outputs: request_outputs, inputs })
            },
        };

        let response = self.http_client.post(self.config.http_uri.to_string())
            .json(&request)
            .send()
            .await
            .map_err(|e| Error::TritonInferenceError(format!("Triton inference error: {:?}", e)))?;

        let response = response.json::<InferResponse>().await
            .map_err(|e| Error::TritonInferenceError(format!("Triton inference error: {:?}", e)))?;
        Ok(
            response.outputs.into_iter().map(|v| (v.name.clone(), v)).collect::<HashMap<String, InferResponseOutput>>()
        )
        
    }

    pub async fn infer_grpc(&self, model_name: String, model_version: String, inputs: InferInputs, outputs: Vec<&str>) -> Result<HashMap<String, InferResponseOutput>> {
        let outputs = outputs.iter().map(|o| 
            InferRequestedOutputTensor {
                name: o.to_string(), 
                ..Default::default()
            }).collect();
        let request = tonic::Request::new(grpc::inference::ModelInferRequest {
            model_name,
            model_version,
            inputs: inputs.into(),
            outputs,
            ..Default::default()
        });
        let result  = self.client.clone().borrow_mut()  
            .model_infer(request)
            .await
            .map_err(|e| Error::TritonInferenceError(format!("Triton inference error: {:?}", e)))?
            .into_inner();
        let mut outputs = HashMap::new();

        let raw_output_contents = result.raw_output_contents;

        for (i, output) in result.outputs.iter().enumerate() {
            let shape = output.shape.iter().map(|x| *x as usize).collect::<Vec<usize>>();
            // but this is None
            
            let raw_output_content = raw_output_contents.get(i).ok_or(Error::TritonInferenceError("No raw output content found".to_string()))?;
            let data = utils::bytes::bytes_to_array(raw_output_content, &shape)?;
            let new_output = InferResponseOutput {
                name: output.name.clone(),
                shape,
                datatype: output.datatype.clone(),
                parameters: HashMap::new(),
                data: InferResponseData::F32(data.into_raw_vec_and_offset().0),
            };
            outputs.insert(output.name.clone(), new_output);
        }


        Ok(outputs)
    }
}

impl InferInputs {
    pub fn new_i64() -> Self {
        Self::I64(Vec::new())
    }
    pub fn new_f16() -> Self {
        Self::F16(Vec::new())
    }
    pub fn new_f32() -> Self {
        Self::F32(Vec::new())
    }
    pub fn add_input(&mut self, name: &str, contents: ArrayD<f32>) {
        match self {
            InferInputs::F32(inputs) => {
                inputs.push(InferInputRequest::new(name.to_string(), contents.shape().to_vec(), "FP32".to_string(), contents.into_raw_vec_and_offset().0));
            },
            _ => panic!("Invalid input type"),
        }
    }
    
    pub fn add_input_i64(&mut self, name: &str, contents: ArrayD<i64>) {
        match self {
            InferInputs::I64(inputs) => {
                inputs.push(InferInputRequest::new(name.to_string(), contents.shape().to_vec(), "INT64".to_string(), contents.into_raw_vec_and_offset().0));
            },
            _ => panic!("Invalid input type"),
        }
    }
    pub fn add_input_fp16(mut self, name: &str, contents: ArrayD<f32>) -> Self {
        match self {
            InferInputs::F16(ref mut inputs) => {
                inputs.push(InferInputRequest::new(name.to_string(), contents.shape().to_vec(), "FP16".to_string(), contents.into_raw_vec_and_offset().0));
            },
            _ => panic!("Invalid input type"),
        }
        self
    }
}