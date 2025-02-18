use ndarray::{ArrayD, IxDyn};

use crate::error::{Error, Result};

pub fn bytes_to_array(bytes: &Vec<u8>, shape: &[usize]) -> Result<ArrayD<f32>> {
    // Check if the total size matches
    let total_elements: usize = shape.iter().product();
    if bytes.len() != total_elements * 4 {
        return Err(Error::new("Byte length doesn't match the expected shape".into()));
    }

    // Convert bytes to f32 values
    let f32_vec: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| {
            let arr: [u8; 4] = chunk.try_into().unwrap();
            f32::from_le_bytes(arr)
        })
        .collect();

        // Create ArrayD with the specified shape
        Ok(ArrayD::from_shape_vec(IxDyn(shape), f32_vec).map_err(|e| Error::from_string(e.to_string()))?)
}


pub fn array_to_bytes(array: &ArrayD<f32>) -> Vec<u8> {
    array
        .iter()
        .flat_map(|&x| x.to_le_bytes().to_vec())
        .collect()
}