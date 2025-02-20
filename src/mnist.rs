use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

use nalgebra::{SMatrix};

fn read_data(path: &str) -> Result<Vec<u8>, u8> {
    let path = Path::new(path);
    let display = path.display();

    let mut file = match File::open(&path) {
        Err(why) => return Err(0),
        Ok(file) => file,
    };

    let mut data: Vec<u8> = vec![];

    let result = file.read_to_end(&mut data);

    match result {
        Ok(n) => Ok(data),
        Err(e) => Err(0),
    }
}

fn extract_labels(data: Vec<u8>) -> Result<Vec<u8>, u8> {
    let (magic, rest) = data.split_at(4);
    assert!(u32::from_be_bytes(magic.try_into().unwrap()) == 0x00000801);
    let (_size, bytes) = rest.split_at(4);

    Ok(bytes.to_vec())
}

fn extract_images(data: Vec<u8>) -> Result<Vec<SMatrix<f64, {28}, {28}>>, u8> {
    let (magic, rest) = data.split_at(4);
    assert!(u32::from_be_bytes(magic.try_into().unwrap()) == 0x00000803);

    let (size, mut bytes) = rest.split_at(4);
    let n_images = u32::from_be_bytes(size.try_into().unwrap());
    assert!(n_images == 60_000);

    let (msize, matrixrest) = bytes.split_at(8); // Two 4 bytes for the image size 28x28
    let (mrows, mcols) = msize.split_at(4);
    assert!(u32::from_be_bytes(mrows.try_into().unwrap()) == 28);
    assert!(u32::from_be_bytes(mcols.try_into().unwrap()) == 28);

    let mut result = vec![];

    bytes = matrixrest;
    for enu in 0..n_images {
        let (matrix, rest) = bytes.split_at(784); // read 28*28 bytes into matrix
        bytes = rest;

        result.push(SMatrix::from_fn(|i, j| matrix[j*28+i] as f64));
    }

    Ok(result)
}

pub fn load_training_data() -> Result<Vec<(SMatrix<f64, {28}, {28}>, u8)>, u8> {

    let imagedata = match read_data("assets/train-images-idx3-ubyte") {
        Ok(data) => data,
        Err(n) => return Err(n),
    };

    let labeldata = match read_data("assets/train-labels-idx1-ubyte") {
        Ok(data) => data,
        Err(n) => return Err(n),
    };

    let labels = match extract_labels(labeldata) {
        Ok(data) => data,
        Err(n) => return Err(n),
    };

    let images = match extract_images(imagedata) {
        Ok(data) => data,
        Err(n) => return Err(n),
    };

    let mut cosa = vec![];

   // FIXME: I dont have enough memory for this, so I'll curb the data for i in 0..images.len() {
    for i in 0..10_000 {
        cosa.push((images[i], labels[i]));
    }
    Ok(cosa)
}
