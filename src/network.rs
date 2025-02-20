use nalgebra::{DMatrix, DVector};
use rand::prelude::*;

pub struct Network {
    pub weights: Vec<DMatrix<f64>>,
    pub biases: Vec<DMatrix<f64>>,
    num_layers: usize,
}

impl Network {

    pub fn new(sizes: Vec<usize>) -> Network {
        let mut rng = rand::rng();
        let mut network = Network {
            weights: Vec::with_capacity(sizes.len() - 1),
            biases: Vec::with_capacity(sizes.len() - 1),
            num_layers: sizes.len(),
        };

        for (i, y) in (&sizes[1..]).into_iter().enumerate() {
            network.biases[i] = DMatrix::from_fn(*y, 1, |_, _| rng.random::<f64>());
        }

        for (i, (x, y)) in std::iter::zip((&sizes[0..sizes.len()-1]).into_iter(), (&sizes[1..]).into_iter()).enumerate() {
            network.biases[i] = DMatrix::from_fn(*y, *x, |_, _| rng.random::<f64>());
        }
        return network;
    }

    fn sigmoid(z: DVector<f64>) -> DVector<f64> {
        DVector::from_fn(z.nrows(), |i, _| 1.0/(1.0+(-z[i].exp())))
    }

    pub fn feedforward(&self, a: DMatrix<f64>) -> DMatrix<f64> {
        
        let mut activation: DMatrix<f64> = a.clone();

        for (b, w) in std::iter::zip(self.biases.clone(), self.weights.clone()) { // FIXME: this clones are not good
            activation = self.feedforward(w * &activation + b);
        }
        activation
    }
}

