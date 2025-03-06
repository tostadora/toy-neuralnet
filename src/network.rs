use rand::prelude::*;
use rand::rng;
use ndarray::Array2;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

pub struct Network {
    pub weights: Vec<Array2<f64>>,
    pub biases: Vec<Array2<f64>>,
    num_layers: usize,
}

impl Network {

    pub fn new(sizes: Vec<usize>) -> Network {
        let mut network = Network {
            weights: Vec::with_capacity(sizes.len() - 1),
            biases: Vec::with_capacity(sizes.len() - 1),
            num_layers: sizes.len(),
        };
        for y in (&sizes[1..]).into_iter() {
            network.biases.push(Array2::random((*y, 1), Uniform::new(0., 1.)));
        }

        for (x, y) in std::iter::zip((&sizes[0..sizes.len()-1]).into_iter(), (&sizes[1..]).into_iter()) {
            network.weights.push(Array2::random((*y, *x), Uniform::new(0., 1.)));
        }
        return network;
    }

    fn sigmoid(z: &Array2<f64>) -> Array2<f64> {
        z.map(|i| 1.0/(1.0+(-i.exp())))
    }

    fn sigmoid_prime(z: &Array2<f64>) -> Array2<f64> {
        Self::sigmoid(z) * (1.0 - Self::sigmoid(z))
    }

    fn cost_derivative (&self, output_activations: &Array2<f64>, y: u8) -> Array2<f64> {
        output_activations - y as f64
    }

    pub fn feedforward(&self, a: Array2<f64>) -> Array2<f64> {
        
        let mut activation: Array2<f64> = a.clone();

        for (b, w) in std::iter::zip(self.biases.clone(), self.weights.clone()) { // FIXME: this clones are not good
            activation = Self::sigmoid(&(w * &activation + b));
        }
        activation
    }


    pub fn sgd(&mut self, mut tr_data: Vec<(Array2<f64>, u8)>, epochs: usize, mini_batch_size: usize, eta: f64) {
        let _n = tr_data.len();

        for _ in 0..epochs {
            tr_data.shuffle(&mut rng());

            let mini_batches = (0..mini_batch_size).map(|offset| {
                            tr_data.iter()
                            .cloned()
                            .skip(offset)
                            .step_by(mini_batch_size)
                            .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
            for mini_batch in mini_batches {
                self.update_mini_batch(mini_batch, eta);
            }

            // TODO: testing the data
        }

        println!("{:?}", self.weights);
        println!("{:?}", self.biases);

    }
    

    fn update_mini_batch(&mut self, mini_batch: Vec<(Array2<f64>, u8)>, eta: f64) {
        let mut nabla_b: Vec<Array2<f64>> = Vec::with_capacity(self.biases.len());

        for b in &self.biases {
            nabla_b.push(Array2::zeros((b.shape()[0], b.shape()[1])));
        }

        let mut nabla_w: Vec<Array2<f64>> = Vec::with_capacity(self.weights.len());

        for w in &self.weights {
            nabla_w.push(Array2::zeros((w.shape()[0], w.shape()[1])));
        }

        for (image, solution) in mini_batch.clone() { // FIXME: bad clone
            let deltas = self.backprop(image, solution);
            for (i, (delta_nabla_b, delta_nabla_w)) in deltas.into_iter().enumerate() {
                nabla_w[i] = &nabla_w[i] + delta_nabla_w;
                nabla_b[i] = &nabla_b[i] + delta_nabla_b;
            }
        }

        for (w, nw) in std::iter::zip(&mut self.weights, nabla_w) {
            *w = &*w - (eta / mini_batch.len() as f64) * nw;
        }

        for (b, nb) in std::iter::zip(&mut self.biases, nabla_b) {
            *b = &*b - (eta / mini_batch.len() as f64) * nb;
        }
    }
    

    fn backprop(&self, image: Array2<f64>, solution: u8) -> Vec<(Array2<f64>, Array2<f64>)> {
        let mut nabla_b: Vec<Array2<f64>> = Vec::with_capacity(self.biases.len());

        for b in &self.biases {
            nabla_b.push(Array2::zeros((b.shape()[0], b.shape()[1])));
        }

        let mut nabla_w: Vec<Array2<f64>> = Vec::with_capacity(self.weights.len());

        for w in &self.weights {
            nabla_w.push(Array2::zeros((w.shape()[0], w.shape()[1])));
        }

        let mut activations = Vec::<Array2<f64>>::with_capacity(self.weights.len());
        let mut activation = image.clone();
        activations.push(activation);
        let mut zs: Vec<Array2<f64>> = vec![];

        for (b, w) in std::iter::zip(&self.biases, &self.weights) {
            // println!("w: {:?}, a: {:?}, b: {:?}", w.shape(), &activations[activations.len() - 1].shape(), b.shape());
            let mut z = Array2::zeros((b.shape()[0], b.shape()[1]));
            ndarray::linalg::general_mat_mul(1.0, &w, &activations[activations.len() - 1], 1.0, &mut z);
            z = z + b;
            zs.push(z);
            activation = Self::sigmoid(&zs[zs.len() - 1]);
            activations.push(activation);
        }

        let mut delta = self.cost_derivative(&activations[activations.len()-1], solution) * Self::sigmoid_prime(&zs[zs.len()-1]);
        // The vectors for the biases and the weights have an element less than the number of layers
        // and start indexing at 0, so the last element of the vector is the number of layers minus
        // 2.
        nabla_b[self.num_layers - 2] = delta.clone();
        ndarray::linalg::general_mat_mul(1.0, &delta, &activations[activations.len()-2].t(), 1.0, &mut nabla_w[self.num_layers - 2]);
        
        for l in (self.num_layers - 2)..0 {
            let sp = Self::sigmoid_prime(&zs[l]);
            delta = &(self.weights[l+1]) * &delta * sp; // FIXME: transpose
            nabla_b[l] = delta.clone();
            nabla_w[l] = &nabla_b[l]  * &activations[l+1]; // FIXME: transpose
        }

        std::iter::zip(nabla_b, nabla_w).collect() // Create a Vec of tuples

    }
}

