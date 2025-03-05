mod network;
mod mnist;

use network::Network;

fn main() {
    let nw = Network::new(vec![784,30,10]);

    let training_data = match mnist::load_training_data() {
        Ok(t) => t,
        Err(_) => panic!("There was an error reading the files."),
    };

    nw.sgd(training_data, 30, 10, 3.0);

}
