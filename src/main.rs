mod network;
mod mnist;

use network::Network;

fn main() {
   // let mut nw = Network::new(vec![2,4,3]);

    let training_data = match mnist::load_training_data() {
        Ok(t) => t,
        Err(_) => panic!("There was an error reading the files."),
    };

}
