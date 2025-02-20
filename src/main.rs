mod network;

use network::Network;

fn main() {
    let mut nw = Network::new(vec![2,4,3]);
    println!("Hello, world!");
}
