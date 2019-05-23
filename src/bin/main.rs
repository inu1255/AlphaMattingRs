extern crate alpha_matting;

use alpha_matting::Shared;
use std::env;

fn main() {
    let mut args = Vec::new();
    for s in env::args() {
        args.push(s);
    }
    let mut shared = Shared::new(&args[1], &args[2]);
    shared.run(&args[3]);
}
