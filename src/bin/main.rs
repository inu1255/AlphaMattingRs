extern crate alpha_matting;

use alpha_matting::Shared;

fn main() {
    let mut shared = Shared::new("input.png");
    shared.run("trimap.png", "output.png");
}