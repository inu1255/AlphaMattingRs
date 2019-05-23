extern crate image;
extern crate time;

mod lib;

use lib::Matting;

fn main() {
    let mut matting = Matting::new("a.png", "b.png");
    let start = time::now();
    matting.run();
    let end = time::now();
    println!("expand_known: {:?}", (end - start).num_milliseconds());
    matting.ori_state.save("1.png").unwrap();
    matting.tri_state.save("2.png").unwrap();
}
