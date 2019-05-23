extern crate image;
extern crate time;

mod lib;

use lib::shared;
use std::env;

fn main() {
    let mut args = Vec::new();
    for s in env::args() {
        args.push(s);
    }
    shared(&args[1], &args[2], &args[3]);
}
