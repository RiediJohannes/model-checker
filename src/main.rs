
unsafe extern "C" {
    fn sqrt(x: f64) -> f64;


}

fn main() {
    println!("Hello, world!");

    unsafe {
        println!("{}", sqrt(2.0));
    }
}
