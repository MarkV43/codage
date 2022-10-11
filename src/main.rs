#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(inline_const)]
#![feature(const_for)]
#![feature(trace_macros)]

use encoder::*;

pub mod encoder;

#[macro_export]
macro_rules! print_bits {
    ($x: expr) => {{
        print!("{} {}:\n", $x.len(), stringify!($x));
        let y = $x.clone().into_iter();
        for x in y {
            print!("{}", if x { 1 } else { 0 });
        }
        println!();
    }};
}

#[macro_export]
macro_rules! bits {
    (0) => {false};
    (1) => {true};
    [$($x:tt)+] => {
        [$(bits!($x)),+]
    };
}

#[macro_export]
macro_rules! gen {
    ($acc:literal) => {
        $acc
    };
    ($l: literal, $($gen: literal),+) => {
        (gen!($($gen),+) << 3) | $l
    };
}

#[allow(dead_code)]
const fn bit_xor<const S: usize>(a: [bool; S], b: [bool; S]) -> [bool; S] {
    let mut result = [false; S];

    let mut i = 0;
    while i < S {
        result[i] = a[i] ^ b[i];
        i += 1;
    }

    result
}

fn main() {
    let msg = bits![0 0 1 0 1 1 0 1 0];

    const GEN: usize = gen!(5, 7);

    println!("Initializing encoder");

    let mut encoder = ConvolutionalEncoder::<1, 2, 3, GEN>::default();

    println!("Encoding");

    let encoded = encoder.encode(&msg);

    assert_eq!(bits![0 0 0 0 1 1 0 1 0 0 1 0 1 0 0 0 0 1].to_vec(), encoded);

    println!("Resetting");

    encoder.reset();

    println!("Decoding");
    let decoded = encoder.decode(&encoded);

    assert_eq!(msg.to_vec(), decoded);
}
