use std::collections::HashMap;

use crate::bit_xor;

pub use super::{bits, gen, print_bits};

fn concat_arrays<T, const M: usize, const N: usize, const O: usize>(
    a: [T; M],
    b: [T; N],
) -> [T; O] {
    let mut result = std::mem::MaybeUninit::uninit();
    let dest = result.as_mut_ptr() as *mut T;
    unsafe {
        std::ptr::copy_nonoverlapping(a.as_ptr(), dest, M);
        std::ptr::copy_nonoverlapping(b.as_ptr(), dest.add(M), N);
        std::mem::forget(a);
        std::mem::forget(b);
        result.assume_init()
    }
}

const fn to_bit_array(num: usize) -> [bool; usize::BITS as usize] {
    let mut result = [false; usize::BITS as usize];

    let mut i = 0;
    while i < usize::BITS as usize {
        result[i] = (num >> i) & 1 > 0;
        i += 1;
    }

    result
}

const fn from_bit_array(bit_arr: &[bool]) -> usize {
    let mut result = 0;

    let mut i = 0;
    while i < bit_arr.len() {
        result += if bit_arr[i] { 1 } else { 0 };
        result <<= 1;
        i += 1;
    }

    result
}

const fn bit_and<const S: usize>(a: [bool; S], b: [bool; S]) -> [bool; S] {
    let mut result = [false; S];

    let mut i = 0;
    while i < S {
        result[i] = a[i] & b[i];
        i += 1;
    }

    result
}

pub trait Encoder<const K: usize, const N: usize> {
    fn encode_once(&mut self, bits: [bool; K]) -> [bool; N];

    fn decode_once(&mut self, bits: [bool; N]) -> [bool; K];

    fn encode(&mut self, bits: &[bool]) -> Vec<bool> {
        self._encode(bits)
    }

    fn _encode(&mut self, bits: &[bool]) -> Vec<bool> {
        let mut result = Vec::with_capacity(bits.len() * N / K);
        for chunk in bits.chunks_exact(K) {
            let bs: [bool; K] = unsafe { chunk.try_into().unwrap_unchecked() };

            let code = self.encode_once(bs);

            result.extend_from_slice(&code)
        }

        let rest = &bits[(bits.len() / K) * K..];
        if rest.len() > 0 {
            let bs: [bool; K] = [rest, &[false].repeat(K - rest.len())]
                .concat()
                .try_into()
                .unwrap();
            result.extend_from_slice(&bs);
        }

        result
    }

    fn decode(&mut self, bits: &[bool]) -> Vec<bool> {
        self._decode(bits)
    }

    fn _decode(&mut self, bits: &[bool]) -> Vec<bool> {
        let mut result = Vec::with_capacity(bits.len() * K / N);
        for chunk in bits.chunks_exact(N) {
            let bs: [bool; N] = unsafe { chunk.try_into().unwrap_unchecked() };

            let code = self.decode_once(bs);

            result.extend_from_slice(&code)
        }

        let rest = &bits[(bits.len() / N) * N..];
        if rest.len() > 0 {
            let bs: [bool; N] = [rest, &[false].repeat(N - rest.len())]
                .concat()
                .try_into()
                .unwrap();
            result.extend_from_slice(&bs);
        }

        result
    }

    fn reset(&mut self) {}
}

pub struct RepetitionEncoder<const N: usize>();

impl<const N: usize> Encoder<1, N> for RepetitionEncoder<N> {
    fn encode_once(&mut self, bits: [bool; 1]) -> [bool; N] {
        [bits[0]; N]
    }

    fn decode_once(&mut self, bits: [bool; N]) -> [bool; 1] {
        let sum: usize = bits.into_iter().map(|x| x as usize).sum();
        let bit = sum > N / 2;
        [bit]
    }
}

#[derive(Clone)]
pub struct ViterbiPath {
    path: Vec<bool>,
    distance: usize,
}

impl Default for ViterbiPath {
    fn default() -> Self {
        Self {
            path: Vec::new(),
            distance: usize::MAX,
        }
    }
}

impl std::fmt::Debug for ViterbiPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let path: String = self
            .path
            .iter()
            .map(|x| if *x { '1' } else { '0' })
            .collect();

        f.debug_struct("ViterbiPath")
            .field("path", &path)
            .field("distance", &self.distance)
            .finish()
    }
}

pub struct ConvolutionalEncoder<const K: usize, const N: usize, const L: usize, const G: usize>
where
    [(); K * (L - 1)]: Sized,
    [(); 1 << (K * (L - 1))]: Sized,
{
    state: [bool; K * (L - 1)],
    viterbi: [ViterbiPath; 1 << (K * (L - 1))],
}

impl<const K: usize, const N: usize, const L: usize, const G: usize>
    ConvolutionalEncoder<K, N, L, G>
where
    [(); K * (L - 1)]: Sized,
    [(); 1 << (K * (L - 1))]: Sized,
{
    fn encode_with_state(&self, bits: [bool; K * L]) -> [bool; N]
    where
        [(); K * L * N]: Sized,
        [(); K * L]: Sized,
    {
        let mut result = [false; N];

        let gen: [bool; K * L * N] = const { to_bit_array(G) }[..K * L * N].try_into().unwrap();

        for n in 0..N {
            let key: [bool; K * L] = gen[n * (K * L)..(n + 1) * (K * L)].try_into().unwrap();

            let dekeyed = bit_and(key, bits);

            result[n] = dekeyed.into_iter().reduce(|acc, x| acc ^ x).unwrap();
        }

        result
    }
}

impl<const K: usize, const N: usize, const L: usize, const G: usize> Encoder<K, N>
    for ConvolutionalEncoder<K, N, L, G>
where
    [(); K * (L - 1)]: Sized,
    [(); K * L * N]: Sized,
    [(); K * L]: Sized,
    [(); 1 << (K * (L - 1))]: Sized,
{
    fn encode_once(&mut self, bits: [bool; K]) -> [bool; N] {
        let bits: [bool; K * L] =
            concat_arrays::<_, K, { K * (L - 1) }, { K * L }>(bits, self.state);

        for i in (0..self.state.len()).rev() {
            if i < K {
                self.state[i] = bits[i];
            } else {
                let v = self.state[i - K];
                self.state[i] = v;
            }
        }

        self.encode_with_state(bits)
    }

    fn decode_once(&mut self, bits: [bool; N]) -> [bool; K] {
        let mut next_viterbi: [ViterbiPath; 1 << (K * (L - 1))] =
            vec![ViterbiPath::default(); 1 << (K * (L - 1))]
                .try_into()
                .unwrap();

        for (i, viterbi) in self.viterbi.iter().enumerate() {
            if viterbi.distance == usize::MAX {
                continue;
            }

            for in_int in 0..(1 << K) {
                let inp: [bool; K] = to_bit_array(in_int)[..K].try_into().unwrap();

                let next_state_with_inp = i | (in_int << (K * (L - 1)));
                let next_state = next_state_with_inp >> K;

                let out = self.encode_with_state(
                    to_bit_array(next_state_with_inp)[..K * L]
                        .try_into()
                        .unwrap(),
                );

                let dist: usize = bit_xor(out, bits)
                    .into_iter()
                    .map(|x| x as usize)
                    .sum::<usize>()
                    .saturating_add(viterbi.distance);

                println!("inp: {:#05b}", next_state_with_inp);
                println!("nex: {:#04b}", next_state);
                println!(
                    "out: {:?}",
                    out.iter()
                        .map(|&x| if x { '1' } else { '0' })
                        .collect::<String>()
                );
                println!("dst: {dist}");

                if next_viterbi[next_state].distance > dist {
                    let nv = &mut next_viterbi[next_state];
                    nv.distance = dist;
                    nv.path = viterbi.path.clone();
                    nv.path.extend_from_slice(&inp);

                    // dbg!(&nv.path);
                    // dbg!(&inp);
                }
            }
        }

        println!("{next_viterbi:#?}");
        self.viterbi = next_viterbi;

        [false; K]
    }

    fn decode(&mut self, bits: &[bool]) -> Vec<bool> {
        self._decode(bits);

        let vi = self.viterbi.iter().min_by_key(|x| x.distance).unwrap();
        assert_ne!(usize::MAX, vi.distance);

        vi.path.clone()
    }

    fn reset(&mut self) {
        self.state = [false; K * (L - 1)];
        self.viterbi = vec![ViterbiPath::default(); 1 << (K * (L - 1))]
            .try_into()
            .unwrap();
        self.viterbi[0].distance = 0;
    }
}

impl<const K: usize, const N: usize, const L: usize, const G: usize> Default
    for ConvolutionalEncoder<K, N, L, G>
where
    [(); K * (L - 1)]: Sized,
    [(); 1 << (K * (L - 1))]: Sized,
{
    fn default() -> Self {
        let mut viterbi: [ViterbiPath; 1 << (K * (L - 1))] =
            vec![ViterbiPath::default(); 1 << (K * (L - 1))]
                .try_into()
                .unwrap();
        viterbi[0].distance = 0;
        Self {
            state: [false; K * (L - 1)],
            viterbi,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::bit_xor;

    use super::{bits, gen, print_bits, ConvolutionalEncoder, Encoder, RepetitionEncoder};

    #[test]
    fn repetition_trivial() {
        let msg = bits![0 0 1];

        let mut encoder = RepetitionEncoder::<3>();

        let encoded = encoder.encode(&msg);

        encoder.reset();
        assert_eq!(bits![0 0 0 0 0 0 1 1 1].to_vec(), encoded);

        let decoded = encoder.decode(&encoded);

        assert_eq!(msg.to_vec(), decoded);
    }

    #[test]
    fn repetition_error() {
        let msg = bits![0 0 1];
        let err = bits![0 0 1 0 0 1 0 0 1];

        let mut encoder = RepetitionEncoder::<3>();

        let encoded = encoder.encode(&msg);

        encoder.reset();
        assert_eq!(bits![0 0 0 0 0 0 1 1 1].to_vec(), encoded);

        let decoded = encoder.decode(&bit_xor(encoded.try_into().expect("Invalid size"), err));

        assert_eq!(msg.to_vec(), decoded);
    }

    #[test]
    fn convolutional_trivial() {
        let msg = bits![0 0 1 0 1 1 0 1 0];

        const GEN: usize = gen!(5, 7);

        let mut encoder = ConvolutionalEncoder::<1, 2, 3, GEN>::default();

        let encoded = encoder.encode(&msg);

        assert_eq!(bits![0 0 0 0 1 1 0 1 0 0 1 0 1 0 0 0 0 1].to_vec(), encoded);

        encoder.reset();
        let decoded = encoder.decode(&encoded);

        assert_eq!(msg.to_vec(), decoded);
    }

    #[test]
    fn convolutional_error() {
        let msg = bits![0 0 1 0 1 1 0 1 0];
        let err = bits![0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0];

        const GEN: usize = gen!(5, 7);

        let mut encoder = ConvolutionalEncoder::<1, 2, 3, GEN>::default();

        let encoded = encoder.encode(&msg);

        assert_eq!(bits![0 0 0 0 1 1 0 1 0 0 1 0 1 0 0 0 0 1].to_vec(), encoded);

        encoder.reset();
        let decoded = encoder.decode(&bit_xor(encoded.try_into().expect("Invalid size"), err));

        assert_eq!(msg.to_vec(), decoded);
    }
}
