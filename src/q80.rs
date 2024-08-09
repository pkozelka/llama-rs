#![allow(dead_code)]

use std::io::Read;

use byteorder::{LittleEndian, ReadBytesExt};

#[derive(Default, Clone)]
pub struct QuantizedTensor {
    /// quantized values
    q: Vec<i8>,
    /// scaling factors
    s: Vec<f32>,
}

static GS: usize = 64;

impl QuantizedTensor {
    /// initialize `n` x quantized tensor (with `size_each` elements), starting from memory pointed at *ptr
    /// (note: in C this was about pointing to mmapped memory, but here we use a Read trait)
    pub fn init_quantized_tensors<R: Read>(p: &mut R, n: usize, size_each: usize) -> anyhow::Result<Vec<QuantizedTensor>> {
        let mut res = vec![QuantizedTensor::default(); n];
        for _ in 0..n {
            let mut q = Vec::with_capacity(size_each);
            [0..q.capacity()].iter().try_for_each(|_| Ok::<(), anyhow::Error>(q.push(p.read_i8()?)))?;

            let mut s = Vec::with_capacity(size_each / GS);
            [0..s.capacity()].iter().try_for_each(|_| Ok::<(), anyhow::Error>(s.push(p.read_f32::<LittleEndian>()?)))?;

            res.push(QuantizedTensor { q, s });
        }
        Ok(res)
    }

    pub fn dequantize(&self) -> Vec<f32> {
        let mut x = Vec::with_capacity(self.q.len());
        for i in 0..x.capacity() {
            x.push(self.q[i] as f32 * self.s[i / GS]);
        }
        x
    }

    pub fn quantize(&mut self, x: &[f32]) {
        let num_groups = x.len() / GS;
        let q_max = 127.0;

        for group in 0..num_groups {
            // find the max absolute value in the current group
            let mut wmax = 0.0;
            for i in 0..GS {
                let val = x[group * GS + i].abs();
                if val > wmax {
                    wmax = val;
                }
            }

            // calculate and write the scaling factor
            let scale = wmax / q_max;
            self.s[group] = scale;

            // calculate and write the quantized values
            for i in 0..GS {
                let quant_value = x[group * GS + i] / scale; // scale
                let quantized = quant_value.round() as i8; // round and clamp
                self.q[group * GS + i] = quantized;
            }
        }
    }

    fn matmul(&self, xout: &mut [f32], w: &QuantizedTensor, n: usize, d: usize) {
        // W (d,n) @ x (n,) -> xout (d,)
        // by far the most amount of time is spent inside this little function
        // inputs to this function are both quantized

        for i in 0..d {
            let mut val = 0.0;
            let mut ival = 0;
            let ixn = i * n;

            // do the matmul in groups of GS
            for j in (0..=n - GS).step_by(GS) {
                for k in 0..GS {
                    ival += self.q[j + k] as i32 * w.q[ixn + j + k] as i32;
                }
                val += ival as f32 * w.s[(ixn + j) / GS] * self.s[j / GS];
                ival = 0;
            }

            xout[i] = val;
        }
    }
}