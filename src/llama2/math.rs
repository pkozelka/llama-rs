
pub fn softmax(array: &mut [f32]) {
    // find max value (for numerical stability)
    let mut max_val = array[0];
    for i in 1..array.len() {
        if array[i] > max_val {
            max_val = array[i];
        }
    }
    // exp and sum
    let mut sum = 0.0;
    for i in 0..array.len() {
        array[i] = (array[i] - max_val).exp();
        sum += array[i];
    }
    // normalize
    for i in 0..array.len() {
        array[i] /= sum;
    }
}


pub fn rmsnorm(o: &mut [f32], x: &[f32], weight: &[f32]) {
    // log::debug!("rmsnorm(o.len={}, x.len={}, weight.len={})", o.len(), x.len(), weight.len());
    // calculate sum of squares
    let mut ss = 0.0;
    for j in 0..x.len() {
        ss += x[j] * x[j];
    }
    ss /= x.len() as f32;
    ss += 1e-5;
    ss = 1.0 / ss.sqrt();
    // normalize and scale
    for j in 0..x.len() {
        o[j] = weight[j] * (ss * x[j]);
    }
}
pub fn rmsnorm_inplace(x: &mut [f32], weight: &[f32]) {
    // calculate sum of squares
    let mut ss = 0.0;
    for j in 0..x.len() {
        ss += x[j] * x[j];
    }
    ss /= x.len() as f32;
    ss += 1e-5;
    ss = 1.0 / ss.sqrt();
    // normalize and scale
    for j in 0..x.len() {
        x[j] = weight[j] * (ss * x[j]);
    }
}


/// Params:
/// * `xout`: pointer to array(d); can be part of a larger one
/// * `x`: array(n)
/// * `w`: array(d)(n)
///
/// W (d,n) @ x (n,) -> xout (d,)
pub fn matmul(xout: &mut [f32], x: &[f32], w: &[f32]) {
    // log::debug!("matmul(xout.len={}, x.len={}, w.len={}, n={}, d={})", xout.len(), x.len(), w.len(), n, d);
    // by far the most amount of time is spent inside this little function
    // #pragma omp parallel for private(i)
    // println!("matmul: xout.len={}, x.len={}, w.len={}, n={_n}, d={d}", xout.len(), x.len(), w.len());
    let mut wp = w.iter();
    for xouti in xout {
        *xouti = x.iter()
            .zip(wp.by_ref())
            .map(|(&xj, &wj)| xj * wj)
            .sum();
    }
}
