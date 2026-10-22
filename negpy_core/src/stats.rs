use rayon::prelude::*;

const BUCKETS: usize = 1 << 16;
const CHUNK: usize = 1 << 16;
/// Below this size one `select_nth_unstable` beats the histogram passes.
const DIRECT_MAX: usize = 1 << 15;

/// Total-order key of an f32. Every NaN maps to the top, where `np.partition` puts it.
fn order_key(v: f32) -> u32 {
    if v.is_nan() {
        return u32::MAX;
    }
    let bits = v.to_bits();
    if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits | 0x8000_0000
    }
}

fn from_key(key: u32) -> f32 {
    if key & 0x8000_0000 != 0 {
        f32::from_bits(key & 0x7fff_ffff)
    } else {
        f32::from_bits(!key)
    }
}

/// The lower rank of `q` and its interpolation weight, in numpy's float64 arithmetic.
fn virtual_rank(n: usize, q: f64) -> (usize, f64) {
    let virtual_index = (n - 1) as f64 * (q / 100.0);
    let lo = (virtual_index.floor().max(0.0) as usize).min(n - 1);
    (lo, virtual_index - lo as f64)
}

/// numpy's interpolation for a Python-number q: the weight joins float32 math (NEP 50),
/// and the far half counts back from the upper sample.
fn interpolate(a: f32, b: f32, t: f64) -> f32 {
    let diff = b - a;
    if t >= 0.5 {
        b - diff * ((1.0 - t) as f32)
    } else {
        a + diff * (t as f32)
    }
}

/// The value at each rank in `ranks` (sorted, unique) if `values` were sorted.
fn select_ranks(values: &[f32], ranks: &[usize]) -> Vec<f32> {
    if values.len() <= DIRECT_MAX {
        let mut keys: Vec<u32> = values.iter().map(|&v| order_key(v)).collect();
        return ranks
            .iter()
            .map(|&r| from_key(*keys.select_nth_unstable(r).1))
            .collect();
    }

    let hist = values
        .par_chunks(CHUNK)
        .fold(
            || vec![0usize; BUCKETS],
            |mut h, chunk| {
                for &v in chunk {
                    h[(order_key(v) >> 16) as usize] += 1;
                }
                h
            },
        )
        .reduce(
            || vec![0usize; BUCKETS],
            |mut a, b| {
                a.iter_mut().zip(&b).for_each(|(x, y)| *x += y);
                a
            },
        );

    // Each rank becomes (bucket, rank inside the bucket); ranks are sorted, so one walk finds all.
    let mut located = Vec::with_capacity(ranks.len());
    let mut below = 0usize;
    let mut bucket = 0usize;
    for &r in ranks {
        while below + hist[bucket] <= r {
            below += hist[bucket];
            bucket += 1;
        }
        located.push((bucket, r - below));
    }

    let mut slot_of = vec![u32::MAX; BUCKETS];
    let mut wanted: Vec<usize> = located.iter().map(|&(b, _)| b).collect();
    wanted.dedup();
    for (slot, &b) in wanted.iter().enumerate() {
        slot_of[b] = slot as u32;
    }

    let gathered = values
        .par_chunks(CHUNK)
        .fold(
            || vec![Vec::new(); wanted.len()],
            |mut g: Vec<Vec<u32>>, chunk| {
                for &v in chunk {
                    let key = order_key(v);
                    let slot = slot_of[(key >> 16) as usize];
                    if slot != u32::MAX {
                        g[slot as usize].push(key);
                    }
                }
                g
            },
        )
        .reduce(
            || vec![Vec::new(); wanted.len()],
            |mut a, b| {
                a.iter_mut().zip(b).for_each(|(x, y)| x.extend(y));
                a
            },
        );

    let mut gathered = gathered;
    located
        .iter()
        .map(|&(b, r)| {
            let keys = &mut gathered[slot_of[b] as usize];
            from_key(*keys.select_nth_unstable(r).1)
        })
        .collect()
}

/// `np.percentile(values, q)` (method "linear") for each q, bit for bit, except that -0.0 and
/// 0.0 may swap: `np.partition` ranks them equal and returns either. `None` for an empty input.
pub fn percentiles(values: &[f32], qs: &[f64]) -> Option<Vec<f32>> {
    let n = values.len();
    if n == 0 {
        return None;
    }
    let spans: Vec<(usize, usize, f64)> = qs
        .iter()
        .map(|&q| {
            let (lo, t) = virtual_rank(n, q);
            (lo, (lo + 1).min(n - 1), t)
        })
        .collect();
    let mut ranks: Vec<usize> = spans.iter().flat_map(|&(lo, hi, _)| [lo, hi]).collect();
    ranks.sort_unstable();
    ranks.dedup();
    let picked = select_ranks(values, &ranks);
    let at = |r: usize| picked[ranks.binary_search(&r).unwrap_or_default()];
    Some(
        spans
            .iter()
            .map(|&(lo, hi, t)| interpolate(at(lo), at(hi), t))
            .collect(),
    )
}

/// NumPy's float32 sum of a contiguous run: one pairwise sum (blocks of 128, eight
/// accumulators), the order `np.sum` and `mean(axis=-1)` use.
pub fn pairwise_sum_f32(a: &[f32]) -> f32 {
    let n = a.len();
    if n < 8 {
        return a.iter().fold(-0.0f32, |s, &v| s + v);
    }
    if n <= 128 {
        let mut r = [0f32; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i < n - n % 8 {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for &v in &a[i..] {
            res += v;
        }
        return res;
    }
    let mut n2 = n / 2;
    n2 -= n2 % 8;
    pairwise_sum_f32(&a[..n2]) + pairwise_sum_f32(&a[n2..])
}

/// `np.percentile` itself, which is NaN for every q when any value is NaN; `percentiles`
/// matches NegPy's partition helper, which callers feed only finite values.
pub fn np_percentiles(values: &[f32], qs: &[f64]) -> Option<Vec<f32>> {
    let out = percentiles(values, qs)?;
    if values.iter().any(|v| v.is_nan()) {
        return Some(vec![f32::NAN; qs.len()]);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sorted_percentile(values: &[f32], q: f64) -> f32 {
        let mut s = values.to_vec();
        s.sort_by(f32::total_cmp);
        let (lo, t) = virtual_rank(s.len(), q);
        interpolate(s[lo], s[(lo + 1).min(s.len() - 1)], t)
    }

    fn lcg(n: usize, seed: u64) -> Vec<f32> {
        let mut x = seed;
        (0..n)
            .map(|_| {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((x >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn pairwise_sum_regions() {
        let v: Vec<f32> = (0..1000).map(|i| (i as f32).sqrt()).collect();
        for n in [0, 1, 7, 8, 9, 128, 129, 1000] {
            let exact: f64 = v[..n].iter().map(|&x| x as f64).sum();
            assert!(
                (pairwise_sum_f32(&v[..n]) as f64 - exact).abs() < 1e-2,
                "n={n}"
            );
        }
    }

    #[test]
    fn empty_is_none() {
        assert!(percentiles(&[], &[50.0]).is_none());
    }

    #[test]
    fn one_element() {
        assert_eq!(
            percentiles(&[3.5], &[0.0, 50.0, 100.0]).unwrap(),
            vec![3.5; 3]
        );
    }

    #[test]
    fn two_elements_interpolate() {
        assert_eq!(
            percentiles(&[1.0, 3.0], &[50.0, 25.0, 75.0]).unwrap(),
            vec![2.0, 1.5, 2.5]
        );
    }

    #[test]
    fn all_equal() {
        let v = vec![0.25f32; 100_000];
        assert_eq!(percentiles(&v, &[1.0, 50.0, 99.0]).unwrap(), vec![0.25; 3]);
    }

    #[test]
    fn infinities_and_signed_zero() {
        let v = [f32::INFINITY, -0.0, 0.0, f32::NEG_INFINITY, 1.0];
        // numpy's lerp gives NaN at an infinite end (inf * 0), and so does the port.
        let got = percentiles(&v, &[0.0, 50.0, 100.0]).unwrap();
        assert!(got[0].is_nan() && got[2].is_nan());
        assert_eq!(got[1], 0.0);
    }

    #[test]
    fn nan_sorts_last() {
        let v = [f32::NAN, 2.0, 1.0, -f32::NAN, 3.0];
        assert_eq!(percentiles(&v, &[0.0, 25.0]).unwrap(), vec![1.0, 2.0]);
        assert!(percentiles(&v, &[50.0]).unwrap()[0].is_nan());
    }

    #[test]
    fn order_key_round_trips_and_orders() {
        let vals = [
            f32::NEG_INFINITY,
            -2.5,
            -f32::MIN_POSITIVE,
            -0.0,
            0.0,
            1e-30,
            7.0,
            f32::INFINITY,
        ];
        for w in vals.windows(2) {
            assert!(order_key(w[0]) < order_key(w[1]));
        }
        for v in vals {
            assert_eq!(from_key(order_key(v)).to_bits(), v.to_bits());
        }
    }

    #[test]
    fn histogram_path_matches_sort() {
        let qs = [0.0, 0.5, 1.0, 20.0, 50.0, 60.0, 80.0, 99.0, 99.5, 100.0];
        for (n, seed) in [(DIRECT_MAX + 1, 1), (1_000_003, 2)] {
            let v = lcg(n, seed);
            let got = percentiles(&v, &qs).unwrap();
            for (g, &q) in got.iter().zip(&qs) {
                assert_eq!(
                    g.to_bits(),
                    sorted_percentile(&v, q).to_bits(),
                    "n={n} q={q}"
                );
            }
        }
    }

    #[test]
    fn histogram_path_with_more_buckets_than_a_byte() {
        let v = lcg(100_000, 4);
        let qs: Vec<f64> = (0..300).map(|i| i as f64 / 3.0).collect();
        let got = percentiles(&v, &qs).unwrap();
        for (g, &q) in got.iter().zip(&qs) {
            assert_eq!(g.to_bits(), sorted_percentile(&v, q).to_bits(), "q={q}");
        }
    }

    #[test]
    fn histogram_path_with_heavy_repeats() {
        let v: Vec<f32> = lcg(200_000, 3)
            .iter()
            .map(|x| (x * 4.0).round() / 4.0)
            .collect();
        let qs = [1.0, 50.0, 99.0];
        let got = percentiles(&v, &qs).unwrap();
        for (g, &q) in got.iter().zip(&qs) {
            assert_eq!(g.to_bits(), sorted_percentile(&v, q).to_bits());
        }
    }
}
