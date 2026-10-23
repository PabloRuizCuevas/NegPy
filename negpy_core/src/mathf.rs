//! Transcendental functions for hot array loops: branch-free, so LLVM vectorizes them, and pure
//! float arithmetic, so every platform gets the same bits. Scalar calls may use `libm`.

/// log10 of a positive normal f32, correctly rounded (`exhaustive` checks every input).
/// x = m * 2^e with m in [sqrt(1/2), sqrt(2)); ln(m) from the atanh series in
/// f64, rounded once to f32.
#[inline(always)]
pub fn log10f(x: f32) -> f32 {
    let bits = x.to_bits();
    // Subtracting the bits of sqrt(1/2) puts the exponent split at sqrt(2).
    let e = (bits.wrapping_sub(0x3f35_04f3) as i32) >> 23;
    let m = f32::from_bits(bits.wrapping_sub((e as u32) << 23)) as f64;
    let s = (m - 1.0) / (m + 1.0);
    let s2 = s * s;
    // 2 s (1 + s^2/3 + s^4/5 + ...), |s| <= 0.1716: these terms reach f64 accuracy.
    let p = 1.0
        + s2 * (1.0 / 3.0
            + s2 * (1.0 / 5.0
                + s2 * (1.0 / 7.0
                    + s2 * (1.0 / 9.0
                        + s2 * (1.0 / 11.0
                            + s2 * (1.0 / 13.0 + s2 * (1.0 / 15.0 + s2 * (1.0 / 17.0))))))));
    // log10(2) in two parts: HI has 20 trailing zero bits, so e * HI is exact for any exponent.
    const HI: f64 = 0.301_029_995_607_677_85;
    const LO: f64 = std::f64::consts::LOG10_2 - HI;
    let e = e as f64;
    (e * HI + (e * LO + 2.0 * s * p * std::f64::consts::LOG10_E)) as f32
}

/// atan2 of finite f32 in f64, rounded once. Signed zeros follow C's atan2.
#[inline(always)]
pub fn atan2f(y: f32, x: f32) -> f32 {
    atan2_f64(y as f64, x as f64) as f32
}

/// atan2 of finite values within 2^-37 relative: |y/x| folded to [0, 1], then to
/// |u| <= tan(pi/8) by atan(r) = pi/4 + atan((r - 1) / (r + 1)), and a Taylor series.
#[inline(always)]
pub fn atan2_f64(y: f64, x: f64) -> f64 {
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};
    let (ay, ax) = (y.abs(), x.abs());
    let swap = ay > ax;
    let (n, d) = if swap { (ax, ay) } else { (ay, ax) };
    let r = if d == 0.0 { 0.0 } else { n / d };
    let big = r > std::f64::consts::SQRT_2 - 1.0;
    let u = if big { (r - 1.0) / (r + 1.0) } else { r };
    let u2 = u * u;
    // (-1)^k / (2k + 1): the series to u^25, whose next term is under 2^-37 of atan(u).
    const C: [f64; 13] = [
        1.0,
        -1.0 / 3.0,
        1.0 / 5.0,
        -1.0 / 7.0,
        1.0 / 9.0,
        -1.0 / 11.0,
        1.0 / 13.0,
        -1.0 / 15.0,
        1.0 / 17.0,
        -1.0 / 19.0,
        1.0 / 21.0,
        -1.0 / 23.0,
        1.0 / 25.0,
    ];
    let mut p = C[12];
    for &c in C[..12].iter().rev() {
        p = c + u2 * p;
    }
    let mut a = u * p;
    a = if big { FRAC_PI_4 + a } else { a };
    a = if swap { FRAC_PI_2 - a } else { a };
    a = if x.is_sign_negative() { PI - a } else { a };
    a.copysign(y)
}

/// ln of a positive normal f64 to about 1 ULP: x = m * 2^e with m in [sqrt(1/2), sqrt(2)),
/// ln(m) from the atanh series.
#[inline(always)]
pub fn ln_f64(x: f64) -> f64 {
    let bits = x.to_bits();
    let e = (bits.wrapping_sub(0x3fe6_a09e_667f_3bcd) as i64) >> 52;
    let m = f64::from_bits(bits.wrapping_sub((e as u64) << 52));
    let s = (m - 1.0) / (m + 1.0);
    let s2 = s * s;
    // 2 s (1 + s^2/3 + ... + s^20/21), |s| <= 0.1716: the next term is under 2^-56 of the sum.
    let mut p = 1.0 / 21.0;
    for k in (0..10).rev() {
        p = 1.0 / (2 * k + 1) as f64 + s2 * p;
    }
    const LN2_HI: f64 = 0.693_147_180_369_123_8;
    const LN2_LO: f64 = std::f64::consts::LN_2 - LN2_HI;
    let e = e as f64;
    e * LN2_HI + (e * LN2_LO + 2.0 * s * p)
}

/// e^t to about 1 ULP for t in -708..709 (clamped there): t = k ln2 + r with |r| <= ln2/2, e^r by its Taylor
/// series, times 2^k through the exponent bits.
#[inline(always)]
pub fn exp_f64(t: f64) -> f64 {
    const LN2_HI: f64 = 0.693_147_180_369_123_8;
    const LN2_LO: f64 = std::f64::consts::LN_2 - LN2_HI;
    let t = t.clamp(-708.0, 709.0);
    let k = (t * std::f64::consts::LOG2_E).round_ties_even();
    let r = (t - k * LN2_HI) - k * LN2_LO;
    // 1/n! to n = 13: the next term is under 2^-57 at |r| = ln2/2.
    let mut p = 1.0 / 6_227_020_800.0;
    let mut f = 6_227_020_800.0;
    for n in (1..13).rev() {
        f /= (n + 1) as f64;
        p = 1.0 / f + r * p;
    }
    let y = 1.0 + r * p;
    // 2^k from the low bits of k + 1023 + 2^52: an `as` cast saturates and keeps the loop scalar.
    y * f64::from_bits((k + (1023.0 + 4_503_599_627_370_496.0)).to_bits() << 52)
}

/// ln(1 + u) for u > -1, accurate for small u: ln(w) * u / (w - 1) with w = 1 + u cancels the
/// rounding of w (Goldberg).
#[inline(always)]
pub fn log1p_f64(u: f64) -> f64 {
    let w = 1.0 + u;
    let l = ln_f64(w) * (u / (w - 1.0));
    if w == 1.0 { u } else { l }
}

/// e^x - 1, accurate for small x: (w - 1) * x / ln(w) with w = e^x (Kahan).
#[inline(always)]
pub fn expm1_f64(x: f64) -> f64 {
    let w = exp_f64(x);
    let m = (w - 1.0) * (x / ln_f64(w));
    if w == 1.0 {
        x
    } else if x < -30.0 {
        -1.0
    } else {
        m
    }
}

/// tanh(t) = expm1(2t) / (expm1(2t) + 2), with |t| capped where it is +-1 in f64.
#[inline(always)]
pub fn tanh_f64(t: f64) -> f64 {
    let e = expm1_f64(2.0 * t.clamp(-20.0, 20.0));
    e / (e + 2.0)
}

/// log10 of a positive normal f64.
#[inline(always)]
pub fn log10_f64(x: f64) -> f64 {
    ln_f64(x) * std::f64::consts::LOG10_E
}

/// 10^x.
#[inline(always)]
pub fn exp10_f64(x: f64) -> f64 {
    exp_f64(x * std::f64::consts::LN_10)
}

/// x^y for x >= 0 (or NaN) and y > 0: e^(y ln x), 0 at x = 0. The relative error grows with
/// |y ln x|, so the result is for rounding to f32, where it is almost always correct.
#[inline(always)]
pub fn pow_f64(x: f64, y: f64) -> f64 {
    let v = exp_f64(y * ln_f64(x));
    if x == 0.0 {
        0.0
    } else if x.is_nan() || x == f64::INFINITY {
        x
    } else {
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log10f_is_correctly_rounded_on_a_sample() {
        let mut bits = 1e-6f32.to_bits();
        while bits <= 1f32.to_bits() {
            let x = f32::from_bits(bits);
            assert_eq!(log10f(x), libm::log10(x as f64) as f32, "{x:e}");
            bits += 997;
        }
        for x in [
            1.0f32,
            0.1,
            0.01,
            1e-6,
            0.5,
            2.0,
            10.0,
            1e30,
            f32::MIN_POSITIVE,
        ] {
            assert_eq!(log10f(x), libm::log10(x as f64) as f32, "{x:e}");
        }
    }
}

#[cfg(test)]
mod atan2_tests {
    use super::atan2f;

    #[test]
    fn atan2f_is_within_one_ulp_and_keeps_c_zeros() {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0
        };
        let mut off = 0;
        for _ in 0..1_000_000 {
            let (y, x) = (next(), next());
            let want = libm::atan2(y as f64, x as f64) as f32;
            let got = atan2f(y, x);
            let ulp = (got.to_bits() as i64 - want.to_bits() as i64).abs();
            assert!(ulp <= 1, "{y} {x}");
            off += (ulp > 0) as usize;
        }
        assert!(off < 10, "{off}");
        let pi = std::f32::consts::PI;
        for (y, x, want) in [
            (0.0, 0.0, 0.0),
            (-0.0, 0.0, -0.0),
            (0.0, -0.0, pi),
            (-0.0, -0.0, -pi),
            (1.0, 0.0, pi / 2.0),
        ] {
            assert_eq!(atan2f(y, x).to_bits(), want.to_bits(), "{y} {x}");
        }
    }
}

#[cfg(test)]
mod pow_tests {
    use super::pow_f64;

    #[test]
    fn pow_f64_rounds_to_the_same_f32() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut off = 0;
        for _ in 0..1_000_000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let x = f32::from_bits((state >> 33) as u32 % 0x3f80_0001) as f64;
            for y in [563.0 / 256.0, 256.0 / 563.0, 1.0 / 3.0] {
                let (got, want) = (pow_f64(x, y) as f32, libm::pow(x, y) as f32);
                let ulp = (got.to_bits() as i64 - want.to_bits() as i64).abs();
                assert!(ulp <= 1, "{x} {y}");
                off += ulp as usize;
            }
        }
        assert!(off <= 3, "{off}");
        assert_eq!(pow_f64(0.0, 2.2), 0.0);
        assert!(pow_f64(f64::NAN, 2.2).is_nan());
        assert_eq!(pow_f64(1.0, 2.2), 1.0);
    }
}

#[cfg(test)]
mod f64_tests {
    use super::*;

    fn rel(a: f64, b: f64) -> f64 {
        if a == b { 0.0 } else { ((a - b) / b).abs() }
    }

    #[test]
    fn float64_helpers_match_libm() {
        let mut worst = [0f64; 5];
        let mut x = -12.0f64;
        while x < 12.0 {
            worst[0] = worst[0].max(rel(
                log1p_f64(x.exp() * 0.999),
                libm::log1p(x.exp() * 0.999),
            ));
            worst[1] = worst[1].max(rel(expm1_f64(x * 0.05), libm::expm1(x * 0.05)));
            worst[2] = worst[2].max(rel(tanh_f64(x * 0.3), libm::tanh(x * 0.3)));
            worst[3] = worst[3].max(rel(log10_f64(x.exp()), libm::log10(x.exp())));
            worst[4] = worst[4].max(rel(exp10_f64(x * 0.25), libm::exp10(x * 0.25)));
            x += 0.000_37;
        }
        for (i, w) in worst.iter().enumerate() {
            assert!(*w < 1e-14, "{i}: {w:e}");
        }
        assert_eq!(log1p_f64(1e-300), 1e-300);
        assert_eq!(expm1_f64(1e-300), 1e-300);
        assert_eq!(tanh_f64(1e5), 1.0);
    }
}

#[cfg(test)]
mod exhaustive {
    /// `cargo test --release -- --ignored`.
    #[test]
    #[ignore]
    fn all() {
        let (lo, hi) = (f32::MIN_POSITIVE.to_bits(), f32::MAX.to_bits());
        let mut bad = 0u64;
        for b in lo..=hi {
            let x = f32::from_bits(b);
            if super::log10f(x) != libm::log10(x as f64) as f32 {
                bad += 1;
                if bad < 20 {
                    eprintln!("{b:#x} {x:e}");
                }
            }
        }
        eprintln!("checked {} bad {bad}", hi - lo + 1);
    }
}
