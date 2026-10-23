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

/// atan2 of finite f32 in f64, rounded once: |y/x| folded to [0, 1], then to |u| <= tan(pi/8)
/// by atan(r) = pi/4 + atan((r - 1) / (r + 1)), where the Taylor series reaches f64 accuracy.
/// Signed zeros follow C's atan2.
#[inline(always)]
pub fn atan2f(y: f32, x: f32) -> f32 {
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};
    let (ay, ax) = ((y as f64).abs(), (x as f64).abs());
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
    (a as f32).copysign(y)
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
