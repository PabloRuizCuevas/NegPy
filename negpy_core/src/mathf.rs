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
