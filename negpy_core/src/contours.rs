//! `cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)`, OpenCV 4.13's
//! Suzuki–Abe scanner (contours_new.cpp) with its marks, so contours and points come out in
//! OpenCV's order.

const DELTAS: [(isize, isize); 8] = [
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
];
const RIGHT: i8 = -128; // 0x80
const NEW: i8 = 2;
const FLAGS: i8 = -2; // 0xFE

/// Outer contours of the nonzero pixels of an `h` x `w` uint8 image, each a list of (x, y)
/// corner points, in OpenCV's output order (last found first).
pub fn find_external_contours(src: &[u8], h: usize, w: usize) -> Vec<Vec<(i32, i32)>> {
    // One pixel of zero border, values thresholded to 0/1, as findContours prepares them.
    let (pw, ph) = (w + 2, h + 2);
    let mut img = vec![0i8; pw * ph];
    for y in 0..h {
        for x in 0..w {
            img[(y + 1) * pw + x + 1] = (src[y * w + x] > 0) as i8;
        }
    }
    let delta = |s: usize| DELTAS[s % 8].0 + DELTAS[s % 8].1 * pw as isize;
    let mut found: Vec<Vec<(i32, i32)>> = Vec::new();

    let (width, height) = (pw - 1, ph - 1);
    let mut last_pos = (0usize, 1usize);
    for y in 1..height {
        let mut prev: i8 = 0;
        let mut x = 1;
        while x < width {
            let mut p = img[y * pw + x];
            while x < width && p == prev {
                x += 1;
                p = img[y * pw + x];
            }
            if x >= width {
                break;
            }
            // contourScan for RETR_EXTERNAL: start only at an outer border whose left
            // neighbour is not inside an already traced contour.
            let mut start = false;
            if prev == 0 && p == 1 {
                start = img[last_pos.1 * pw + last_pos.0] <= 0;
            } else if p == 0 && prev >= 1 && prev & FLAGS != 0 {
                last_pos.0 = x - 1;
            }
            if start {
                last_pos.0 = x;
                found.push(trace(
                    &mut img,
                    y * pw + x,
                    (x as i32 - 1, y as i32 - 1),
                    &delta,
                ));
            } else {
                prev = p;
                if prev & FLAGS != 0 {
                    last_pos.0 = x;
                }
            }
            x += 1;
            if start {
                prev = img[y * pw + x - 1];
            }
        }
        last_pos = (0, y + 1);
    }
    found.reverse();
    found
}

/// `icvFetchContourEx<schar>` for an outer border from `start` (an index into the padded image),
/// with CHAIN_APPROX_SIMPLE: a point wherever the chain direction changes.
fn trace(
    img: &mut [i8],
    start: usize,
    origin: (i32, i32),
    delta: &dyn Fn(usize) -> isize,
) -> Vec<(i32, i32)> {
    let at = |i: usize, d: isize| (i as isize + d) as usize;
    let mut pts = Vec::new();
    let mut pt = origin;
    let i0 = start;
    let s_end_outer = 4usize;
    let mut s = s_end_outer;
    let mut i1;
    loop {
        s = (s + 7) & 7;
        i1 = at(i0, delta(s));
        if img[i1] != 0 || s == s_end_outer {
            break;
        }
    }
    if s == s_end_outer {
        img[i0] = NEW | RIGHT;
        pts.push(pt);
        return pts;
    }
    let mut i3 = i0;
    let mut prev_s = s ^ 4;
    loop {
        let s_end = s;
        let mut i4 = i3;
        s = s.min(15);
        while s < 15 {
            s += 1;
            i4 = at(i3, delta(s));
            if img[i4] != 0 {
                break;
            }
        }
        s &= 7;
        if s.wrapping_sub(1) < s_end {
            img[i3] = NEW | RIGHT;
        } else if img[i3] == 1 {
            img[i3] = NEW;
        }
        if s != prev_s {
            pts.push(pt);
        }
        prev_s = s;
        pt = (pt.0 + DELTAS[s].0 as i32, pt.1 + DELTAS[s].1 as i32);
        if i4 == i0 && i3 == i1 {
            break;
        }
        i3 = i4;
        s = (s + 4) & 7;
    }
    pts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_gives_its_four_corners() {
        let (h, w) = (6, 7);
        let mut img = vec![0u8; h * w];
        for y in 1..4 {
            for x in 2..6 {
                img[y * w + x] = 255;
            }
        }
        let c = find_external_contours(&img, h, w);
        assert_eq!(c, vec![vec![(2, 1), (2, 3), (5, 3), (5, 1)]]);
    }

    #[test]
    fn single_pixel_and_nested_blob() {
        let (h, w) = (9, 9);
        let mut img = vec![0u8; h * w];
        img[w + 1] = 1;
        for y in 3..8 {
            for x in 3..8 {
                img[y * w + x] = 1;
            }
        }
        img[5 * w + 5] = 0; // a hole: not an external contour
        let c = find_external_contours(&img, h, w);
        assert_eq!(c.len(), 2);
        assert_eq!(c[1], vec![(1, 1)]);
    }
}
