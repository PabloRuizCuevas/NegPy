//! `cv2.distanceTransform` for a uint8 mask: chamfer distances to the nearest zero pixel.
//! Without labels, opencv-python runs IPP's float32 two-pass chamfer; with labels, OpenCV's own
//! 5x5 pass in 16-bit fixed point. IPP's 3x3 pass picks a path one float32 unit longer at rare
//! pixels than this two-pass order does (negpy_core/LEDGER.md).

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    /// DIST_C, 3x3: steps of 1.
    Chessboard,
    /// DIST_L2, 3x3: 0.955 and 1.3693.
    L2Mask3,
    /// DIST_L2, 5x5: 1, 1.4 and 2.1969.
    L2Mask5,
}

/// (row offset, column offset, step index) of the forward pass; the backward pass mirrors it.
fn forward(metric: Metric) -> &'static [(isize, isize, usize)] {
    match metric {
        Metric::Chessboard | Metric::L2Mask3 => &[(-1, -1, 1), (-1, 0, 0), (-1, 1, 1), (0, -1, 0)],
        Metric::L2Mask5 => &[
            (-2, -1, 2),
            (-2, 1, 2),
            (-1, -2, 2),
            (-1, -1, 1),
            (-1, 0, 0),
            (-1, 1, 1),
            (-1, 2, 2),
            (0, -1, 0),
        ],
    }
}

fn steps(metric: Metric) -> [f32; 3] {
    match metric {
        Metric::Chessboard => [1.0, 1.0, 0.0],
        Metric::L2Mask3 => [0.955, 1.3693, 0.0],
        Metric::L2Mask5 => [1.0, 1.4, 2.1969],
    }
}

/// Distances in float32: a forward and a backward pass over a frame two pixels wide.
pub fn distance_transform(src: &[u8], h: usize, w: usize, metric: Metric) -> Vec<f32> {
    const B: usize = 2;
    let (pw, ph) = (w + 2 * B, h + 2 * B);
    let far = f32::MAX;
    let mut t = vec![far; pw * ph];
    let step = steps(metric);
    let fwd = forward(metric);
    let at = |y: usize, x: usize, dy: isize, dx: isize| {
        ((y as isize + dy) as usize) * pw + (x as isize + dx) as usize
    };
    for y in B..h + B {
        for x in B..w + B {
            if src[(y - B) * w + x - B] == 0 {
                t[y * pw + x] = 0.0;
                continue;
            }
            let mut v = far;
            for &(dy, dx, s) in fwd {
                v = v.min(t[at(y, x, dy, dx)] + step[s]);
            }
            t[y * pw + x] = v;
        }
    }
    let mut out = vec![0f32; h * w];
    for y in (B..h + B).rev() {
        for x in (B..w + B).rev() {
            let mut v = t[y * pw + x];
            if v > step[0] {
                for &(dy, dx, s) in fwd {
                    v = v.min(t[at(y, x, -dy, -dx)] + step[s]);
                }
                t[y * pw + x] = v;
            }
            out[(y - B) * w + x - B] = v;
        }
    }
    out
}

/// `cv2.distanceTransformWithLabels(src, DIST_L2, ..., labelType=DIST_LABEL_PIXEL)`: OpenCV's
/// 5x5 fixed-point pass, each pixel labelled with its nearest zero pixel, zeros numbered from 1
/// in raster order.
pub fn distance_transform_labels(src: &[u8], h: usize, w: usize) -> (Vec<f32>, Vec<i32>) {
    const B: usize = 2;
    const SHIFT: i32 = 16;
    let fix = |v: f32| ((v as f64) * (1 << SHIFT) as f64).round_ties_even() as u32;
    let [hv, diag, long] = steps(Metric::L2Mask5).map(fix);
    let cost = [hv, diag, long];
    let far = u32::MAX - long;
    let scale = 1.0f32 / (1 << SHIFT) as f32;
    let (pw, ph) = (w + 2 * B, h + 2 * B);
    let mut t = vec![far; pw * ph];
    let mut lab = vec![0i32; pw * ph];
    let mut next = 1;
    for y in 0..h {
        for x in 0..w {
            if src[y * w + x] == 0 {
                lab[(y + B) * pw + x + B] = next;
                next += 1;
            }
        }
    }
    let fwd = forward(Metric::L2Mask5);
    let at = |y: usize, x: usize, dy: isize, dx: isize| {
        ((y as isize + dy) as usize) * pw + (x as isize + dx) as usize
    };
    for y in B..h + B {
        for x in B..w + B {
            if src[(y - B) * w + x - B] == 0 {
                t[y * pw + x] = 0;
                continue;
            }
            let (mut v, mut l) = (far, 0);
            for &(dy, dx, s) in fwd {
                let i = at(y, x, dy, dx);
                let c = t[i].wrapping_add(cost[s]);
                if v > c {
                    v = c;
                    l = lab[i];
                }
            }
            t[y * pw + x] = v;
            lab[y * pw + x] = l;
        }
    }
    let mut dist = vec![0f32; h * w];
    let mut labels = vec![0i32; h * w];
    for y in (B..h + B).rev() {
        for x in (B..w + B).rev() {
            let (mut v, mut l) = (t[y * pw + x], lab[y * pw + x]);
            if v > hv {
                for &(dy, dx, s) in fwd {
                    let i = at(y, x, -dy, -dx);
                    let c = t[i].wrapping_add(cost[s]);
                    if v > c {
                        v = c;
                        l = lab[i];
                    }
                }
                t[y * pw + x] = v;
                lab[y * pw + x] = l;
            }
            dist[(y - B) * w + x - B] = v as f32 * scale;
            labels[(y - B) * w + x - B] = l;
        }
    }
    (dist, labels)
}
