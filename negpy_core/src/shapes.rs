//! OpenCV's shape helpers for integer contours: convexHull, minAreaRect (rotating calipers),
//! boxPoints, boundingRect of float points, and fillConvexPoly. Ported from convhull.cpp,
//! rotcalipers.cpp, geometry.cpp, types.cpp and drawing.cpp, which build without FMA on
//! x86-64, so plain float arithmetic matches them.

/// `convexHull(points, clockwise=false, returnPoints=true)` of integer points.
pub fn convex_hull(points: &[(i32, i32)]) -> Vec<(i32, i32)> {
    let total = points.len();
    if total == 0 {
        return Vec::new();
    }
    // Sorted by x, then y, then original position (the comparator's pointer fallback).
    let mut order: Vec<usize> = (0..total).collect();
    order.sort_by(|&a, &b| {
        points[a]
            .0
            .cmp(&points[b].0)
            .then(points[a].1.cmp(&points[b].1))
            .then(a.cmp(&b))
    });
    let pt = |i: usize| points[order[i]];
    let (mut miny, mut maxy) = (0usize, 0usize);
    for i in 1..total {
        let y = pt(i).1;
        if pt(miny).1 > y {
            miny = i;
        }
        if pt(maxy).1 < y {
            maxy = i;
        }
    }
    let mut hull: Vec<usize> = Vec::new();
    if pt(0) == pt(total - 1) {
        hull.push(0);
    } else {
        let mut stack = vec![0usize; total + 2];
        let tl_count = sklansky(&pt, 0, maxy as isize, &mut stack[..], -1, 1);
        let tl: Vec<usize> = stack[..tl_count].to_vec();
        let tr_count = sklansky(
            &pt,
            total as isize - 1,
            maxy as isize,
            &mut stack[..],
            -1,
            -1,
        );
        let tr: Vec<usize> = stack[..tr_count].to_vec();
        // Counter-clockwise: the right chain first.
        let (tl, tr) = (tr, tl);
        for &i in &tl[..tl.len().saturating_sub(1)] {
            hull.push(i);
        }
        for i in (1..tr.len()).rev() {
            hull.push(tr[i]);
        }
        let stop_idx: isize = if tr.len() > 2 {
            tr[1] as isize
        } else if tl.len() > 2 {
            tl[tl.len() - 2] as isize
        } else {
            -1
        };
        let bl_count = sklansky(&pt, 0, miny as isize, &mut stack[..], 1, -1);
        let bl: Vec<usize> = stack[..bl_count].to_vec();
        let br_count = sklansky(&pt, total as isize - 1, miny as isize, &mut stack[..], 1, 1);
        let br: Vec<usize> = stack[..br_count].to_vec();
        let (mut blc, mut brc) = (bl.len(), br.len());
        if stop_idx >= 0 {
            let check_idx: isize = if blc > 2 {
                bl[1] as isize
            } else if blc + brc > 2 {
                br[2 - blc] as isize
            } else {
                -1
            };
            if check_idx == stop_idx
                || (check_idx >= 0 && pt(check_idx as usize) == pt(stop_idx as usize))
            {
                blc = blc.min(2);
                brc = brc.min(2);
            }
        }
        for &i in &bl[..blc.saturating_sub(1)] {
            hull.push(i);
        }
        for i in (1..brc).rev() {
            hull.push(br[i]);
        }
    }
    let mut idx: Vec<usize> = hull.iter().map(|&i| order[i]).collect();
    // Rotate toward an ascending or descending index sequence, as OpenCV does.
    let nout = idx.len();
    if nout >= 3 {
        let (mut min_idx, mut max_idx, mut lt) = (0usize, 0usize, 0usize);
        for i in 1..nout {
            let v = idx[i];
            lt += (idx[i - 1] < v) as usize;
            if lt > 1 && lt + 2 <= i {
                break;
            }
            if v < idx[min_idx] {
                min_idx = i;
            }
            if v > idx[max_idx] {
                max_idx = i;
            }
        }
        let mmdist = max_idx.abs_diff(min_idx);
        if (mmdist == 1 || mmdist == nout - 1) && (lt <= 1 || lt + 2 >= nout) {
            let ascending = (max_idx + 1) % nout == min_idx;
            let i0 = if ascending { min_idx } else { max_idx };
            if i0 > 0 {
                let mut out = Vec::with_capacity(nout);
                let mut j = i0;
                let mut complete = true;
                for i in 0..nout {
                    let curr = idx[j];
                    out.push(curr);
                    let next_j = if j + 1 < nout { j + 1 } else { 0 };
                    if i < nout - 1 && (ascending != (curr < idx[next_j])) {
                        complete = false;
                        break;
                    }
                    j = next_j;
                }
                if complete {
                    idx = out;
                }
            }
        }
    }
    idx.iter().map(|&i| points[i]).collect()
}

/// `Sklansky_<int, int64>`: one monotone chain between `start` and `end` (sorted indices).
fn sklansky(
    pt: &dyn Fn(usize) -> (i32, i32),
    start: isize,
    end: isize,
    stack: &mut [usize],
    nsign: i32,
    sign2: i32,
) -> usize {
    let incr: isize = if end > start { 1 } else { -1 };
    let (mut pprev, mut pcur) = (start, start + incr);
    let mut pnext = pcur + incr;
    let mut size = 3usize;
    let p = |i: isize| pt(i as usize);
    if start == end || p(start) == p(end) {
        stack[0] = start as usize;
        return 1;
    }
    stack[0] = pprev as usize;
    stack[1] = pcur as usize;
    stack[2] = pnext as usize;
    let end = end + incr;
    while pnext != end {
        let (cury, nexty) = (p(pcur).1, p(pnext).1);
        let by = nexty - cury;
        if by.signum() != nsign {
            let a = (p(pcur).0 - p(pprev).0, cury - p(pprev).1);
            let b = (p(pnext).0 - p(pcur).0, by);
            let convexity = a.1 as i64 * b.0 as i64 - a.0 as i64 * b.1 as i64;
            if convexity.signum() as i32 == sign2 && (a.0 != 0 || a.1 != 0) {
                pprev = pcur;
                pcur = pnext;
                pnext += incr;
                stack[size] = pnext as usize;
                size += 1;
            } else if pprev == start {
                pcur = pnext;
                stack[1] = pcur as usize;
                pnext += incr;
                stack[2] = pnext as usize;
            } else {
                stack[size - 2] = pnext as usize;
                pcur = pprev;
                pprev = stack[size - 4] as isize;
                size -= 1;
            }
        } else {
            pnext += incr;
            stack[size - 1] = pnext as usize;
        }
    }
    size - 1
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RotatedRect {
    pub center: (f32, f32),
    pub size: (f32, f32),
    pub angle: f32,
}

fn right_of(v1: (f32, f32), v2: (f32, f32)) -> bool {
    // rotate90CW(v1) . v2 < 0
    v1.1 * v2.0 + (-v1.0) * v2.1 < 0.0
}

/// `cv2.minAreaRect` of integer points.
pub fn min_area_rect(points: &[(i32, i32)]) -> RotatedRect {
    let hull: Vec<(f32, f32)> = convex_hull(points)
        .iter()
        .map(|&(x, y)| (x as f32, y as f32))
        .collect();
    let n = hull.len();
    let mut angle = -std::f64::consts::PI / 2.0;
    let mut rect = RotatedRect {
        center: (0.0, 0.0),
        size: (0.0, 0.0),
        angle: 0.0,
    };
    if n > 2 {
        let out = rotating_calipers(&hull);
        rect.center = (
            out[0].0 + (out[1].0 + out[2].0) * 0.5,
            out[0].1 + (out[1].1 + out[2].1) * 0.5,
        );
        rect.size.0 =
            ((out[2].0 as f64 * out[2].0 as f64 + out[2].1 as f64 * out[2].1 as f64).sqrt()) as f32;
        rect.size.1 =
            ((out[1].0 as f64 * out[1].0 as f64 + out[1].1 as f64 * out[1].1 as f64).sqrt()) as f32;
        if out[1].0 == 0.0 && out[1].1 > 0.0 {
            rect.size = (rect.size.1, rect.size.0);
        } else {
            angle = -libm::atan2(out[1].0 as f64, out[1].1 as f64);
        }
    } else if n == 2 {
        rect.center = ((hull[0].0 + hull[1].0) * 0.5, (hull[0].1 + hull[1].1) * 0.5);
        let dx = hull[0].0 as f64 - hull[1].0 as f64;
        let dy = hull[0].1 as f64 - hull[1].1 as f64;
        rect.size = (0.0, (dx * dx + dy * dy).sqrt() as f32);
        if dx == 0.0 {
            rect.size = (rect.size.1, rect.size.0);
        } else if dy < 0.0 {
            angle = libm::atan2(dy, dx);
            rect.size = (rect.size.1, rect.size.0);
        } else if dy > 0.0 {
            angle = -libm::atan2(dx, dy);
        }
    } else if n == 1 {
        rect.center = hull[0];
    }
    rect.angle = (angle * 180.0 / std::f64::consts::PI) as f32;
    rect
}

/// `rotatingCalipers(..., CALIPERS_MINAREARECT)` for a counter-clockwise hull: the corner point
/// and the two side vectors.
fn rotating_calipers(points: &[(f32, f32)]) -> [(f32, f32); 3] {
    let n = points.len();
    let mut vect = vec![(0f32, 0f32); n];
    let mut inv_len = vec![0f32; n];
    let (mut left, mut bottom, mut right, mut top) = (0usize, 0usize, 0usize, 0usize);
    let mut pt0 = points[0];
    let (mut left_x, mut right_x, mut top_y, mut bottom_y) = (pt0.0, pt0.0, pt0.1, pt0.1);
    for i in 0..n {
        if pt0.0 < left_x {
            left_x = pt0.0;
            left = i;
        }
        if pt0.0 > right_x {
            right_x = pt0.0;
            right = i;
        }
        if pt0.1 > top_y {
            top_y = pt0.1;
            top = i;
        }
        if pt0.1 < bottom_y {
            bottom_y = pt0.1;
            bottom = i;
        }
        let pt = points[if i + 1 < n { i + 1 } else { 0 }];
        let dx = pt.0 as f64 - pt0.0 as f64;
        let dy = pt.1 as f64 - pt0.1 as f64;
        vect[i] = (dx as f32, dy as f32);
        inv_len[i] = (1.0 / (dx * dx + dy * dy).sqrt()) as f32;
        pt0 = pt;
    }
    let mut seq = [bottom, right, top, left];
    let mut minarea = f32::MAX;
    let mut best_left = 0usize;
    let mut best_bottom = 0usize;
    let (mut ba, mut bw, mut bb, mut bh) = (0f32, 0f32, 0f32, 0f32);
    for _ in 0..n {
        let rot = [
            vect[seq[0]],
            (vect[seq[1]].1, -vect[seq[1]].0),
            (-vect[seq[2]].0, -vect[seq[2]].1),
            (-vect[seq[3]].1, vect[seq[3]].0),
        ];
        let mut main = 0usize;
        for i in 1..4 {
            if right_of(rot[i], rot[main]) {
                main = i;
            }
        }
        let pindex = seq[main];
        let lead_x = vect[pindex].0 * inv_len[pindex];
        let lead_y = vect[pindex].1 * inv_len[pindex];
        let (base_a, base_b) = match main {
            0 => (lead_x, lead_y),
            1 => (lead_y, -lead_x),
            2 => (-lead_x, -lead_y),
            _ => (-lead_y, lead_x),
        };
        seq[main] = if seq[main] + 1 == n { 0 } else { seq[main] + 1 };
        let dx = points[seq[1]].0 - points[seq[3]].0;
        let dy = points[seq[1]].1 - points[seq[3]].1;
        let width = dx * base_a + dy * base_b;
        let dx = points[seq[2]].0 - points[seq[0]].0;
        let dy = points[seq[2]].1 - points[seq[0]].1;
        let height = -dx * base_b + dy * base_a;
        let area = width * height;
        if area <= minarea {
            minarea = area;
            best_left = seq[3];
            ba = base_a;
            bw = width;
            bb = base_b;
            bh = height;
            best_bottom = seq[0];
        }
    }
    let (a1, b1) = (ba, bb);
    let (a2, b2) = (-bb, ba);
    let c1 = a1 * points[best_left].0 + points[best_left].1 * b1;
    let c2 = a2 * points[best_bottom].0 + points[best_bottom].1 * b2;
    let idet = 1.0f32 / (a1 * b2 - a2 * b1);
    let px = (c1 * b2 - c2 * b1) * idet;
    let py = (a1 * c2 - a2 * c1) * idet;
    [(px, py), (a1 * bw, b1 * bw), (a2 * bh, b2 * bh)]
}

/// `cv2.boxPoints(rect)` (`RotatedRect::points`); cos and sin are libm's (see LEDGER.md).
pub fn box_points(r: &RotatedRect) -> [(f32, f32); 4] {
    let a_rad = r.angle as f64 * std::f64::consts::PI / 180.0;
    let b = libm::cos(a_rad) as f32 * 0.5;
    let a = libm::sin(a_rad) as f32 * 0.5;
    let (cx, cy) = r.center;
    let (w, h) = r.size;
    let p0 = (cx - a * h - b * w, cy + b * h - a * w);
    let p1 = (cx + a * h - b * w, cy - b * h - a * w);
    let p2 = (2.0 * cx - p0.0, 2.0 * cy - p0.1);
    let p3 = (2.0 * cx - p1.0, 2.0 * cy - p1.1);
    [p0, p1, p2, p3]
}

/// `cv2.boundingRect` of float points: floors of the extremes, (x, y, width, height).
pub fn bounding_rect_f32(pts: &[(f32, f32)]) -> (i32, i32, i32, i32) {
    if pts.is_empty() {
        return (0, 0, 0, 0);
    }
    let floor = |v: f32| v.floor() as i32;
    let (mut x0, mut y0, mut x1, mut y1) = (
        floor(pts[0].0),
        floor(pts[0].1),
        floor(pts[0].0),
        floor(pts[0].1),
    );
    for &(x, y) in pts {
        x0 = x0.min(floor(x));
        y0 = y0.min(floor(y));
        x1 = x1.max(floor(x));
        y1 = y1.max(floor(y));
    }
    (x0, y0, x1 - x0 + 1, y1 - y0 + 1)
}

const XY_SHIFT: i64 = 16;
const XY_ONE: i64 = 1 << XY_SHIFT;

/// `clipLine(Size2l, pt1, pt2)`: false when the segment misses the image.
fn clip_line(w: i64, h: i64, p1: &mut (i64, i64), p2: &mut (i64, i64)) -> bool {
    let (right, bottom) = (w - 1, h - 1);
    if w <= 0 || h <= 0 {
        return false;
    }
    let code = |p: &(i64, i64)| {
        (p.0 < 0) as i32
            + (p.0 > right) as i32 * 2
            + (p.1 < 0) as i32 * 4
            + (p.1 > bottom) as i32 * 8
    };
    let (mut c1, mut c2) = (code(p1), code(p2));
    if (c1 & c2) == 0 && (c1 | c2) != 0 {
        if c1 & 12 != 0 {
            let a = if c1 < 8 { 0 } else { bottom };
            p1.0 += ((a - p1.1) as f64 * (p2.0 - p1.0) as f64 / (p2.1 - p1.1) as f64) as i64;
            p1.1 = a;
            c1 = (p1.0 < 0) as i32 + (p1.0 > right) as i32 * 2;
        }
        if c2 & 12 != 0 {
            let a = if c2 < 8 { 0 } else { bottom };
            p2.0 += ((a - p2.1) as f64 * (p2.0 - p1.0) as f64 / (p2.1 - p1.1) as f64) as i64;
            p2.1 = a;
            c2 = (p2.0 < 0) as i32 + (p2.0 > right) as i32 * 2;
        }
        if (c1 & c2) == 0 && (c1 | c2) != 0 {
            if c1 != 0 {
                let a = if c1 == 1 { 0 } else { right };
                p1.1 += ((a - p1.0) as f64 * (p2.1 - p1.1) as f64 / (p2.0 - p1.0) as f64) as i64;
                p1.0 = a;
                c1 = 0;
            }
            if c2 != 0 {
                let a = if c2 == 1 { 0 } else { right };
                p2.1 += ((a - p2.0) as f64 * (p2.1 - p1.1) as f64 / (p2.0 - p1.0) as f64) as i64;
                p2.0 = a;
                c2 = 0;
            }
        }
    }
    (c1 | c2) == 0
}

/// `Line(img, pt1, pt2, color, 8)`: an 8-connected line drawn left to right.
fn draw_line(mask: &mut [u8], w: usize, h: usize, pt1: (i64, i64), pt2: (i64, i64), color: u8) {
    let (mut p1, mut p2) = (pt1, pt2);
    let outside = |p: &(i64, i64)| p.0 < 0 || p.0 >= w as i64 || p.1 < 0 || p.1 >= h as i64;
    if (outside(&p1) || outside(&p2)) && !clip_line(w as i64, h as i64, &mut p1, &mut p2) {
        return;
    }
    let (mut dx, mut dy) = (p2.0 - p1.0, p2.1 - p1.1);
    let (mut delta_x, mut delta_y) = (1i64, 1i64);
    if dx < 0 {
        dx = -dx;
        dy = -dy;
        p1 = p2;
    }
    if dy < 0 {
        dy = -dy;
        delta_y = -1;
    }
    let vert = dy > dx;
    if vert {
        std::mem::swap(&mut dx, &mut dy);
        std::mem::swap(&mut delta_x, &mut delta_y);
    }
    let mut err = dx - (dy + dy);
    let plus_delta = dx + dx;
    let minus_delta = -(dy + dy);
    // (x, y) moves. OpenCV swaps step and shift for a steep line, after swapping the deltas.
    let (minus, plus) = if vert {
        ((0, delta_x), (delta_y, 0))
    } else {
        ((delta_x, 0), (0, delta_y))
    };
    let (mut x, mut y) = p1;
    for _ in 0..=dx {
        if x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h {
            mask[y as usize * w + x as usize] = color;
        }
        let m = err < 0;
        err += minus_delta + if m { plus_delta } else { 0 };
        x += minus.0 + if m { plus.0 } else { 0 };
        y += minus.1 + if m { plus.1 } else { 0 };
    }
}

/// `cv2.fillConvexPoly(mask, pts, color)` (8-connected outline, shift 0) on a w x h uint8 mask.
pub fn fill_convex_poly(mask: &mut [u8], w: usize, h: usize, v: &[(i64, i64)], color: u8) {
    let npts = v.len();
    if npts == 0 {
        return;
    }
    let delta1 = XY_ONE >> 1;
    let delta2 = XY_ONE >> 1;
    let mut p0 = (v[npts - 1].0 << XY_SHIFT, v[npts - 1].1 << XY_SHIFT);
    let (mut xmin, mut xmax, mut ymin, mut ymax) = (v[0].0, v[0].0, v[0].1, v[0].1);
    let mut imin = 0usize;
    for (i, &q) in v.iter().enumerate() {
        if q.1 < ymin {
            ymin = q.1;
            imin = i;
        }
        ymax = ymax.max(q.1);
        xmax = xmax.max(q.0);
        xmin = xmin.min(q.0);
        let p = (q.0 << XY_SHIFT, q.1 << XY_SHIFT);
        draw_line(
            mask,
            w,
            h,
            (p0.0 >> XY_SHIFT, p0.1 >> XY_SHIFT),
            (p.0 >> XY_SHIFT, p.1 >> XY_SHIFT),
            color,
        );
        p0 = p;
    }
    if npts < 3 || xmax < 0 || ymax < 0 || xmin >= w as i64 || ymin >= h as i64 {
        return;
    }
    let ymax = ymax.min(h as i64 - 1);
    #[derive(Clone, Copy)]
    struct Edge {
        idx: usize,
        di: usize,
        x: i64,
        dx: i64,
        ye: i64,
    }
    let mut edge = [
        Edge {
            idx: imin,
            di: 1,
            x: -XY_ONE,
            dx: 0,
            ye: ymin,
        },
        Edge {
            idx: imin,
            di: npts - 1,
            x: -XY_ONE,
            dx: 0,
            ye: ymin,
        },
    ];
    let mut edges = npts as i64;
    let mut y = ymin;
    loop {
        for e in edge.iter_mut() {
            if y >= e.ye {
                let mut idx0 = e.idx;
                let mut idx = (idx0 + e.di) % npts;
                loop {
                    let go = edges > 0;
                    edges -= 1;
                    if !go {
                        break;
                    }
                    let ty = v[idx].1;
                    if ty > y {
                        let xs = v[idx0].0 << XY_SHIFT;
                        let xe = v[idx].0 << XY_SHIFT;
                        e.ye = ty;
                        e.dx = ((xe - xs) * 2 + (ty - y)) / (2 * (ty - y));
                        e.x = xs;
                        e.idx = idx;
                        break;
                    }
                    idx0 = idx;
                    idx = (idx + e.di) % npts;
                }
            }
        }
        if edges < 0 {
            break;
        }
        if y >= 0 {
            let (left, right) = if edge[0].x > edge[1].x {
                (1, 0)
            } else {
                (0, 1)
            };
            let mut xx1 = (edge[left].x + delta1) >> XY_SHIFT;
            let mut xx2 = (edge[right].x + delta2) >> XY_SHIFT;
            if xx2 >= 0 && xx1 < w as i64 {
                xx1 = xx1.max(0);
                xx2 = xx2.min(w as i64 - 1);
                let row = y as usize * w;
                mask[row + xx1 as usize..=row + xx2 as usize].fill(color);
            }
        }
        edge[0].x += edge[0].dx;
        edge[1].x += edge[1].dx;
        y += 1;
        if y > ymax {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hull_of_a_square_with_inner_points() {
        let pts = [(0, 0), (5, 5), (10, 0), (10, 10), (0, 10), (3, 4)];
        let hull = convex_hull(&pts);
        assert_eq!(hull.len(), 4);
    }

    #[test]
    fn axis_aligned_rect() {
        let pts = [(2, 3), (12, 3), (12, 8), (2, 8)];
        let r = min_area_rect(&pts);
        assert_eq!(r.center, (7.0, 5.5));
        let mut s = [r.size.0, r.size.1];
        s.sort_by(f32::total_cmp);
        assert_eq!(s, [5.0, 10.0]);
    }

    #[test]
    fn filled_square() {
        let mut mask = vec![0u8; 10 * 10];
        fill_convex_poly(&mut mask, 10, 10, &[(2, 2), (6, 2), (6, 6), (2, 6)], 1);
        assert_eq!(mask.iter().filter(|&&v| v == 1).count(), 25);
    }
}
