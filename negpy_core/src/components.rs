//! `cv2.connectedComponents(WithStats)` for a uint8 mask, with OpenCV's labels: its Spaghetti
//! labelling numbers 4-connected components in raster order of their first pixel, and
//! 8-connected ones in raster order of their first 2x2 block. The numbering does not depend on
//! OpenCV's thread count.

pub struct Components {
    pub count: usize,
    pub labels: Vec<i32>,
    /// Per label: left, top, width, height, area.
    pub stats: Vec<[i32; 5]>,
    pub centroids: Vec<[f64; 2]>,
}

fn find(parent: &mut [u32], mut i: u32) -> u32 {
    while parent[i as usize] != i {
        let p = parent[i as usize];
        parent[i as usize] = parent[p as usize];
        i = p;
    }
    i
}

fn union(parent: &mut [u32], a: u32, b: u32) -> u32 {
    let (ra, rb) = (find(parent, a), find(parent, b));
    let (lo, hi) = (ra.min(rb), ra.max(rb));
    parent[hi as usize] = lo;
    lo
}

/// Components of the nonzero pixels of an `h` x `w` mask, `connectivity` 4 or 8.
pub fn connected_components(mask: &[u8], h: usize, w: usize, connectivity: u8) -> Components {
    let mut prov = vec![0u32; h * w];
    let mut parent: Vec<u32> = vec![0];
    for y in 0..h {
        for x in 0..w {
            if mask[y * w + x] == 0 {
                continue;
            }
            let mut neighbours = [0u32; 4];
            let mut k = 0;
            let mut push = |l: u32| {
                if l != 0 {
                    neighbours[k] = l;
                    k += 1;
                }
            };
            if x > 0 {
                push(prov[y * w + x - 1]);
            }
            if y > 0 {
                push(prov[(y - 1) * w + x]);
                if connectivity == 8 {
                    if x > 0 {
                        push(prov[(y - 1) * w + x - 1]);
                    }
                    if x + 1 < w {
                        push(prov[(y - 1) * w + x + 1]);
                    }
                }
            }
            prov[y * w + x] = if k == 0 {
                parent.push(parent.len() as u32);
                (parent.len() - 1) as u32
            } else {
                let mut l = neighbours[0];
                for &n in &neighbours[1..k] {
                    l = union(&mut parent, l, n);
                }
                find(&mut parent, l)
            };
        }
    }

    // Number the roots by the first pixel (4-connected) or 2x2 block (8-connected) of their
    // component in raster order.
    let blocks_w = w.div_ceil(2);
    let mut first = vec![usize::MAX; parent.len()];
    for y in 0..h {
        for x in 0..w {
            let l = prov[y * w + x];
            if l != 0 {
                let root = find(&mut parent, l) as usize;
                let key = if connectivity == 8 {
                    (y / 2) * blocks_w + x / 2
                } else {
                    y * w + x
                };
                first[root] = first[root].min(key);
            }
        }
    }
    let mut roots: Vec<usize> = (1..parent.len())
        .filter(|&r| first[r] != usize::MAX)
        .collect();
    roots.sort_by_key(|&r| first[r]);
    let mut final_label = vec![0i32; parent.len()];
    for (i, &r) in roots.iter().enumerate() {
        final_label[r] = i as i32 + 1;
    }

    let count = roots.len() + 1;
    let mut labels = vec![0i32; h * w];
    let mut lo = vec![[i32::MAX, i32::MAX, i32::MIN, i32::MIN, 0i32]; count];
    let mut integrals = vec![[0u64; 2]; count];
    for y in 0..h {
        for x in 0..w {
            let p = prov[y * w + x];
            let l = if p == 0 {
                0
            } else {
                final_label[find(&mut parent, p) as usize]
            };
            labels[y * w + x] = l;
            let s = &mut lo[l as usize];
            let (xi, yi) = (x as i32, y as i32);
            s[0] = s[0].min(xi);
            s[1] = s[1].min(yi);
            s[2] = s[2].max(xi);
            s[3] = s[3].max(yi);
            s[4] += 1;
            integrals[l as usize][0] += x as u64;
            integrals[l as usize][1] += y as u64;
        }
    }
    let mut stats = Vec::with_capacity(count);
    let mut centroids = Vec::with_capacity(count);
    for (s, i) in lo.iter().zip(&integrals) {
        let area = s[4] as f64;
        if area > 0.0 {
            stats.push([s[0], s[1], s[2] - s[0] + 1, s[3] - s[1] + 1, s[4]]);
            centroids.push([i[0] as f64 / area, i[1] as f64 / area]);
        } else {
            stats.push([-1, s[1], 0, 0, 0]);
            centroids.push([f64::NAN, f64::NAN]);
        }
    }
    Components {
        count,
        labels,
        stats,
        centroids,
    }
}
