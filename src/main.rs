extern crate image;
extern crate num_complex;
extern crate time;

use image::ImageBuffer;
use std::collections::{HashMap, LinkedList};
use vulkano::instance::Instance;
use vulkano::instance::InstanceExtensions;

const KI: i32 = 10;
const KC: f32 = 25.0;
const KG: u32 = 4;

struct Tuple {
    f: image::Rgb<u8>,
    b: image::Rgb<u8>,
    sigmaf: f32,
    sigmab: f32,
}

#[derive(Clone)]
struct Ftuple {
    f: image::Rgb<u8>,
    b: image::Rgb<u8>,
    alphar: f32,
    confidence: f32,
}

impl Ftuple {
    fn new() -> Ftuple {
        return Ftuple {
            f: image::Rgb([0, 0, 0]),
            b: image::Rgb([0, 0, 0]),
            alphar: 0.0,
            confidence: 0.0,
        };
    }
}

fn distance_point(a: (u32, u32), b: (u32, u32)) -> i32 {
    return (a.0 as i32 - b.0 as i32).pow(2) + (a.1 as i32 - b.1 as i32).pow(2);
}

fn distence_color(a: image::Rgb<u8>, b: image::Rgb<u8>) -> f32 {
    return ((a[0] as i32 - b[0] as i32).pow(2)
        + (a[1] as i32 - b[1] as i32).pow(2)
        + (a[2] as i32 - b[2] as i32).pow(2)) as f32;
}

fn sample(
    ori_state: &ImageBuffer<image::Rgb<u8>, Vec<u8>>,
    tri_state: &ImageBuffer<image::Luma<u8>, std::vec::Vec<u8>>,
    front: &mut LinkedList<LinkedList<(u32, u32)>>,
    back: &mut LinkedList<LinkedList<(u32, u32)>>,
    u_t: &LinkedList<(u32, u32)>,
) {
    let a = (360 / KG) as f32;
    let b = 1.7 * a / 9.0;
    let (w, h) = ori_state.dimensions();
    for p in u_t {
        let (x, y) = *p;
        let angle = (x + y) as f32 * b % a;
        let mut b_pts: LinkedList<(u32, u32)> = LinkedList::new();
        let mut f_pts: LinkedList<(u32, u32)> = LinkedList::new();
        for i in 0..KG {
            let mut f1 = false;
            let mut f2 = false;

            let z = (angle + i as f32 * a) / 180.0 * 3.1415926;
            let ex = z.sin();
            let ey = z.cos();
            let step = (1.0 / (ex.abs() + 1e-10)).min(1.0 / (ey.abs() + 1e-10));
            let mut t = 0.0;
            loop {
                let p = x + (ex * t + 0.5) as u32;
                let q = y + (ey * t + 0.5) as u32;
                if p >= w || q >= h {
                    break;
                }

                let gray = tri_state.get_pixel(p, q);
                if !f1 && gray[0] < 50 {
                    b_pts.push_back((p, q));
                    f1 = true;
                } else if !f2 && gray[0] > 200 {
                    f_pts.push_back((p, q));
                    f2 = true;
                } else if f1 && f2 {
                    break;
                }
                t += step;
            }
        }

        front.push_back(f_pts);
        back.push_back(b_pts);
    }
}

fn comalpha(c: image::Rgb<u8>, f: image::Rgb<u8>, b: image::Rgb<u8>) -> f32 {
    let alpha = ((c[0] as f32 - b[0] as f32) * (f[0] as f32 - b[0] as f32)
        + (c[1] as f32 - b[1] as f32) * (f[1] as f32 - b[1] as f32)
        + (c[2] as f32 - b[2] as f32) * (f[2] as f32 - b[2] as f32))
        / ((f[0] as f32 - b[0] as f32) * (f[0] as f32 - b[0] as f32)
            + (f[1] as f32 - b[1] as f32) * (f[1] as f32 - b[1] as f32)
            + (f[2] as f32 - b[2] as f32) * (f[2] as f32 - b[2] as f32)
            + 0.0000001);
    return (0.0f32).max(alpha).min(1.0);
}

fn d_p(s: (u32, u32), d: (u32, u32)) -> f32 {
    return (((s.0 as i64 - d.0 as i64) * (s.0 as i64 - d.0 as i64)
        + (s.1 as i64 - d.1 as i64) * (s.1 as i64 - d.1 as i64)) as f32)
        .sqrt();
}

fn m_p(
    ori_state: &ImageBuffer<image::Rgb<u8>, Vec<u8>>,
    (i, j): (u32, u32),
    f: image::Rgb<u8>,
    b: image::Rgb<u8>,
) -> f32 {
    let c = ori_state[(i, j)];

    let alpha = comalpha(c, f, b);

    let result = ((c[0] as f32 - alpha * f[0] as f32 - (1.0 - alpha) * b[0] as f32)
        * (c[0] as f32 - alpha * f[0] as f32 - (1.0 - alpha) * b[0] as f32)
        + (c[1] as f32 - alpha * f[1] as f32 - (1.0 - alpha) * b[1] as f32)
            * (c[1] as f32 - alpha * f[1] as f32 - (1.0 - alpha) * b[1] as f32)
        + (c[2] as f32 - alpha * f[2] as f32 - (1.0 - alpha) * b[2] as f32)
            * (c[2] as f32 - alpha * f[2] as f32 - (1.0 - alpha) * b[2] as f32))
        .sqrt();
    return result / 255.0;
}

fn n_p(
    ori_state: &ImageBuffer<image::Rgb<u8>, Vec<u8>>,
    (i, j): (u32, u32),
    f: image::Rgb<u8>,
    b: image::Rgb<u8>,
) -> f32 {
    let (width, height) = ori_state.dimensions();
    let i1 = (1).max(i) - 1;
    let i2 = (i + 1).min(width - 1);
    let j1 = (1).max(j) - 1;
    let j2 = (j + 1).min(height - 1);

    let mut result = 0.0;

    for k in i1..=i2 {
        for l in j1..=j2 {
            let m = m_p(ori_state, (k, l), f, b);
            result += m * m;
        }
    }

    return result;
}

fn a_p(
    ori_state: &ImageBuffer<image::Rgb<u8>, Vec<u8>>,
    (i, j): (u32, u32),
    pf: f32,
    f: image::Rgb<u8>,
    b: image::Rgb<u8>,
) -> f32 {
    let c = ori_state[(i, j)];

    let alpha = comalpha(c, f, b);

    return pf + (1.0 - 2.0 * pf) * alpha;
}

fn g_p(
    ori_state: &ImageBuffer<image::Rgb<u8>, Vec<u8>>,
    p: (u32, u32),
    fp: (u32, u32),
    bp: (u32, u32),
    dpf: f32,
    pf: f32,
) -> f32 {
    let f = ori_state[fp];
    let b = ori_state[bp];

    let tn = n_p(ori_state, p, f, b).powf(3.0);
    let ta = a_p(ori_state, p, pf, f, b).powf(2.0);
    let tf = dpf;
    let tb = d_p(p, bp).powf(4.0);

    return tn * ta * tf * tb;
}

fn e_p(
    ori_state: &ImageBuffer<image::Rgb<u8>, Vec<u8>>,
    (i1, j1): (u32, u32),
    (i2, j2): (u32, u32),
) -> f32 {
    let ci = i2 as f32 - i1 as f32;
    let cj = j2 as f32 - j1 as f32;
    let z = ((ci * ci + cj * cj) as f32).sqrt();

    let ei = ci / (z + 0.0000001);
    let ej = cj / (z + 0.0000001);

    let stepinc = (1.0 / (ei.abs() + 1e-10)).min(1.0 / (ej.abs() + 1e-10));

    let mut result = 0.0;

    let mut pre = ori_state[(i1, j1)];

    let mut ti = i1;
    let mut tj = j1;
    let mut t = 1.0;
    loop {
        let inci = ei * t;
        let incj = ej * t;
        let i = i1 + (inci + 0.5) as u32;
        let j = j1 + (incj + 0.5) as u32;

        let mut z = 1.0;

        let cur = ori_state[(i, j)];

        if ti > j && tj == j {
            z = ej;
        } else if ti == i && tj > j {
            z = ei;
        }

        result += distence_color(cur, pre) as f32 * z;
        pre = cur;

        ti = i;
        tj = j;

        if ci.abs() >= inci.abs() || cj.abs() >= incj.abs() {
            break;
        }
        t += stepinc;
    }
    return result;
}

fn pf_p(
    ori_state: &ImageBuffer<image::Rgb<u8>, Vec<u8>>,
    p: (u32, u32),
    f: &LinkedList<(u32, u32)>,
    b: &LinkedList<(u32, u32)>,
) -> f32 {
    let mut fmin = 1e10;
    for p1 in f {
        let fp = e_p(ori_state, p, *p1);
        if fp < fmin {
            fmin = fp;
        }
    }
    let mut bmin = 1e10;
    for p1 in b {
        let bp = e_p(ori_state, p, *p1);
        if bp < bmin {
            bmin = bp;
        }
    }
    return bmin / (fmin + bmin + 1e-10);
}

fn expand_known(
    ori_state: &ImageBuffer<image::Rgb<u8>, Vec<u8>>,
    tri_state: &mut ImageBuffer<image::Luma<u8>, std::vec::Vec<u8>>,
) -> LinkedList<(u32, u32)> {
    let (imgx, imgy) = tri_state.dimensions();
    let mut u_t: LinkedList<(u32, u32)> = LinkedList::new();
    let mut list: LinkedList<(u32, u32, image::Luma<u8>)> = LinkedList::new();
    for x in 0..imgx {
        for y in 0..imgy {
            let pixel = tri_state.get_pixel(x, y);
            let tri = pixel[0];
            let mut ok = false;
            if tri > 0 && tri < 255 {
                let cur_color = ori_state[(x, y)];
                'outer: for i in -KI..KI {
                    for j in -KI..KI {
                        let w = (x as i32 + i) as u32;
                        let h = (y as i32 + j) as u32;
                        if w > 0 && w < imgx && h > 0 && h < imgy {
                            let tri_color = tri_state[(w, h)];
                            if tri_color[0] == 0 || tri_color[0] == 255 {
                                let color = ori_state[(w, h)];
                                if distance_point((x, y), (w, h)) < KI
                                    && distence_color(color, cur_color) < KC
                                {
                                    list.push_back((x, y, tri_color));
                                    ok = true;
                                    break 'outer;
                                }
                            }
                        }
                    }
                }
                if !ok {
                    u_t.push_back((x, y));
                }
            }
        }
    }
    for (x, y, pixel) in list {
        tri_state.put_pixel(x, y, pixel);
    }
    return u_t;
}

fn sigma2(ori_state: &ImageBuffer<image::Rgb<u8>, Vec<u8>>, (xi, yj): (u32, u32)) -> f32 {
    let pc = ori_state[(xi, yj)];
    let (width, height) = ori_state.dimensions();

    let i1 = (2).max(xi) - 2;
    let i2 = (xi + 2).min(width - 1);
    let j1 = (2).max(yj) - 2;
    let j2 = (yj + 2).min(height - 1);

    let mut result = 0.0;
    let mut num = 0.0;
    for i in i1..=i2 {
        for j in j1..=j2 {
            let temp = ori_state[(i, j)];
            result += distence_color(pc, temp) as f32;
            num += 1.0;
        }
    }

    return result / (num + 1e-10);
}

fn gathering(
    ori_state: &ImageBuffer<image::Rgb<u8>, Vec<u8>>,
    tri_state: &ImageBuffer<image::Luma<u8>, std::vec::Vec<u8>>,
    u_t: &LinkedList<(u32, u32)>,
) -> HashMap<(u32, u32), Tuple> {
    let mut front: LinkedList<LinkedList<(u32, u32)>> = LinkedList::new();
    let mut back: LinkedList<LinkedList<(u32, u32)>> = LinkedList::new();
    let start = time::now();
    sample(ori_state, &tri_state, &mut front, &mut back, &u_t);
    let end = time::now();
    println!("Sample: {:?}", (end - start).num_milliseconds());
    let mut unknown_index: HashMap<(u32, u32), Tuple> = HashMap::new();

    for ((p, fm), bm) in u_t.into_iter().zip(front).zip(back) {
        let pfp = pf_p(ori_state, *p, &fm, &bm);
        let mut gmin = 1.0e10;

        let mut flag = false;
        let mut tf = (0, 0);
        let mut tb = (0, 0);

        for it1 in fm {
            let dpf = d_p(*p, it1);
            for it2 in &bm {
                let gp = g_p(ori_state, *p, it1, *it2, dpf, pfp);
                if gp < gmin {
                    gmin = gp;
                    tf = it1;
                    tb = *it2;
                    flag = true;
                }
            }
        }

        if flag {
            let st = Tuple {
                f: ori_state[tf],
                b: ori_state[tb],
                sigmaf: sigma2(ori_state, tf),
                sigmab: sigma2(ori_state, tb),
            };
            unknown_index.insert(*p, st);
        }
    }
    return unknown_index;
}

fn refine_sample(
    ori_state: &ImageBuffer<image::Rgb<u8>, Vec<u8>>,
    tri_state: &ImageBuffer<image::Luma<u8>, std::vec::Vec<u8>>,
    u_t: &LinkedList<(u32, u32)>,
    unknown_index: &HashMap<(u32, u32), Tuple>,
) -> (Vec<Ftuple>, Vec<Vec<u8>>) {
    let (width, height) = ori_state.dimensions();
    let mut ftuples: Vec<Ftuple> = vec![Ftuple::new(); (width * height + 1) as usize];
    let mut alpha = vec![vec![0 as u8; height as usize]; width as usize];
    for i in 0..width {
        for j in 0..height {
            let c = ori_state[(i, j)];
            let indexf = (i * height + j) as usize;
            let gray = tri_state.get_pixel(i, j);
            if gray[0] == 0 {
                ftuples[indexf].f = c;
                ftuples[indexf].b = c;
                ftuples[indexf].alphar = 0.0;
                ftuples[indexf].confidence = 1.0;
                alpha[i as usize][j as usize] = 0;
            } else if gray[0] == 255 {
                ftuples[indexf].f = c;
                ftuples[indexf].b = c;
                ftuples[indexf].alphar = 1.0;
                ftuples[indexf].confidence = 1.0;
                alpha[i as usize][j as usize] = 255;
            }
        }
    }
    for p in u_t {
        let (xi, yj) = *p;
        let i1 = (5).max(xi) - 5;
        let i2 = (xi + 5).min(width - 1);
        let j1 = (5).max(yj) - 5;
        let j2 = (yj + 5).min(height - 1);

        let mut minvalue = vec![1e10, 1e10, 1e10];
        let mut p = vec![(0, 0); 3];
        let mut num = 0;
        for k in i1..=i2 {
            for l in j1..=j2 {
                let temp = tri_state.get_pixel(k, l);
                if temp[0] == 0 || temp[0] == 255 {
                    continue;
                }
                if let Some(t) = unknown_index.get(&(k, l)) {
                    let m = m_p(ori_state, (xi, yj), t.f, t.b);
                    if m > minvalue[2] {
                        continue;
                    }

                    if m < minvalue[0] {
                        minvalue[2] = minvalue[1];
                        p[2] = p[1];

                        minvalue[1] = minvalue[0];
                        p[1] = p[0];

                        minvalue[0] = m;
                        p[0].0 = k;
                        p[0].1 = l;

                        num += 1;
                    } else if m < minvalue[1] {
                        minvalue[2] = minvalue[1];
                        p[2] = p[1];

                        minvalue[1] = m;
                        p[1].0 = k;
                        p[1].1 = l;

                        num += 1;
                    } else if m < minvalue[2] {
                        minvalue[2] = m;
                        p[2].0 = k;
                        p[2].1 = l;

                        num += 1;
                    }
                }
            }
        }

        num = (num).min(3);

        let mut fb = 0.0;
        let mut fg = 0.0;
        let mut fr = 0.0;
        let mut bb = 0.0;
        let mut bg = 0.0;
        let mut br = 0.0;
        let mut sf = 0.0;
        let mut sb = 0.0;

        for k in 0..num {
            if let Some(t) = unknown_index.get(&p[k]) {
                fb += t.f[0] as f32;
                fg += t.f[1] as f32;
                fr += t.f[2] as f32;
                bb += t.b[0] as f32;
                bg += t.b[1] as f32;
                br += t.b[2] as f32;
                sf += t.sigmaf;
                sb += t.sigmab;
            }
        }

        fb /= num as f32 + 1e-10;
        fg /= num as f32 + 1e-10;
        fr /= num as f32 + 1e-10;
        bb /= num as f32 + 1e-10;
        bg /= num as f32 + 1e-10;
        br /= num as f32 + 1e-10;
        sf /= num as f32 + 1e-10;
        sb /= num as f32 + 1e-10;

        let mut fc = image::Rgb([fb as u8, fg as u8, fr as u8]);
        let mut bc = image::Rgb([bb as u8, bg as u8, br as u8]);
        let pc = ori_state[(xi, yj)];
        let df = distence_color(pc, fc);
        let db = distence_color(pc, bc);
        let tf = fc;
        let tb = bc;

        let index = (xi * height + yj) as usize;
        if df < sf {
            fc = pc;
        }
        if db < sb {
            bc = pc;
        }
        if fc[0] == bc[0] && fc[1] == bc[1] && fc[2] == bc[2] {
            ftuples[index].confidence = 0.00000001;
        } else {
            ftuples[index].confidence = (-10.0 * m_p(ori_state, (xi, yj), tf, tb)).exp();
        }

        ftuples[index].f = fc;
        ftuples[index].b = bc;

        ftuples[index].alphar = (0.0f32).max(comalpha(pc, fc, bc)).min(1.0);
        //cvSet2D(matte, xi, yj, ScalarAll(ftuples[index].alphar * 255));
    }
    return (ftuples, alpha);
}

fn local_smooth(
    ori_state: &ImageBuffer<image::Rgb<u8>, Vec<u8>>,
    tri_state: &ImageBuffer<image::Luma<u8>, std::vec::Vec<u8>>,
    u_t: &LinkedList<(u32, u32)>,
    ftuples: &Vec<Ftuple>,
    alpha: &mut Vec<Vec<u8>>,
) {
    let (width, height) = ori_state.dimensions();
    let sig2: f32 = 100.0 / (9.0 * 3.1415926);
    let r = 3.0 * sig2.sqrt();
    for it in u_t {
        let (xi, yj) = *it;
        let i1 = (xi as f32 - r).max(0.0) as u32;
        let i2 = (xi + r as u32).min(width - 1);
        let j1 = (yj as f32 - r).max(0.0) as u32;
        let j2 = (yj + r as u32).min(height - 1);

        let indexp = (xi * height + yj) as usize;
        let ptuple = &ftuples[indexp];

        let mut wcfsumup = image::Rgb([0.0, 0.0, 0.0]);
        let mut wcbsumup = image::Rgb([0.0, 0.0, 0.0]);
        let mut wcfsumdown = 0.0;
        let mut wcbsumdown = 0.0;
        let mut wfbsumup = 0.0;
        let mut wfbsundown = 0.0;
        let mut wasumup = 0.0;
        let mut wasumdown = 0.0;

        for k in i1..=i2 {
            for l in j1..=j2 {
                let indexq = (k * height + l) as usize;
                let qtuple = &ftuples[indexq];

                let d = d_p((xi, yj), (k, l));

                if d > r {
                    continue;
                }

                let wc;
                if d == 0.0 {
                    wc = (-(d * d) / sig2).exp() * qtuple.confidence;
                } else {
                    wc = (-(d * d) / sig2).exp()
                        * qtuple.confidence
                        * (qtuple.alphar - ptuple.alphar).abs();
                }
                wcfsumdown += wc * qtuple.alphar;
                wcbsumdown += wc * (1.0 - qtuple.alphar);

                wcfsumup[0] += wc * qtuple.alphar * qtuple.f[0] as f32;
                wcfsumup[1] += wc * qtuple.alphar * qtuple.f[1] as f32;
                wcfsumup[2] += wc * qtuple.alphar * qtuple.f[2] as f32;

                wcbsumup[0] += wc * (1.0 - qtuple.alphar) * qtuple.b[0] as f32;
                wcbsumup[1] += wc * (1.0 - qtuple.alphar) * qtuple.b[1] as f32;
                wcbsumup[2] += wc * (1.0 - qtuple.alphar) * qtuple.b[2] as f32;

                let wfb = qtuple.confidence * qtuple.alphar * (1.0 - qtuple.alphar);
                wfbsundown += wfb;
                wfbsumup += wfb * distence_color(qtuple.f, qtuple.b).sqrt();

                let mut delta = 0.0;
                let temp = tri_state.get_pixel(k, l);
                if temp[0] == 0 || temp[0] == 255 {
                    delta = 1.0;
                }
                let wa = qtuple.confidence * (-(d * d) / sig2).exp() + delta;
                wasumdown += wa;
                wasumup += wa * qtuple.alphar;
            }
        }

        let cp = ori_state[(xi, yj)];
        let mut fp = image::Rgb([0, 0, 0]);
        let mut bp = image::Rgb([0, 0, 0]);

        bp[0] = (255.0f32).min(wcbsumup[0] / (wcbsumdown + 1e-200)).max(0.0) as u8;
        bp[1] = (255.0f32).min(wcbsumup[1] / (wcbsumdown + 1e-200)).max(0.0) as u8;
        bp[2] = (255.0f32).min(wcbsumup[2] / (wcbsumdown + 1e-200)).max(0.0) as u8;

        fp[0] = (255.0f32).min(wcfsumup[0] / (wcfsumdown + 1e-200)).max(0.0) as u8;
        fp[1] = (255.0f32).min(wcfsumup[1] / (wcfsumdown + 1e-200)).max(0.0) as u8;
        fp[2] = (255.0f32).min(wcfsumup[2] / (wcfsumdown + 1e-200)).max(0.0) as u8;

        //double tempalpha = comalpha(cp, fp, bp);
        let dfb = wfbsumup / (wfbsundown + 1e-200);

        let conp = (distence_color(fp, bp).sqrt() / dfb).min(1.0)
            * (-10.0 * m_p(ori_state, (xi, yj), fp, bp)).exp();
        let alp = wasumup / (wasumdown + 1e-200);

        let alpha_t = conp * comalpha(cp, fp, bp) + (1.0 - conp) * alp.min(1.0).max(0.0);

        alpha[xi as usize][yj as usize] = (alpha_t * 255.0) as u8;
    }
}

fn get_matte(alpha: Vec<Vec<u8>>) -> ImageBuffer<image::Luma<u8>, Vec<u8>> {
    let h = alpha[0].len();
    let w = alpha.len();
    let mut img: ImageBuffer<image::Luma<u8>, Vec<u8>> = ImageBuffer::new(w as u32, h as u32);
    for i in 0..w {
        for j in 0..h {
            img.put_pixel(i as u32, j as u32, image::Luma([alpha[i][j]]))
        }
    }
    return img;
}

fn main() {
    let instance =
        Instance::new(None, &InstanceExtensions::none(), None).expect("failed to create instance");
    println!("{:?}", instance);
    let ori = image::open("input.png").unwrap();
    let tri = image::open("trimap.png").unwrap();
    let t = -0.5;
    let ori = ori.to_rgb();
    let mut tri_extend = tri.to_luma();
    println!("{:?}", t as u32);
    let start = time::now();
    let u_t = expand_known(&ori, &mut tri_extend);
    let end = time::now();
    println!("expand_known: {:?}", (end - start).num_milliseconds());
    let start = time::now();
    let unknown_index = gathering(&ori, &tri_extend, &u_t);
    let end = time::now();
    println!("gathering: {:?}", (end - start).num_milliseconds());
    let start = time::now();
    let (ftuples, mut alpha) = refine_sample(&ori, &tri_extend, &u_t, &unknown_index);
    let end = time::now();
    println!("refineSample: {:?}", (end - start).num_milliseconds());
    let start = time::now();
    local_smooth(&ori, &tri_extend, &u_t, &ftuples, &mut alpha);
    let end = time::now();
    println!("localSmooth: {:?}", (end - start).num_milliseconds());
    let start = time::now();
    let out = get_matte(alpha);
    let end = time::now();
    println!("getMatte: {:?}", (end - start).num_milliseconds());
    out.save("out.png").unwrap();

    // let tri_extend = image::open("triExtend.png").unwrap();
}
