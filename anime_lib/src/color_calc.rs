use crate::palette::*;
use argmm::ArgMinMax;
use lazy_static::lazy_static;

// D65 standard illuminant refs
static REF_X: f64 = 95.047;
static REF_Y: f64 = 100.000;
static REF_Z: f64 = 108.883;
use lab::Lab;

lazy_static! {
    static ref LAB_PALETTE: [(f32, f32, f32); 256] = {
        let mut pal = [(0.0f32,0.0f32,0.0f32); 256];
        for i in 0..256 {
            let (r, g, b) = PALETTE[i];
            let lab = Lab::from_rgb(&[r, g, b]);
            pal[i] = (lab.l, lab.a, lab.b)
        }

        pal
    };

    // extra zero for simd convenience
    static ref LAB_PALETTE_FLATTENED: [f32; 1024] = {
        let mut pal = [0.0f32; 1024];
        for i in 0..256 {
            let (r, g, b) = PALETTE[i];
            let lab = Lab::from_rgb(&[r, g, b]);
            pal[i * 4] = lab.l;
            pal[i * 4 + 1] = lab.a;
            pal[i * 4 + 2] = lab.b;
            pal[i * 4 + 3] = 0.0;
        }

        pal
    };
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
pub unsafe fn closest_ansi_avx(r: u8, g: u8, b: u8) -> u8 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let lab = Lab::from_rgb(&[r, g, b]);
    let lab_arr = [lab.l, lab.a, lab.b, 0.0, lab.l, lab.a, lab.b, 0.0];
    let lab_mm = _mm256_loadu_ps(lab_arr.as_ptr() as *const f32);

    let mut res_array: [f32; 256] = [0.0; 256]; // full delta E
    let mut tmp: [f32; 8] = [0.0; 8]; // tmp array for storing intermediate values

    LAB_PALETTE_FLATTENED
        .chunks_exact(16)
        .enumerate()
        .for_each(|(i, step)| {
            let pal_a = _mm256_loadu_ps(step.as_ptr() as *const f32); // load in 8 values (l,a,b,0,l,a,b,0)
            let mut a = _mm256_sub_ps(lab_mm, pal_a); // subtract (lhs.l - rhs.l), (lhs.a - rhs.a), (lhs.b - rhs.b)
            a = _mm256_mul_ps(a, a); // raise to power of two

            let pal_b = _mm256_loadu_ps(step.as_ptr().add(8) as *const f32); // load in 8 values (l,a,b,0,l,a,b,0)
            let mut b = _mm256_sub_ps(lab_mm, pal_b); // subtract (lhs.l - rhs.l), (lhs.a - rhs.a), (lhs.b - rhs.b)
            b = _mm256_mul_ps(b, b); // raise to power of two

            _mm256_store_ps(tmp.as_mut_ptr() as *mut f32, _mm256_hadd_ps(a, b)); // add up (l + a) for every value and then store
            let start = i * 4;
            // add up (l + a) + b
            *res_array.get_unchecked_mut(start) = tmp.get_unchecked(0) + tmp.get_unchecked(1);
            *res_array.get_unchecked_mut(start + 1) = tmp.get_unchecked(4) + tmp.get_unchecked(5);
            *res_array.get_unchecked_mut(start + 2) = tmp.get_unchecked(2) + tmp.get_unchecked(3);
            *res_array.get_unchecked_mut(start + 3) = tmp.get_unchecked(6) + tmp.get_unchecked(7);
        });

    res_array.argmin().unwrap() as u8
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse")]
pub unsafe fn closest_ansi_sse(r: u8, g: u8, b: u8) -> u8 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let lab = Lab::from_rgb(&[r, g, b]);
    let lab_arr = [lab.l, lab.a, lab.b, 0.0];
    let lab_mm = _mm_loadu_ps(lab_arr.as_ptr() as *const f32);

    let mut results: [f32; 256] = [0.0; 256];
    let mut res_array: [f32; 4] = [0.0; 4];

    LAB_PALETTE_FLATTENED
        .chunks_exact(4)
        .enumerate()
        .for_each(|(i, step)| {
            let pal_mm = _mm_loadu_ps(step.as_ptr() as *const f32);
            let mut res = _mm_sub_ps(lab_mm, pal_mm);
            res = _mm_mul_ps(res, res);
            _mm_store_ps(res_array.as_mut_ptr() as *mut f32, res); // store back

            *results.get_unchecked_mut(i) = res_array.get_unchecked(0) // add up left delta E
                + res_array.get_unchecked(1)
                + res_array.get_unchecked(2);
        });

    results.argmin().unwrap() as u8
}

#[cfg(any(target_arch = "powerpc64le", target_arch = "powerpc64", target_arch = "powerpcle", target_arch = "powerpc"))]
#[target_feature(enable = "altivec")]
pub unsafe fn closest_ansi_altivec(r: u8, g: u8, b: u8) -> u8 {
    #[cfg(any(target_arch = "powerpc64le", target_arch = "powerpc64"))]
    use std::arch::powerpc64::*;
    #[cfg(any(target_arch = "powerpcle", target_arch = "powerpc"))]
    use std::arch::powerpc::*;

    let lab = Lab::from_rgb(&[r, g, b]);
    let lab_altivec = vec_ld(0, [lab.l, lab.a, lab.b, 0.0].as_ptr());
    let zero = vec_splats(0.0);

    let mut results: [f32; 256] = [0.0; 256];

    LAB_PALETTE_FLATTENED
        .chunks_exact(4)
        .enumerate()
        .for_each(|(i, step)| {
            let pal_altivec = vec_ld(0, step.as_ptr() as *const f32);
            let mut res = vec_sub(lab_altivec, pal_altivec);
            res = vec_madd(res, res, zero);

            let (r1, r2, r3, _): (f32, f32, f32, f32) = core::mem::transmute(res);

            *results.get_unchecked_mut(i) = r1 + r2 + r3; // add up left delta E
        });

    results.argmin().unwrap() as u8
}

pub fn closest_ansi_scalar(r: u8, g: u8, b: u8) -> u8 {
    let lab = Lab::from_rgb(&[r, g, b]);
    let mut results: [f32; 256] = [0.0; 256];
    for i in 0..256 {
        let (p_l, p_a, p_b) = LAB_PALETTE[i];
        results[i] = (lab.l - p_l).powi(2) + (lab.a - p_a).powi(2) + (lab.b - p_b).powi(2);
    }

    results.argmin().unwrap() as u8
}

pub fn closest_ansi(r: u8, g: u8, b: u8) -> u8 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx") {
            return unsafe { closest_ansi_avx(r, g, b) };
        } else if is_x86_feature_detected!("sse") {
            return unsafe { closest_ansi_sse(r, g, b) };
        }
    }
    #[cfg(any(target_arch = "powerpc64le", target_arch = "powerpc64"))]
    {
        if is_powerpc64_feature_detected!("altivec") {
            return unsafe { closest_ansi_altivec(r, g, b) };
        } // TODO: else if is_powerpc64_feature_detected("vsx") {}
    }
    #[cfg(any(target_arch = "powerpcle", target_arch = "powerpc"))]
    {
        if is_powerpc_feature_detected!("altivec") {
            return unsafe { closest_ansi_altivec(r, g, b) };
        }
    }

    closest_ansi_scalar(r, g, b)
    // let lab = Lab::from_rgb(&[r, g, b]);

    // LAB_PALETTE
    //     .iter()
    //     .map(|(p_l, p_a, p_b)| {
    //         (lab.l - p_l).powi(2) + (lab.a - p_a).powi(2) + (lab.b - p_b).powi(2)
    //     })
    //     .collect::<Vec<f32>>()
    //     .argmin()
    //     .unwrap() as u8
}

pub fn rgb_to_xyz(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let r = {
        let r_ = r / 255.0;
        if r_ > 0.04045 {
            ((r_ + 0.055) / 1.055).powf(2.4)
        } else {
            r_ / 12.92
        }
    } * 100.0;

    let g = {
        let g_ = g / 255.0;
        if g_ > 0.04045 {
            ((g_ + 0.055) / 1.055).powf(2.4)
        } else {
            g_ / 12.92
        }
    } * 100.0;

    let b = {
        let b_ = b / 255.0;
        if b_ > 0.04045 {
            ((b_ + 0.055) / 1.055).powf(2.4)
        } else {
            b_ / 12.92
        }
    } * 100.0;

    (
        r * 0.4124 + g * 0.3576 + b * 0.1805, // x
        r * 0.2166 + g * 0.7152 + b * 0.0722, // y
        r * 0.0193 + g * 0.1192 + b * 0.9505, // z
    )
}

pub fn xyz_to_lab(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    let x = {
        let x_ = x / REF_X;
        if x_ > 0.008856 {
            x_.powf(1.0 / 3.0)
        } else {
            (7.787 * x_) + 16.0 / 116.0
        }
    };

    let y = {
        let y_ = y / REF_Y;
        if y_ > 0.008856 {
            y_.powf(1.0 / 3.0)
        } else {
            (7.787 * y_) + 16.0 / 116.0
        }
    };

    let z = {
        let z_ = z / REF_Z;
        if z_ > 0.008856 {
            z_.powf(1.0 / 3.0)
        } else {
            (7.787 * z_) + 16.0 / 116.0
        }
    };

    (
        (116.0 * y) - 16.0, // l
        500.0 * (x - y),    // a
        200.0 * (y - z),    // b
    )
}
