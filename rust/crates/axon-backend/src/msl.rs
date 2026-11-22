//! Metal kernel sources, extracted VERBATIM from `log/bifrost-mlx/kernel_qgemv.py`
//! by `log/bifrost-mlx/export_msl.py` — do not edit by hand.
//!
//! Placeholders (`G_PLACEHOLDER`, `EPS_PLACEHOLDER`, `HD_CONST`,
//! `LOG2_THETA`, `%%FATTN_LOOP%%`) are substituted at registration time in
//! `kernels.rs`, exactly like the Python `make_*` factories do.


pub const V6: &str = r##"
    uint pair = threadgroup_position_in_grid.x;
    uint lane = thread_index_in_threadgroup;
    uint K2 = q_shape[1];
    uint NG = sm_shape[1];
    uint r0 = pair * 2;
    const device uchar* qrowA = q + (size_t)r0 * K2;
    const device uchar* qrowB = qrowA + K2;
    float accA = 0.0f;
    float accB = 0.0f;
    for (uint b8 = lane * 8; b8 < K2; b8 += 256) {
        uchar4 p0a = *(const device uchar4*)(&qrowA[b8]);
        uchar4 p1a = *(const device uchar4*)(&qrowA[b8 + 4]);
        uchar4 p0b = *(const device uchar4*)(&qrowB[b8]);
        uchar4 p1b = *(const device uchar4*)(&qrowB[b8 + 4]);
        uint kk = b8 * 2;
        uint gi = kk / G_PLACEHOLDER;
        half2 smA = *(const device half2*)(&sm[(r0 * NG + gi) * 2]);
        half2 smB = *(const device half2*)(&sm[((r0 + 1) * NG + gi) * 2]);
        float sA = (float)smA.x;
        float mA = (float)smA.y;
        float sB = (float)smB.x;
        float mB = (float)smB.y;
        half4 xa = *(const device half4*)(&x[kk]);
        half4 xb = *(const device half4*)(&x[kk + 4]);
        half4 xc = *(const device half4*)(&x[kk + 8]);
        half4 xd = *(const device half4*)(&x[kk + 12]);
        accA += (float)xa.x * ((float)(p0a[0] & 0xF) * sA + mA);
        accA += (float)xa.y * ((float)(p0a[0] >> 4) * sA + mA);
        accA += (float)xa.z * ((float)(p0a[1] & 0xF) * sA + mA);
        accA += (float)xa.w * ((float)(p0a[1] >> 4) * sA + mA);
        accA += (float)xb.x * ((float)(p0a[2] & 0xF) * sA + mA);
        accA += (float)xb.y * ((float)(p0a[2] >> 4) * sA + mA);
        accA += (float)xb.z * ((float)(p0a[3] & 0xF) * sA + mA);
        accA += (float)xb.w * ((float)(p0a[3] >> 4) * sA + mA);
        accA += (float)xc.x * ((float)(p1a[0] & 0xF) * sA + mA);
        accA += (float)xc.y * ((float)(p1a[0] >> 4) * sA + mA);
        accA += (float)xc.z * ((float)(p1a[1] & 0xF) * sA + mA);
        accA += (float)xc.w * ((float)(p1a[1] >> 4) * sA + mA);
        accA += (float)xd.x * ((float)(p1a[2] & 0xF) * sA + mA);
        accA += (float)xd.y * ((float)(p1a[2] >> 4) * sA + mA);
        accA += (float)xd.z * ((float)(p1a[3] & 0xF) * sA + mA);
        accA += (float)xd.w * ((float)(p1a[3] >> 4) * sA + mA);
        accB += (float)xa.x * ((float)(p0b[0] & 0xF) * sB + mB);
        accB += (float)xa.y * ((float)(p0b[0] >> 4) * sB + mB);
        accB += (float)xa.z * ((float)(p0b[1] & 0xF) * sB + mB);
        accB += (float)xa.w * ((float)(p0b[1] >> 4) * sB + mB);
        accB += (float)xb.x * ((float)(p0b[2] & 0xF) * sB + mB);
        accB += (float)xb.y * ((float)(p0b[2] >> 4) * sB + mB);
        accB += (float)xb.z * ((float)(p0b[3] & 0xF) * sB + mB);
        accB += (float)xb.w * ((float)(p0b[3] >> 4) * sB + mB);
        accB += (float)xc.x * ((float)(p1b[0] & 0xF) * sB + mB);
        accB += (float)xc.y * ((float)(p1b[0] >> 4) * sB + mB);
        accB += (float)xc.z * ((float)(p1b[1] & 0xF) * sB + mB);
        accB += (float)xc.w * ((float)(p1b[1] >> 4) * sB + mB);
        accB += (float)xd.x * ((float)(p1b[2] & 0xF) * sB + mB);
        accB += (float)xd.y * ((float)(p1b[2] >> 4) * sB + mB);
        accB += (float)xd.z * ((float)(p1b[3] & 0xF) * sB + mB);
        accB += (float)xd.w * ((float)(p1b[3] >> 4) * sB + mB);
    }
    accA = simd_sum(accA);
    accB = simd_sum(accB);
    if (lane == 0) {
        out[r0] = (float16_t)accA;
    }
    if (lane == 1) {
        out[r0 + 1] = (float16_t)accB;
    }
"##;

pub const V6R: &str = r##"
    uint pair = threadgroup_position_in_grid.x;
    uint lane = thread_index_in_threadgroup;
    uint K2 = q_shape[1];
    uint NG = sm_shape[1];
    uint r0 = pair * 2;
    const device uchar* qrowA = q + (size_t)r0 * K2;
    const device uchar* qrowB = qrowA + K2;
    float accA = 0.0f;
    float accB = 0.0f;
    for (uint b8 = lane * 8; b8 < K2; b8 += 256) {
        uchar4 p0a = *(const device uchar4*)(&qrowA[b8]);
        uchar4 p1a = *(const device uchar4*)(&qrowA[b8 + 4]);
        uchar4 p0b = *(const device uchar4*)(&qrowB[b8]);
        uchar4 p1b = *(const device uchar4*)(&qrowB[b8 + 4]);
        uint kk = b8 * 2;
        uint gi = kk / G_PLACEHOLDER;
        half2 smA = *(const device half2*)(&sm[(r0 * NG + gi) * 2]);
        half2 smB = *(const device half2*)(&sm[((r0 + 1) * NG + gi) * 2]);
        float sA = (float)smA.x;
        float mA = (float)smA.y;
        float sB = (float)smB.x;
        float mB = (float)smB.y;
        half4 xa = *(const device half4*)(&x[kk]);
        half4 xb = *(const device half4*)(&x[kk + 4]);
        half4 xc = *(const device half4*)(&x[kk + 8]);
        half4 xd = *(const device half4*)(&x[kk + 12]);
        accA += (float)xa.x * ((float)(p0a[0] & 0xF) * sA + mA);
        accA += (float)xa.y * ((float)(p0a[0] >> 4) * sA + mA);
        accA += (float)xa.z * ((float)(p0a[1] & 0xF) * sA + mA);
        accA += (float)xa.w * ((float)(p0a[1] >> 4) * sA + mA);
        accA += (float)xb.x * ((float)(p0a[2] & 0xF) * sA + mA);
        accA += (float)xb.y * ((float)(p0a[2] >> 4) * sA + mA);
        accA += (float)xb.z * ((float)(p0a[3] & 0xF) * sA + mA);
        accA += (float)xb.w * ((float)(p0a[3] >> 4) * sA + mA);
        accA += (float)xc.x * ((float)(p1a[0] & 0xF) * sA + mA);
        accA += (float)xc.y * ((float)(p1a[0] >> 4) * sA + mA);
        accA += (float)xc.z * ((float)(p1a[1] & 0xF) * sA + mA);
        accA += (float)xc.w * ((float)(p1a[1] >> 4) * sA + mA);
        accA += (float)xd.x * ((float)(p1a[2] & 0xF) * sA + mA);
        accA += (float)xd.y * ((float)(p1a[2] >> 4) * sA + mA);
        accA += (float)xd.z * ((float)(p1a[3] & 0xF) * sA + mA);
        accA += (float)xd.w * ((float)(p1a[3] >> 4) * sA + mA);
        accB += (float)xa.x * ((float)(p0b[0] & 0xF) * sB + mB);
        accB += (float)xa.y * ((float)(p0b[0] >> 4) * sB + mB);
        accB += (float)xa.z * ((float)(p0b[1] & 0xF) * sB + mB);
        accB += (float)xa.w * ((float)(p0b[1] >> 4) * sB + mB);
        accB += (float)xb.x * ((float)(p0b[2] & 0xF) * sB + mB);
        accB += (float)xb.y * ((float)(p0b[2] >> 4) * sB + mB);
        accB += (float)xb.z * ((float)(p0b[3] & 0xF) * sB + mB);
        accB += (float)xb.w * ((float)(p0b[3] >> 4) * sB + mB);
        accB += (float)xc.x * ((float)(p1b[0] & 0xF) * sB + mB);
        accB += (float)xc.y * ((float)(p1b[0] >> 4) * sB + mB);
        accB += (float)xc.z * ((float)(p1b[1] & 0xF) * sB + mB);
        accB += (float)xc.w * ((float)(p1b[1] >> 4) * sB + mB);
        accB += (float)xd.x * ((float)(p1b[2] & 0xF) * sB + mB);
        accB += (float)xd.y * ((float)(p1b[2] >> 4) * sB + mB);
        accB += (float)xd.z * ((float)(p1b[3] & 0xF) * sB + mB);
        accB += (float)xd.w * ((float)(p1b[3] >> 4) * sB + mB);
    }
    accA = simd_sum(accA);
    accB = simd_sum(accB);
    if (lane == 0) {
        out[r0] = (float16_t)(accA + (float)res[r0]);
    }
    if (lane == 1) {
        out[r0 + 1] = (float16_t)(accB + (float)res[r0 + 1]);
    }
"##;

pub const V7B: &str = r##"
    uint pair = threadgroup_position_in_grid.x;
    uint lane = thread_index_in_threadgroup;
    uint K = x_shape[0];
    uint K2 = q_shape[1];
    uint NG = sm_shape[1];
    uint r0 = pair * 2;
    const device uchar* qrowA = q + (size_t)r0 * K2;
    const device uchar* qrowB = qrowA + K2;
    float ssq = 0.0f;
    float accA = 0.0f;
    float accB = 0.0f;
    for (uint b8 = lane * 8; b8 < K2; b8 += 256) {
        uchar4 p0a = *(const device uchar4*)(&qrowA[b8]);
        uchar4 p1a = *(const device uchar4*)(&qrowA[b8 + 4]);
        uchar4 p0b = *(const device uchar4*)(&qrowB[b8]);
        uchar4 p1b = *(const device uchar4*)(&qrowB[b8 + 4]);
        uint kk = b8 * 2;
        uint gi = kk / G_PLACEHOLDER;
        half2 smA = *(const device half2*)(&sm[(r0 * NG + gi) * 2]);
        half2 smB = *(const device half2*)(&sm[((r0 + 1) * NG + gi) * 2]);
        float sA = (float)smA.x;
        float mA = (float)smA.y;
        float sB = (float)smB.x;
        float mB = (float)smB.y;
        half4 xa = *(const device half4*)(&x[kk]);
        half4 xb = *(const device half4*)(&x[kk + 4]);
        half4 xc = *(const device half4*)(&x[kk + 8]);
        half4 xd = *(const device half4*)(&x[kk + 12]);
        ssq += (float)xa.x * (float)xa.x + (float)xa.y * (float)xa.y + (float)xa.z * (float)xa.z + (float)xa.w * (float)xa.w
             + (float)xb.x * (float)xb.x + (float)xb.y * (float)xb.y + (float)xb.z * (float)xb.z + (float)xb.w * (float)xb.w
             + (float)xc.x * (float)xc.x + (float)xc.y * (float)xc.y + (float)xc.z * (float)xc.z + (float)xc.w * (float)xc.w
             + (float)xd.x * (float)xd.x + (float)xd.y * (float)xd.y + (float)xd.z * (float)xd.z + (float)xd.w * (float)xd.w;
        accA += (float)xa.x * ((float)(p0a[0] & 0xF) * sA + mA);
        accA += (float)xa.y * ((float)(p0a[0] >> 4) * sA + mA);
        accA += (float)xa.z * ((float)(p0a[1] & 0xF) * sA + mA);
        accA += (float)xa.w * ((float)(p0a[1] >> 4) * sA + mA);
        accA += (float)xb.x * ((float)(p0a[2] & 0xF) * sA + mA);
        accA += (float)xb.y * ((float)(p0a[2] >> 4) * sA + mA);
        accA += (float)xb.z * ((float)(p0a[3] & 0xF) * sA + mA);
        accA += (float)xb.w * ((float)(p0a[3] >> 4) * sA + mA);
        accA += (float)xc.x * ((float)(p1a[0] & 0xF) * sA + mA);
        accA += (float)xc.y * ((float)(p1a[0] >> 4) * sA + mA);
        accA += (float)xc.z * ((float)(p1a[1] & 0xF) * sA + mA);
        accA += (float)xc.w * ((float)(p1a[1] >> 4) * sA + mA);
        accA += (float)xd.x * ((float)(p1a[2] & 0xF) * sA + mA);
        accA += (float)xd.y * ((float)(p1a[2] >> 4) * sA + mA);
        accA += (float)xd.z * ((float)(p1a[3] & 0xF) * sA + mA);
        accA += (float)xd.w * ((float)(p1a[3] >> 4) * sA + mA);
        accB += (float)xa.x * ((float)(p0b[0] & 0xF) * sB + mB);
        accB += (float)xa.y * ((float)(p0b[0] >> 4) * sB + mB);
        accB += (float)xa.z * ((float)(p0b[1] & 0xF) * sB + mB);
        accB += (float)xa.w * ((float)(p0b[1] >> 4) * sB + mB);
        accB += (float)xb.x * ((float)(p0b[2] & 0xF) * sB + mB);
        accB += (float)xb.y * ((float)(p0b[2] >> 4) * sB + mB);
        accB += (float)xb.z * ((float)(p0b[3] & 0xF) * sB + mB);
        accB += (float)xb.w * ((float)(p0b[3] >> 4) * sB + mB);
        accB += (float)xc.x * ((float)(p1b[0] & 0xF) * sB + mB);
        accB += (float)xc.y * ((float)(p1b[0] >> 4) * sB + mB);
        accB += (float)xc.z * ((float)(p1b[1] & 0xF) * sB + mB);
        accB += (float)xc.w * ((float)(p1b[1] >> 4) * sB + mB);
        accB += (float)xd.x * ((float)(p1b[2] & 0xF) * sB + mB);
        accB += (float)xd.y * ((float)(p1b[2] >> 4) * sB + mB);
        accB += (float)xd.z * ((float)(p1b[3] & 0xF) * sB + mB);
        accB += (float)xd.w * ((float)(p1b[3] >> 4) * sB + mB);
    }
    ssq = simd_sum(ssq);
    float inv = rsqrt(ssq / (float)K + EPS_PLACEHOLDER);
    accA = simd_sum(accA) * inv;
    accB = simd_sum(accB) * inv;
    if (lane == 0) {
        out[r0] = (float16_t)accA;
    }
    if (lane == 1) {
        out[r0 + 1] = (float16_t)accB;
    }
"##;

pub const V8A: &str = r##"
    uint pair = threadgroup_position_in_grid.x;
    uint lane = thread_index_in_threadgroup;
    uint K2 = q_shape[1];
    uint K = K2 * 2;
    uint NG = sm_shape[1];
    uint r0 = pair * 2;
    const device uchar* qrowA = q + (size_t)r0 * K2;
    const device uchar* qrowB = qrowA + K2;
    float accA = 0.0f;
    float accB = 0.0f;
    for (uint b8 = lane * 8; b8 < K2; b8 += 256) {
        uchar4 p0a = *(const device uchar4*)(&qrowA[b8]);
        uchar4 p1a = *(const device uchar4*)(&qrowA[b8 + 4]);
        uchar4 p0b = *(const device uchar4*)(&qrowB[b8]);
        uchar4 p1b = *(const device uchar4*)(&qrowB[b8 + 4]);
        uint kk = b8 * 2;
        uint gi = kk / G_PLACEHOLDER;
        half2 smA = *(const device half2*)(&sm[(r0 * NG + gi) * 2]);
        half2 smB = *(const device half2*)(&sm[((r0 + 1) * NG + gi) * 2]);
        float sA = (float)smA.x;
        float mA = (float)smA.y;
        float sB = (float)smB.x;
        float mB = (float)smB.y;
        half4 g1a = *(const device half4*)(&gu[kk]);
        half4 g1b = *(const device half4*)(&gu[kk + 4]);
        half4 g1c = *(const device half4*)(&gu[kk + 8]);
        half4 g1d = *(const device half4*)(&gu[kk + 12]);
        half4 g2a = *(const device half4*)(&gu[K + kk]);
        half4 g2b = *(const device half4*)(&gu[K + kk + 4]);
        half4 g2c = *(const device half4*)(&gu[K + kk + 8]);
        half4 g2d = *(const device half4*)(&gu[K + kk + 12]);
        float x0 = ((float)g1a.x / (1.0f + exp2(-1.4426950408889634f * (float)g1a.x))) * (float)g2a.x;
        float x1 = ((float)g1a.y / (1.0f + exp2(-1.4426950408889634f * (float)g1a.y))) * (float)g2a.y;
        float x2 = ((float)g1a.z / (1.0f + exp2(-1.4426950408889634f * (float)g1a.z))) * (float)g2a.z;
        float x3 = ((float)g1a.w / (1.0f + exp2(-1.4426950408889634f * (float)g1a.w))) * (float)g2a.w;
        float x4 = ((float)g1b.x / (1.0f + exp2(-1.4426950408889634f * (float)g1b.x))) * (float)g2b.x;
        float x5 = ((float)g1b.y / (1.0f + exp2(-1.4426950408889634f * (float)g1b.y))) * (float)g2b.y;
        float x6 = ((float)g1b.z / (1.0f + exp2(-1.4426950408889634f * (float)g1b.z))) * (float)g2b.z;
        float x7 = ((float)g1b.w / (1.0f + exp2(-1.4426950408889634f * (float)g1b.w))) * (float)g2b.w;
        float x8 = ((float)g1c.x / (1.0f + exp2(-1.4426950408889634f * (float)g1c.x))) * (float)g2c.x;
        float x9 = ((float)g1c.y / (1.0f + exp2(-1.4426950408889634f * (float)g1c.y))) * (float)g2c.y;
        float x10 = ((float)g1c.z / (1.0f + exp2(-1.4426950408889634f * (float)g1c.z))) * (float)g2c.z;
        float x11 = ((float)g1c.w / (1.0f + exp2(-1.4426950408889634f * (float)g1c.w))) * (float)g2c.w;
        float x12 = ((float)g1d.x / (1.0f + exp2(-1.4426950408889634f * (float)g1d.x))) * (float)g2d.x;
        float x13 = ((float)g1d.y / (1.0f + exp2(-1.4426950408889634f * (float)g1d.y))) * (float)g2d.y;
        float x14 = ((float)g1d.z / (1.0f + exp2(-1.4426950408889634f * (float)g1d.z))) * (float)g2d.z;
        float x15 = ((float)g1d.w / (1.0f + exp2(-1.4426950408889634f * (float)g1d.w))) * (float)g2d.w;
        accA += x0 * ((float)(p0a[0] & 0xF) * sA + mA);
        accA += x1 * ((float)(p0a[0] >> 4) * sA + mA);
        accA += x2 * ((float)(p0a[1] & 0xF) * sA + mA);
        accA += x3 * ((float)(p0a[1] >> 4) * sA + mA);
        accA += x4 * ((float)(p0a[2] & 0xF) * sA + mA);
        accA += x5 * ((float)(p0a[2] >> 4) * sA + mA);
        accA += x6 * ((float)(p0a[3] & 0xF) * sA + mA);
        accA += x7 * ((float)(p0a[3] >> 4) * sA + mA);
        accA += x8 * ((float)(p1a[0] & 0xF) * sA + mA);
        accA += x9 * ((float)(p1a[0] >> 4) * sA + mA);
        accA += x10 * ((float)(p1a[1] & 0xF) * sA + mA);
        accA += x11 * ((float)(p1a[1] >> 4) * sA + mA);
        accA += x12 * ((float)(p1a[2] & 0xF) * sA + mA);
        accA += x13 * ((float)(p1a[2] >> 4) * sA + mA);
        accA += x14 * ((float)(p1a[3] & 0xF) * sA + mA);
        accA += x15 * ((float)(p1a[3] >> 4) * sA + mA);
        accB += x0 * ((float)(p0b[0] & 0xF) * sB + mB);
        accB += x1 * ((float)(p0b[0] >> 4) * sB + mB);
        accB += x2 * ((float)(p0b[1] & 0xF) * sB + mB);
        accB += x3 * ((float)(p0b[1] >> 4) * sB + mB);
        accB += x4 * ((float)(p0b[2] & 0xF) * sB + mB);
        accB += x5 * ((float)(p0b[2] >> 4) * sB + mB);
        accB += x6 * ((float)(p0b[3] & 0xF) * sB + mB);
        accB += x7 * ((float)(p0b[3] >> 4) * sB + mB);
        accB += x8 * ((float)(p1b[0] & 0xF) * sB + mB);
        accB += x9 * ((float)(p1b[0] >> 4) * sB + mB);
        accB += x10 * ((float)(p1b[1] & 0xF) * sB + mB);
        accB += x11 * ((float)(p1b[1] >> 4) * sB + mB);
        accB += x12 * ((float)(p1b[2] & 0xF) * sB + mB);
        accB += x13 * ((float)(p1b[2] >> 4) * sB + mB);
        accB += x14 * ((float)(p1b[3] & 0xF) * sB + mB);
        accB += x15 * ((float)(p1b[3] >> 4) * sB + mB);
    }
    accA = simd_sum(accA);
    accB = simd_sum(accB);
    if (lane == 0) {
        out[r0] = (float16_t)(accA + (float)res[r0]);
    }
    if (lane == 1) {
        out[r0 + 1] = (float16_t)(accB + (float)res[r0 + 1]);
    }
"##;

pub const FATTN_SOURCE: &str = r##"
    uint head = threadgroup_position_in_grid.x;
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint W = k_shape[1];
    uint NH = k_shape[0];
    uint HD = HD_CONST;
    size_t D = (size_t)NH * HD;
    uint d0 = lane * 4;
    float p = (float)pos[0];
    // gqkv layout: [g (D) | q (D) | k (D) | v (D)] — q/k/v blocks are read at
    // offsets D/2D/3D directly, so the graph passes gqkv whole (no slices,
    // no reshapes, no separate v input).
    const device half* qb = gqkv + D + (size_t)head * HD + d0;
    const device half* kb = gqkv + 2 * D + (size_t)head * HD + d0;
    const device half* vb = gqkv + 3 * D + (size_t)head * HD + d0;
    // NeoX-style half-split rope: dim i pairs with dim i+HD/2 (freq index i%HD/2).
    // Each lane owns 4 contiguous dims, all within one half; partner dims are
    // loaded from global (L1-hot, the qk slice is tiny).
    uint halfd = HD / 2;
    int off;
    float sgn;
    uint i0;
    if (d0 < halfd) {
        off = (int)halfd;
        sgn = -1.0f;
        i0 = d0;
    } else {
        off = -(int)halfd;
        sgn = 1.0f;
        i0 = d0 - halfd;
    }
    float q0 = (float)qb[0], q1 = (float)qb[1], q2 = (float)qb[2], q3 = (float)qb[3];
    float qp0 = (float)qb[off], qp1 = (float)qb[off + 1], qp2 = (float)qb[off + 2], qp3 = (float)qb[off + 3];
    float k0 = (float)kb[0], k1 = (float)kb[1], k2 = (float)kb[2], k3 = (float)kb[3];
    float kp0 = (float)kb[off], kp1 = (float)kb[off + 1], kp2 = (float)kb[off + 2], kp3 = (float)kb[off + 3];
    float v0 = (float)vb[0], v1 = (float)vb[1], v2 = (float)vb[2], v3 = (float)vb[3];
    // rope frequencies computed in-kernel (MLX rope convention, empirically
    // confirmed: the freqs input holds theta^(+2i/HD) and the rotation angle is
    // offset / freqs[i] == offset * theta^(-2i/HD); folding the table into the
    // kernel removes one input per call and the freqs-array plumbing).
    float fr0 = exp2(-(float)(i0 + 0) * (2.0f / (float)HD) * LOG2_THETA);
    float fr1 = exp2(-(float)(i0 + 1) * (2.0f / (float)HD) * LOG2_THETA);
    float fr2 = exp2(-(float)(i0 + 2) * (2.0f / (float)HD) * LOG2_THETA);
    float fr3 = exp2(-(float)(i0 + 3) * (2.0f / (float)HD) * LOG2_THETA);
    float w0 = p * fr0, w1 = p * fr1, w2 = p * fr2, w3 = p * fr3;
    float c0 = cos(w0), s0 = sin(w0);
    float c1 = cos(w1), s1 = sin(w1);
    float c2 = cos(w2), s2 = sin(w2);
    float c3 = cos(w3), s3 = sin(w3);
    float qr0 = q0 * c0 + sgn * qp0 * s0;
    float qr1 = q1 * c1 + sgn * qp1 * s1;
    float qr2 = q2 * c2 + sgn * qp2 * s2;
    float qr3 = q3 * c3 + sgn * qp3 * s3;
    float kr0 = k0 * c0 + sgn * kp0 * s0;
    float kr1 = k1 * c1 + sgn * kp1 * s1;
    float kr2 = k2 * c2 + sgn * kp2 * s2;
    float kr3 = k3 * c3 + sgn * kp3 * s3;
    float sc = rsqrt((float)HD);
    qr0 *= sc; qr1 *= sc; qr2 *= sc; qr3 *= sc;
    out_k[head * HD + d0 + 0] = (float16_t)kr0;
    out_k[head * HD + d0 + 1] = (float16_t)kr1;
    out_k[head * HD + d0 + 2] = (float16_t)kr2;
    out_k[head * HD + d0 + 3] = (float16_t)kr3;
    float mn = -INFINITY;
    float ss = 0.0f;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    uint vis = (uint)fmin(p, (float)W);
    const device half* khead = k + ((size_t)head * W) * HD + d0;
    const device half* vhead = vv + ((size_t)head * W) * HD + d0;
    %%FATTN_LOOP%%
    // merge the 4 simdgroup partials through threadgroup memory
    threadgroup float shm[4];
    threadgroup float shs[4];
    threadgroup float sha[4][HD_CONST];
    if (lane == 0) {
        shm[sg] = mn;
        shs[sg] = ss;
    }
    sha[sg][d0 + 0] = a0;
    sha[sg][d0 + 1] = a1;
    sha[sg][d0 + 2] = a2;
    sha[sg][d0 + 3] = a3;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mstar = fmax(fmax(shm[0], shm[1]), fmax(shm[2], shm[3]));
    float w0m = (shm[0] == -INFINITY) ? 0.0f : exp(shm[0] - mstar);
    float w1m = (shm[1] == -INFINITY) ? 0.0f : exp(shm[1] - mstar);
    float w2m = (shm[2] == -INFINITY) ? 0.0f : exp(shm[2] - mstar);
    float w3m = (shm[3] == -INFINITY) ? 0.0f : exp(shm[3] - mstar);
    a0 = w0m * sha[0][d0] + w1m * sha[1][d0] + w2m * sha[2][d0] + w3m * sha[3][d0];
    a1 = w0m * sha[0][d0 + 1] + w1m * sha[1][d0 + 1] + w2m * sha[2][d0 + 1] + w3m * sha[3][d0 + 1];
    a2 = w0m * sha[0][d0 + 2] + w1m * sha[1][d0 + 2] + w2m * sha[2][d0 + 2] + w3m * sha[3][d0 + 2];
    a3 = w0m * sha[0][d0 + 3] + w1m * sha[1][d0 + 3] + w2m * sha[2][d0 + 3] + w3m * sha[3][d0 + 3];
    mn = mstar;
    ss = w0m * shs[0] + w1m * shs[1] + w2m * shs[2] + w3m * shs[3];
    float s_self = simd_sum(qr0 * kr0 + qr1 * kr1 + qr2 * kr2 + qr3 * kr3);
    float mn2 = fmax(mn, s_self);
    float corr = (mn == -INFINITY) ? 0.0f : exp(mn - mn2);
    float pj = exp(s_self - mn2);
    a0 = a0 * corr + pj * v0;
    a1 = a1 * corr + pj * v1;
    a2 = a2 * corr + pj * v2;
    a3 = a3 * corr + pj * v3;
    ss = ss * corr + pj;
    out_a[head * HD + d0 + 0] = (float16_t)(a0 / ss);
    out_a[head * HD + d0 + 1] = (float16_t)(a1 / ss);
    out_a[head * HD + d0 + 2] = (float16_t)(a2 / ss);
    out_a[head * HD + d0 + 3] = (float16_t)(a3 / ss);
"##;

pub const FATTN_LOOP_BASE: &str = r##"
    // this simdgroup's contiguous slice of the visible rows
    uint j0 = sg * 32;
    uint j1 = min(vis, j0 + 32);
    for (uint j = j0; j < j1; ++j) {
        float kj0 = (float)khead[j * HD + 0];
        float kj1 = (float)khead[j * HD + 1];
        float kj2 = (float)khead[j * HD + 2];
        float kj3 = (float)khead[j * HD + 3];
        // partial dot product (this lane's 4 dims) -> full score via simd reduce
        float sp = qr0 * kj0 + qr1 * kj1 + qr2 * kj2 + qr3 * kj3;
        float s_ = simd_sum(sp);
        float mn2 = fmax(mn, s_);
        float corr = (mn == -INFINITY) ? 0.0f : exp(mn - mn2);
        float pj = exp(s_ - mn2);
        float vj0 = (float)vhead[j * HD + 0];
        float vj1 = (float)vhead[j * HD + 1];
        float vj2 = (float)vhead[j * HD + 2];
        float vj3 = (float)vhead[j * HD + 3];
        a0 = a0 * corr + pj * vj0;
        a1 = a1 * corr + pj * vj1;
        a2 = a2 * corr + pj * vj2;
        a3 = a3 * corr + pj * vj3;
        ss = ss * corr + pj;
        mn = mn2;
    }
"##;
