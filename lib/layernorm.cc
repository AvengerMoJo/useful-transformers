#include "layernorm.h"

#ifdef NEON_OPT
#include <arm_neon.h>
#endif

#include <cassert>
#include <cmath>

void layernorm_A(Matmul *x, int rows, int cols,
                 const std::vector<__fp16> &gamma,
                 const std::vector<__fp16> &beta, __fp16 eps) {
  assert(!(cols % 8));
#ifdef NEON_OPT
#pragma omp parallel for
  for (int i = 0; i < rows; i++) {
    // Calculate mean.
    __fp16 *input_ptr = x->get_A_ptr() + 8 * i;
    float16x8_t mean = vdupq_n_f16(0);
    for (int j = 0; j < cols; j += 8) {
      int offset = rows * j;
      mean = vaddq_f16(vld1q_f16(input_ptr + offset), mean);
    }
    float16x4_t mean_f16x4 = vadd_f16(vget_high_f16(mean), vget_low_f16(mean));
    float32x4_t mean_f32x4 = vcvt_f32_f16(mean_f16x4);
    float32_t mean_f32 = vaddvq_f32(mean_f32x4) / (float32_t)cols;
    mean = vdupq_n_f16((__fp16)mean_f32);

    // Calculate mean squared deviation
    float16x8_t msd = vdupq_n_f16(0);
    for (int j = 0; j < cols; j += 8) {
      int offset = rows * j;
      float16x8_t val = vld1q_f16(input_ptr + offset);
      val = vsubq_f16(val, mean);
      val = vmulq_f16(val, val);
      msd = vaddq_f16(val, msd);
    }

    float16x4_t msd_f16x4 = vadd_f16(vget_high_f16(msd), vget_low_f16(msd));
    float32x4_t msd_f32x4 = vcvt_f32_f16(msd_f16x4);
    float32_t msd_f32 = vaddvq_f32(msd_f32x4);
    float32_t denom_single = msd_f32 / (float32_t)(cols - 1);
    denom_single = std::sqrt(denom_single + eps);
    float16x8_t denom = vdupq_n_f16((__fp16)denom_single);

    // Normalize based on mean + MSD, beta, gamma and epsilon.
    for (int j = 0; j < cols; j += 8) {
      int offset = rows * j;
      float16x8_t val = vld1q_f16(input_ptr + offset);
      float16x8_t gamma_val = vld1q_f16(gamma.data() + j);
      float16x8_t beta_val = vld1q_f16(beta.data() + j);
      val = vsubq_f16(val, mean);
      val = vdivq_f16(val, denom);
      val = vmulq_f16(val, gamma_val);
      val = vaddq_f16(val, beta_val);
      vst1q_f16(input_ptr + offset, val);
    }
  }
#else
#pragma omp parallel for
  for (int i = 0; i < rows; i++) {
    __fp16 *input_ptr = x->get_A_ptr() + 8 * i;
    float mean_f32 = 0.0;
    for (int j = 0; j < cols; j += 8) {
      int offset = rows * j;
      for (int lane = 0; lane < 8; ++lane) {
        mean_f32 += (float)input_ptr[offset + lane];
      }
    }
    mean_f32 /= cols;
    __fp16 mean = (__fp16)mean_f32;

    float msd_f32 = 0.0;
    for (int j = 0; j < cols; j += 8) {
      int offset = rows * j;
      for (int lane = 0; lane < 8; ++lane) {
        float val = (float)input_ptr[offset + lane] - (float)mean;
        msd_f32 += val * val;
      }
    }
    float denom_single = msd_f32 / (float)(cols - 1);
    denom_single = std::sqrt(denom_single + (float)eps);
    __fp16 denom = (__fp16)denom_single;

    for (int j = 0; j < cols; j += 8) {
      int offset = rows * j;
      for (int lane = 0; lane < 8; ++lane) {
        int col = j + lane;
        __fp16 val = input_ptr[offset + lane];
        val = (__fp16)((float)val - (float)mean);
        val = (__fp16)((float)val / (float)denom);
        val = (__fp16)((float)val * (float)gamma[col]);
        val = (__fp16)((float)val + (float)beta[col]);
        input_ptr[offset + lane] = val;
      }
    }
  }
#endif
}