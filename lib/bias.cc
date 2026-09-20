#include "bias.h"

#ifdef NEON_OPT
#include <arm_neon.h>
#endif

#include <cassert>
#include <cmath>
#include <vector>

#include "misc.h"

namespace {

__fp16 gelu(__fp16 x_in) {
  float x = x_in;
  return x * 0.5 * (1 + std::erf(x / std::sqrt(2.0)));
}

struct _gelu_lut {
  static constexpr int LUT_SIZE = 1 << 16;
  uint16_t lut[LUT_SIZE];

  _gelu_lut() {
    for (int i = 0; i < LUT_SIZE; ++i) {
      uint16_t x = i;
      __fp16 y = gelu(to_f16(x));
      lut[i] = to_u16(y);
    }
  }

  __fp16 operator()(__fp16 x) const {
    uint16_t u16 = to_u16(x);
    return to_f16(lut[u16]);
  }
} gelu_lut;

}  // namespace

void bias_and_gelu(float* src, __fp16* dst, const std::vector<__fp16>& bias,
                   int rows, int cols, int src_rows_padded,
                   int dst_rows_padded) {
#ifdef NEON_OPT
  int i = 0;
  for (i = 0; i < cols - (cols % 4); i += 4) {
    // src's actual number of rows (src->M) times 4 is the base for a block of
    // columns
    float32_t* block_base = src + (src_rows_padded * 4) * (i / 4);
    float16_t* dst_block_base = dst + (dst_rows_padded * 8) * (i / 8);
    for (int j = 0; j < rows; ++j) {
      auto v = vld1q_f32(block_base + j * 4);
      auto v_f16 = vcvt_f16_f32(v);
      auto bias_f16 = vld1_f16(&bias.data()[i]);
      v_f16 = vadd_f16(v_f16, bias_f16);
      v_f16 = vset_lane_f16(gelu_lut(vget_lane_f16(v_f16, 0)), v_f16, 0);
      v_f16 = vset_lane_f16(gelu_lut(vget_lane_f16(v_f16, 1)), v_f16, 1);
      v_f16 = vset_lane_f16(gelu_lut(vget_lane_f16(v_f16, 2)), v_f16, 2);
      v_f16 = vset_lane_f16(gelu_lut(vget_lane_f16(v_f16, 3)), v_f16, 3);
      vst1_f16(dst_block_base + j * 8 + ((i % 8) / 4) * 4, v_f16);
    }
  }

  if (cols % 4 != 0) {
    float32_t* block_base = src + (src_rows_padded * 4) * (i / 4);
    float16_t* dst_block_base = dst + (dst_rows_padded * 8) * (i / 8);
    for (int j = 0; j < rows; ++j) {
      for (int rem_i = 0; rem_i < cols - i; ++rem_i) {
        __fp16 elem = block_base[rem_i + j * 4] + bias[i + rem_i];
        elem = gelu_lut(elem);
        dst_block_base[j * 8 + ((i % 8) / 4) * 4 + rem_i] = elem;
      }
    }
  }
#else
  for (int i = 0; i < cols; ++i) {
    for (int j = 0; j < rows; ++j) {
      float v = src[((i / 4) * src_rows_padded + j) * 4 + (i % 4)];
      __fp16 elem = (__fp16)v + bias[i];
      elem = gelu_lut(elem);
      dst[((i / 8) * dst_rows_padded + j) * 8 + (i % 8)] = elem;
    }
  }
#endif
}

void bias_and_gelu_C_to_A(Matmul* src, Matmul* dst,
                          const std::vector<__fp16>& bias, int rows, int cols) {
  bias_and_gelu(src->get_C_ptr(), dst->get_A_ptr(), bias, rows, cols, src->M,
                dst->M);
}

void bias_and_gelu(float* src, __fp16* dst, const std::vector<__fp16>& bias,
                   int rows, int cols) {
  bias_and_gelu(src, dst, bias, rows, cols, rows, rows);
}

void bias_and_scale_C(Matmul* dst, const std::vector<__fp16>& bias, float scale,
                      int rows, int cols) {
#ifdef NEON_OPT
  float32_t* C = dst->get_C_ptr();
  int i = 0;
  for (i = 0; i < cols - (cols % 4); i += 4) {
    float32_t* block_base = C + (dst->M * 4) * (i / 4);
    for (int j = 0; j < rows; ++j) {
      auto v = vld1q_f32(block_base + j * 4);
      auto bias_f16 = vld1_f16(&bias.data()[i]);
      auto bias_f32 = vcvt_f32_f16(bias_f16);
      v = vaddq_f32(v, bias_f32);
      v = vmulq_f32(v, vdupq_n_f32(scale));
      vst1q_f32(block_base + j * 4, v);
    }
  }
  if (cols % 4 != 0) {
    float32_t* block_base = C + (dst->M * 4) * (i / 4);
    for (int j = 0; j < rows; ++j) {
      for (int rem_i = 0; rem_i < cols - i; ++rem_i) {
        block_base[rem_i + j * 4] += bias[i + rem_i];
        block_base[rem_i + j * 4] *= scale;
      }
    }
  }
#else
  float* C = dst->get_C_ptr();
  for (int i = 0; i < cols; ++i) {
    for (int j = 0; j < rows; ++j) {
      int idx = ((i / 4) * dst->M + j) * 4 + (i % 4);
      C[idx] = (C[idx] + (float)bias[i]) * scale;
    }
  }
#endif
}

void add_residual_C_to_A(Matmul* src, Matmul* dst, std::vector<__fp16> residual,
                         int rows, int cols) {
  assert(cols % 8 == 0);
  float* src_ptr = src->get_C_ptr();
  __fp16* dst_ptr = dst->get_A_ptr();
#ifdef NEON_OPT
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j += 8) {
      int dst_idx = j * rows + i * 8;
      int src_idx = j * rows + i * 4;
      auto in0 = vld1q_f32(&src_ptr[src_idx]);
      auto in1 = vld1q_f32(&src_ptr[src_idx + rows * 4]);
      auto res = vld1q_f16(&residual[dst_idx]);
      auto in = vcombine_f16(vcvt_f16_f32(in0), vcvt_f16_f32(in1));
      auto out = vaddq_f16(in, res);
      vst1q_f16(&dst_ptr[dst_idx], out);
    }
  }
#else
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j += 8) {
      int dst_idx = j * rows + i * 8;
      int src_idx = j * rows + i * 4;
      for (int lane = 0; lane < 4; ++lane) {
        dst_ptr[dst_idx + lane] =
            (__fp16)src_ptr[src_idx + lane] + residual[dst_idx + lane];
        dst_ptr[dst_idx + 4 + lane] =
            (__fp16)src_ptr[src_idx + rows * 4 + lane] +
            residual[dst_idx + 4 + lane];
      }
    }
  }
#endif
}

void bias_and_add_residual_C(Matmul* src, std::vector<__fp16>& dst,
                             std::vector<__fp16> residual,
                             std::vector<__fp16> bias, int rows, int cols) {
  assert(cols % 8 == 0);
  float* src_ptr = src->get_C_ptr();
#ifdef NEON_OPT
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j += 8) {
      int dst_idx = j * rows + i * 8;
      int src_idx = j * rows + i * 4;
      auto in0 = vld1q_f32(&src_ptr[src_idx]);
      auto in1 = vld1q_f32(&src_ptr[src_idx + rows * 4]);
      auto res = vld1q_f16(&residual[dst_idx]);
      auto bias_f16 = vld1q_f16(&bias[j]);
      auto in = vcombine_f16(vcvt_f16_f32(in0), vcvt_f16_f32(in1));
      auto out = vaddq_f16(in, res);
      out = vaddq_f16(out, bias_f16);
      vst1q_f16(&dst[i * cols + j], out);
    }
  }
#else
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j += 8) {
      int dst_idx = j * rows + i * 8;
      int src_idx = j * rows + i * 4;
      for (int lane = 0; lane < 4; ++lane) {
        dst[i * cols + j + lane] = (__fp16)src_ptr[src_idx + lane] +
                                   residual[dst_idx + lane] + bias[j + lane];
        dst[i * cols + j + 4 + lane] = (__fp16)src_ptr[src_idx + rows * 4 +
                                                       lane] +
                                       residual[dst_idx + 4 + lane] +
                                       bias[j + 4 + lane];
      }
    }
  }
#endif
}