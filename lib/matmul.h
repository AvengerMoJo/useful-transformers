#ifndef _LIB_MATMUL_H_
#define _LIB_MATMUL_H_

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>

#include "fp16.h"

#define NEXT_MULTIPLE_OF_32(x) (((x) + 31) & ~31)
#define NEXT_MULTIPLE_OF_16(x) (((x) + 15) & ~15)

struct MatmulBackend;

// A Matmul computes C = A * B as a float32 result from the product of two
// float16 operands. The M rows of A and M rows of C are real; K and N are
// padded up to alignment (K_padded, N_padded) and the padding is expected to
// be zero-filled (see zero_A/zero_B).
//
// Operand layouts (indices into logical buffers, indexed via A_at/B_at/C_at):
//   A [j/8, i, j%8] of [K_padded / 8, M, 8]
//   B [j/16, i/32, j%16, i%32] of [N_padded / 16, K_padded / 32, 16, 32]
//   C [j/4, i, j%4] of [N_padded / 4, M, 4]
//
// The backing storage for A, B and C is owned by `backend`, created by the
// constructor via create_matmul_backend().
struct Matmul {
  int M = 0;
  int K = 0;
  int N = 0;

  int K_padded;
  int N_padded;

  // Logical buffers, owned by `backend`.
  __fp16 *A = 0;
  __fp16 *B = 0;
  float *C = 0;

  // Execution backend, owns A/B/C and any backend-specific state.
  MatmulBackend *backend = 0;

  Matmul(int M, int K, int N, int core = 0);
  ~Matmul();

  __fp16 *get_A_ptr() { return A; }

  __fp16 *get_B_ptr() { return B; }

  float *get_C_ptr() { return C; }

  __fp16 &A_at(int i, int j) {
    assert(0 <= i && i < M && 0 <= j && j < K && "A indices out of range");
    // A [j/8, i, j%8] of [K_padded / 8, M, 8] array
    return get_A_ptr()[((j / 8) * M + i) * 8 + (j % 8)];
  }

  __fp16 &B_at(int i, int j) {
    assert(0 <= i && i < K && 0 <= j && j < N && "B indices out of range");
    // B [j/16, i/32, j%16, i%32] of [N_padded/16, K_padded/32, 16, 32] array
    return get_B_ptr()[((((j / 16) * (K_padded / 32)) + (i / 32)) * 16 +
                        (j % 16)) *
                           32 +
                       (i % 32)];
  }

  float &C_at(int i, int j) {
    assert(0 <= i && i < M && 0 <= j && j < N && "C indices out of range");
    // C [j/4, i, j%4] of [N_padded/4, M, 4] array
    return get_C_ptr()[(((j / 4) * M + i) * 4) + (j % 4)];
  }

  template <typename T>
  void set_A(const T *src) {
    zero_A();
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < K; ++j) {
        A_at(i, j) = src[i * K + j];
      }
    }
  }

  template <typename T>
  void set_A(const T *src, int num_rows) {
    zero_A();
    for (int i = 0; i < num_rows; ++i) {
      for (int j = 0; j < K; ++j) {
        A_at(i, j) = src[i * K + j];
      }
    }
  }

  template <typename T>
  void set_B(const T *src) {
    zero_B();
    for (int i = 0; i < K; ++i) {
      for (int j = 0; j < N; ++j) {
        B_at(i, j) = src[i * N + j];
      }
    }
  }

  template <typename T>
  void get_A(T *dst, int dst_rows, int dst_cols) {
    for (int i = 0; i < std::min(M, dst_rows); ++i) {
      for (int j = 0; j < std::min(K, dst_cols); ++j) {
        dst[i * dst_cols + j] = A_at(i, j);
      }
    }
  }

  void zero_A();
  void zero_B();
  void copy_B_to_B(Matmul *other);

  struct Slice {
    int n_rows;
    int n_cols;

    int src_row_offset = 0;
    int dst_row_offset = 0;

    int src_col_offset = 0;
    int dst_col_offset = 0;
  };

  void copy_C_to_A(Matmul *other, Slice slice);
  void copy_C_to_Bt(Matmul *other, Slice slice);
  void copy_C_to_B(Matmul *other, Slice slice);

  void call();
};

#endif  // _LIB_MATMUL_H_
