#include "matmul.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "matmul_backend.h"

Matmul::Matmul(int M, int K, int N, int core) : M(M), K(K), N(N) {
  K_padded = NEXT_MULTIPLE_OF_32(K);
  N_padded = NEXT_MULTIPLE_OF_16(N);
  backend = create_matmul_backend();
  backend->allocate(this, core);
  zero_A();
  zero_B();
}

Matmul::~Matmul() {
  backend->deallocate(this);
  delete backend;
  backend = nullptr;
}

void Matmul::zero_A() { memset(get_A_ptr(), 0, sizeof(__fp16) * M * K_padded); }

void Matmul::zero_B() {
  memset(get_B_ptr(), 0, sizeof(__fp16) * K_padded * N_padded);
}

void Matmul::copy_B_to_B(Matmul* other) {
  other->zero_B();
  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < N; ++j) {
      other->B_at(i, j) = B_at(i, j);
    }
  }
}

void Matmul::copy_C_to_A(Matmul* other, Slice slice) {
  for (int i = 0; i < slice.n_rows; ++i) {
    for (int j = 0; j < slice.n_cols; ++j) {
      other->A_at(i + slice.dst_row_offset, j + slice.dst_col_offset) =
          C_at(i + slice.src_row_offset, j + slice.src_col_offset);
    }
  }
}

void Matmul::copy_C_to_Bt(Matmul* other, Slice slice) {
  for (int i = 0; i < slice.n_rows; ++i) {
    for (int j = 0; j < slice.n_cols; ++j) {
      other->B_at(j + slice.dst_col_offset, i + slice.dst_row_offset) =
          C_at(i + slice.src_row_offset, j + slice.src_col_offset);
    }
  }
}

void Matmul::copy_C_to_B(Matmul* other, Slice slice) {
  for (int i = 0; i < slice.n_rows; ++i) {
    for (int j = 0; j < slice.n_cols; ++j) {
      other->B_at(i + slice.dst_row_offset, j + slice.dst_col_offset) =
          C_at(i + slice.src_row_offset, j + slice.src_col_offset);
    }
  }
}

void Matmul::call() { backend->call(this); }

MatmulBackend *create_matmul_backend() {
  const char *env = getenv("USEFUL_TRANSFORMERS_BACKEND");
#ifdef HAVE_RKNN
  const char *requested = (env != nullptr) ? env : "rknn";
#else
  const char *requested = (env != nullptr) ? env : "cpu";
#endif

  if (strcmp(requested, "rknn") == 0) {
#ifdef HAVE_RKNN
    return create_rknn_backend();
#else
    fprintf(stderr,
            "WARNING: rknn matmul backend requested, but this build has no "
            "RKNN support; falling back to cpu\n");
    return create_cpu_backend();
#endif
  }
  return create_cpu_backend();
}