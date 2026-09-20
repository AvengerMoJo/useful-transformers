#ifdef HAVE_RKNN

#include "matmul.h"
#include "matmul_backend.h"

#include <cassert>
#include <cstring>

#include "rknn_matmul_api.h"

namespace {

// RKNN NPU backend. Owns the rknn_matmul context, its io_attr and the three
// NPU memory objects as backend state. The M/K/N matmul geometry comes from
// the Matmul (M real, K/N padded to the NPU alignment requirements).
struct RknnMatmulBackend : public MatmulBackend {
  rknn_matmul_ctx ctx;
  rknn_matmul_info info;
  rknn_matmul_io_attr io_attr;
  rknn_tensor_mem *memA = nullptr;
  rknn_tensor_mem *memB = nullptr;
  rknn_tensor_mem *memC = nullptr;

  void allocate(Matmul *m, int core) override {
    memset(&info, 0, sizeof(rknn_matmul_info));
    info.M = m->M;
    info.K = m->K_padded;
    info.N = m->N_padded;
    info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
    info.B_layout = 1;
    info.AC_layout = 1;
    memset(&io_attr, 0, sizeof(rknn_matmul_io_attr));

    int ret = 0;

    ret = rknn_matmul_create(&ctx, &info, &io_attr);
    assert(ret >= 0 && "Constructor failed");

    if (core == 0)
      ret = rknn_matmul_set_core_mask(ctx, RKNN_NPU_CORE_0);
    else if (core == 1)
      ret = rknn_matmul_set_core_mask(ctx, RKNN_NPU_CORE_1);
    else if (core == 2)
      ret = rknn_matmul_set_core_mask(ctx, RKNN_NPU_CORE_2);
    else
      ret = -1;
    assert(ret >= 0 && "Set core mask failed");

    assert(io_attr.A.n_dims == 3 && "Expected A to be 3 dimensional");
    assert(io_attr.B.n_dims == 4 && "Expected B to be 4 dimensional");
    assert(io_attr.C.n_dims == 3 && "Expected C to be 3 dimensional");

    assert(io_attr.A.dims[2] == 8 && io_attr.A.dims[1] == m->M &&
           io_attr.A.dims[0] == m->K_padded / 8 &&
           "A dims are not [K/8, M, 8]");
    assert(io_attr.B.dims[3] == 32 && io_attr.B.dims[2] == 16 &&
           io_attr.B.dims[1] == m->K_padded / 32 &&
           io_attr.B.dims[0] == m->N_padded / 16 &&
           "B dims are not [N/16, K/32, 16, 32]");
    assert(io_attr.C.dims[2] == 4 && io_attr.C.dims[1] == m->M &&
           io_attr.C.dims[0] == m->N_padded / 4 &&
           "C dims are not [N/4, M, 4]");

    memA = rknn_create_mem(ctx, io_attr.A.size);
    assert(memA != NULL && "A allocation failed");
    memB = rknn_create_mem(ctx, io_attr.B.size);
    assert(memB != NULL && "B allocation failed");
    memC = rknn_create_mem(ctx, io_attr.C.size);
    assert(memC != NULL && "C allocation failed");

    m->A = (__fp16 *)((size_t)memA->virt_addr + memA->offset);
    m->B = (__fp16 *)((size_t)memB->virt_addr + memB->offset);
    m->C = (float *)((size_t)memC->virt_addr + memC->offset);
  }

  void call(Matmul *m) override {
    (void)m;
    int ret;
    ret = rknn_matmul_set_io_mem(ctx, memA, &io_attr.A);
    assert(ret >= 0 && "Setting A input failed");
    ret = rknn_matmul_set_io_mem(ctx, memB, &io_attr.B);
    assert(ret >= 0 && "Setting B input failed");
    ret = rknn_matmul_set_io_mem(ctx, memC, &io_attr.C);
    assert(ret >= 0 && "Setting C ouput failed");
    ret = rknn_matmul_run(ctx);
    assert(ret >= 0 && "matmul launch failed\n");
  }

  void deallocate(Matmul *m) override {
    rknn_destroy_mem(ctx, memA);
    rknn_destroy_mem(ctx, memB);
    rknn_destroy_mem(ctx, memC);
    memA = nullptr;
    memB = nullptr;
    memC = nullptr;
    rknn_matmul_destroy(ctx);
    m->A = nullptr;
    m->B = nullptr;
    m->C = nullptr;
  }

  const char *name() const override { return "rknn"; }
};

}  // namespace

MatmulBackend *create_rknn_backend() { return new RknnMatmulBackend(); }

#endif  // HAVE_RKNN