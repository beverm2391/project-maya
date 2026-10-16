#pragma once

#include "strata/kernels/glm_full_dsa.hpp"

// Layer-major full GLM prefill. Features are contiguous, then heads, then T.
// The attention wrappers preserve the single-token primitives' arithmetic while
// the owning prefill path batches the expensive projections and expert work.
namespace strata::kernels::full_glm_dsa {

// h[t] += add[t] (optional), x[t] = RMS(h[t],eps)*w (optional); optional x16
// receives IEEE FP16. No 4096-element limit and no embedding-size divisibility
// requirement. h==x is supported; partial overlap is not supported.
void residual_norm_rows(float* h, const float* add, const float* w, float eps,
                        int T, int E, float* x, void* x16, void* stream);
void add_residual_rows(float* h, const float* add, int T, int E, void* stream);

// q[T][heads][nope+rope], kv[T][latent+rope], cache[cell][latent+rope].
// Writes positions [p0,p0+T) and rotates each query at its own global position.
void prepare_rows(float* q, const float* kv, const float* kv_norm, uint16_t* cache,
                  int p0, int T, int heads, int nope, int latent, int rope,
                  float eps, Rope rotary, void* stream);

// iq[T][heads][dim], ik[T][dim], raw iw[T][heads], key cache[cell][dim].
// Cells[T][topk] persist across shared-indexer layers. Each row selects only
// [0,p0+t], sorted descending, padding -1. One score_scratch row of at least
// p0+T floats is reused in stream order; no T*context score allocation is needed.
void index_rows(float* iq, float* ik, const float* norm_w, const float* norm_b,
                float* key_cache, const float* iw, int p0, int T, int heads,
                int dim, int rope, int topk, float eps, Rope rotary,
                float* score_scratch, int* cells, void* stream);

// q[T][heads][nope+rope], cells[T][topk], out[T][heads][vhead]. Each query
// uses min(p0+t+1,topk) entries; cache may already contain later chunk positions.
// Shared-indexer layers pass the earlier full layer's unchanged cells.
void mla_rows(const float* q, const uint16_t* kb, const uint16_t* vb,
              const uint16_t* cache, const int* cells, int p0, int T, int topk,
              int heads, int nope, int latent, int rope, int vhead, float scale,
              float* out, void* stream);

}  // namespace strata::kernels::full_glm_dsa
