#pragma once

#include <cstdint>

// Full GLM DSA (glm-dsa), distinct from glm5-next's pooled indexer. Single-token
// primitives, features contiguous. All pointers are device pointers; stream may
// be null. Calls enqueue work without allocation or synchronization.
namespace strata::kernels::full_glm_dsa {

// GLM-5.3 uses default/interleaved RoPE, theta=8e6, no YaRN. The loader must reject
// extrapolation/frequency-factor metadata: these primitives do not implement it.
struct Rope {
    float base = 8000000.0f;
    float freq_scale = 1.0f;
    float attn_factor = 1.0f;
};

// h += add (optional); x = RMS(h,eps)*w (optional). h==x is supported. xq may
// be null, otherwise receives the existing glmf q8_1 format; n must divide 32.
// No fixed embedding limit: supports full GLM's 6144 and q_lora's 2048.
void residual_norm(float* h, const float* add, const float* w, float eps, int n,
                   float* x, void* xq, void* stream);
void add_residual(float* h, const float* add, int n, void* stream);

// q[h][nope+rope] is modified only in its last rope channels. kv[latent+rope]
// is raw KV-A output. cache[pos][latent+rope] receives RMS(kv[:latent])*kv_norm,
// followed by rotated kv[latent:]. Cache is IEEE FP16, not BF16.
void prepare(float* q, const float* kv, const float* kv_norm, uint16_t* cache,
             int pos, int heads, int nope, int latent, int rope, float eps,
             Rope rotary, void* stream);

// iq[heads][dim] and ik[dim] modified in-place. ik first uses ordinary LayerNorm
// with affine weight AND bias. Q and K then rotate first rope dimensions with
// adjacent-pair RoPE and apply orthonormal Walsh-Hadamard across dim. Stores
// transformed ik to FLOAT32 cache[pos][dim]. dim must be a power of two <=1024.
void index_prepare(float* iq, float* ik, const float* norm_w, const float* norm_b,
                   float* cache, int pos, int heads, int dim, int rope,
                   float eps, Rope rotary, void* stream);

// score[c] = sum_h ReLU(dot(iq[h],cache[c])) * raw_iw[h]/sqrt(dim*heads).
// ncache includes the current token; no pooling or incomplete-tail behavior.
void score(const float* iq, const float* cache, const float* raw_iw, int dim,
           int heads, int ncache, float* scores, void* stream);

// cells[0:min(ncache,topk)] descending score, ties by lower cell index. topk<=2048.
// Later shared-indexer layers reuse these exact cells with their own KV cache.
void select(const float* scores, int ncache, int topk, int* cells, void* stream);

// Absorbed MLA: kb BF16[head][latent][nope], vb BF16[head][vhead][latent].
// q F32[head][nope+rope], cache FP16[cell][latent+rope], out F32[head][vhead].
// logits = (dot(kb*q_nope,latent)+dot(q_rope,key_rope))*scale. GLM-5.3's
// unscaled-RoPE attention scale is 1/sqrt(192+64), NOT 1/sqrt(192 or 512).
// Negative cell entries are padding; an entirely padded row produces zero.
void mla(const float* q, const uint16_t* kb, const uint16_t* vb,
         const uint16_t* cache, const int* cells, int nsel, int heads, int nope,
         int latent, int rope, int vhead, float scale, float* out, void* stream);

}  // namespace strata::kernels::full_glm_dsa
