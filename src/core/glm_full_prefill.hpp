// Full GLM's attention side of the shared prompt engine. GEMM and expert
// staging remain owned by glm_prefill.cu; the row grid retains the decode
// primitives' causal selection and head-local attention arithmetic.
#pragma once
#include "strata/core/glm_model.hpp"
#include "strata/kernels/glm_batch.hpp"
#include "strata/kernels/glm_full_batch.hpp"
#include <cmath>

namespace strata::core::glmfast {
template<class Layer, class Buffers, class QGemm, class B16Gemm>
void full_prefill_attention(const Glm5Geometry& g, const Layer& layer, Buffers& b, bool index,
                            const float* x, const uint16_t* x16, uint16_t* lat, float* keys,
                            int* shared_cells, int p0, int rows, float* mixer, bool skip_output,
                            QGemm& qgemm, B16Gemm& b16gemm, cudaStream_t stream) {
    namespace fd = strata::kernels::full_glm_dsa;
    namespace gb = strata::kernels::glmb;
    const int E = g.n_embd, Q = g.n_head * (g.qk_nope + g.rope_dim), V = g.n_head * g.v_head;
    qgemm(layer.q_a, g.q_lora, E, x16, E, b.qr_raw, g.q_lora, rows, 0.0f);
    qgemm(layer.kv_a, g.kv_lora + g.rope_dim, E, x16, E, b.kv_raw,
           g.kv_lora + g.rope_dim, rows, 0.0f);
    fd::residual_norm_rows(b.qr_raw, nullptr, layer.q_a_norm, g.norm_eps, rows, g.q_lora,
                           b.qr, b.qr16, stream);
    qgemm(layer.q_b, Q, g.q_lora, b.qr16, g.q_lora, b.q, Q, rows, 0.0f);
    const fd::Rope rope{g.rope_base, g.rope_scale, 1.0f};
    fd::prepare_rows(b.q, b.kv_raw, layer.kv_a_norm, lat, p0, rows, g.n_head,
                     g.qk_nope, g.kv_lora, g.rope_dim, g.norm_eps, rope, stream);
    if (index) {
        b16gemm(layer.idx_k, g.idx_key, E, x, E, b.ik_raw, g.idx_key, rows, 1.0f);
        b16gemm(layer.idx_proj, g.idx_heads, E, x, E, b.iw, g.idx_heads, rows, 1.0f);
        b16gemm(layer.idx_q_b, g.idx_heads * g.idx_key, g.q_lora, b.qr, g.q_lora,
                 b.iq, g.idx_heads * g.idx_key, rows, 1.0f);
        // All keys may be written first, but every query sees only [0, p0+t].
        // shared_cells lives across layers, outside the scratch union used by MoE.
        fd::index_rows(b.iq, b.ik_raw, layer.k_norm_w, layer.k_norm_b, keys, b.iw,
                       p0, rows, g.idx_heads, g.idx_key, g.rope_dim, g.idx_top_k,
                       g.idx_norm_eps, rope, b.score, shared_cells, stream);
    }
    fd::mla_rows(b.q, layer.k_b, layer.v_b, lat, shared_cells, p0, rows, g.idx_top_k,
                 g.n_head, g.qk_nope, g.kv_lora, g.rope_dim, g.v_head,
                 1.0f / std::sqrt((float) (g.qk_nope + g.rope_dim)), b.attn, stream);
    if (!skip_output) {
        gb::f32_to_f16(b.attn, b.attn16, (int64_t) rows * V, stream);
        qgemm(layer.out, E, V, b.attn16, V, mixer, E, rows, 0.0f);
    }
}
} // namespace strata::core::glmfast
