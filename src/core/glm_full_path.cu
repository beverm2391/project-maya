// Full GLM-5.3 uses the native expert tier machinery but has a different
// attention/residual contract from Flash. Keep that contract explicit here.
#include "glm_fast_state.hpp"
#include "strata/kernels/glm_full_dsa.hpp"

#include <cmath>

namespace strata::core {
namespace fd = strata::kernels::full_glm_dsa;

bool Glm5Model::fast_full_dsa(int il, int64_t p, std::string& err) {
    auto* F = fast_;
    const auto& g = g_;
    const auto& L = F->L[(size_t) il];
    const auto s = F->cs;
    const bool index = g.indexer_full[(size_t) il] != 0;
    gf::MvJob j[4];
    j[0] = {L.q_a.q, F->xq, F->x, F->qr_raw, nullptr, 1.0f, L.q_a.type, g.n_embd, g.q_lora};
    j[1] = {L.kv_a.q, F->xq, F->x, F->kv_raw, nullptr, 1.0f, L.kv_a.type, g.n_embd, g.kv_lora + g.rope_dim};
    int nj = 2;
    if (index) {
        j[nj++] = {L.idx_k, nullptr, F->x, F->ik_raw, nullptr, 1.0f, gf::kTypeBF16, g.n_embd, g.idx_key};
        j[nj++] = {L.idx_proj, nullptr, F->x, F->iw, nullptr, 1.0f, gf::kTypeBF16, g.n_embd, g.idx_heads};
    }
    if (!gf::mv(j, nj, s)) { err = "glm-dsa: Q/KV/index projections"; return false; }
    fd::residual_norm(F->qr_raw, nullptr, L.q_a_norm, g.norm_eps, g.q_lora, F->qr, F->qr_q, s);
    j[0] = {L.q_b.q, F->qr_q, F->qr, F->q, nullptr, 1.0f, L.q_b.type,
            g.q_lora, g.n_head * (g.qk_nope + g.rope_dim)};
    nj = 1;
    if (index) j[nj++] = {L.idx_q_b, nullptr, F->qr, F->iq, nullptr, 1.0f, gf::kTypeBF16,
                          g.q_lora, g.idx_heads * g.idx_key};
    if (!gf::mv(j, nj, s)) { err = "glm-dsa: Q/index expansion"; return false; }
    const fd::Rope rope{g.rope_base, g.rope_scale, 1.0f};
    auto* lat = (uint16_t*) (state_ + dsa_lat_[(size_t) il]);
    fd::prepare(F->q, F->kv_raw, L.kv_a_norm, lat, (int) p, g.n_head, g.qk_nope,
                g.kv_lora, g.rope_dim, g.norm_eps, rope, s);
    const int nsel = (int) std::min<int64_t>(p + 1, g.idx_top_k);
    if (index) {
        auto* keys = state_ + dsa_ik_[(size_t) il];
        fd::index_prepare(F->iq, F->ik_raw, L.k_norm_w, L.k_norm_b, keys, (int) p,
                          g.idx_heads, g.idx_key, g.rope_dim, g.norm_eps, rope, s);
        fd::score(F->iq, keys, F->iw, g.idx_key, g.idx_heads, (int) p + 1, F->score, s);
        fd::select(F->score, (int) p + 1, g.idx_top_k, F->cells, s);
    }
    // A shared indexer reuses the previous full layer's exact selected positions.
    // Every split starts at a full layer, so this state never crosses a device.
    fd::mla(F->q, L.k_b, L.v_b, lat, F->cells, nsel, g.n_head, g.qk_nope, g.kv_lora,
            g.rope_dim, g.v_head, 1.0f / std::sqrt((float) (g.qk_nope + g.rope_dim)), F->full_attn, s);
    gf::quantize_q8_1(F->full_attn, F->attn_q, g.n_head * g.v_head, s);
    gf::MvJob out{L.out.q, F->attn_q, F->full_attn, F->mixer, nullptr, 1.0f, L.out.type,
                  g.n_head * g.v_head, g.n_embd};
    if (!gf::mv(&out, 1, s)) { err = "glm-dsa: attention output"; return false; }
    return true;
}

bool Glm5Model::fast_full_layers(int64_t p, std::string& err) {
    auto* F = fast_;
    const auto& g = g_;
    const auto s = F->cs;
    bool pf_pending = false;
    for (int il = l0_; il < l1_; ++il) {
        const auto& L = F->L[(size_t) il];
        fd::residual_norm(state_, nullptr, L.attn_norm, g.norm_eps, g.n_embd, F->x, F->xq, s);
        if (!fast_full_dsa(il, p, err)) return false;
        fd::residual_norm(state_, F->mixer, L.ffn_norm, g.norm_eps, g.n_embd, F->x, F->xq, s);
        if (L.moe) {
            if (!fast_moe(il, pf_pending, err)) return false;
        } else {
            gf::MvJob gu[2] = {
                {L.ffn_gate.q, F->xq, F->x, F->dg, nullptr, 1.0f, L.ffn_gate.type, g.n_embd, g.n_ff_dense},
                {L.ffn_up.q, F->xq, F->x, F->du, nullptr, 1.0f, L.ffn_up.type, g.n_embd, g.n_ff_dense}};
            if (!gf::mv(gu, 2, s)) { err = "glm-dsa: dense FFN gate/up"; return false; }
            gf::swiglu_q8(F->dg, F->du, g.swiglu_shexp, g.n_ff_dense, F->dhq, s);
            gf::MvJob dn{L.ffn_down.q, F->dhq, nullptr, F->ffn, nullptr, 1.0f, L.ffn_down.type,
                         g.n_ff_dense, g.n_embd};
            if (!gf::mv(&dn, 1, s)) { err = "glm-dsa: dense FFN down"; return false; }
        }
        fd::add_residual(state_, F->ffn, g.n_embd, s);
        if (F->prof_on) F->mark("full_layer");
    }
    return glmfast::cuda_ok(cudaGetLastError(), "glm-dsa layers", err);
}
} // namespace strata::core
