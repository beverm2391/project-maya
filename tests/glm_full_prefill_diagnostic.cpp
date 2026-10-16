// Actual-pack numerical diagnostics, separate from the CUDA primitive oracles.
// Decode rounds activations to Q8_1; prefill uses FP16 GEMMs for dense projections.
// Decode is therefore not a mathematical oracle for a logit-error threshold. Even
// GEMM accumulation changes can flip near-tied expert choices across 78 layers.
// Report the differences; only path, state and finite-value invariants fail here.
// Usage: glm_full_prefill_diagnostic PACK CAPACITY PREFIX_TOKENS [SPLIT]
#include "strata/core/glm_model.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {
void require(bool ok, const std::string& message) {
    if (!ok) { std::fprintf(stderr, "glm_full_prefill_diagnostic: %s\n", message.c_str()); std::exit(1); }
}
void env(const char* key, const char* value) {
#ifdef _WIN32
    _putenv_s(key, value);
#else
    setenv(key, value, 1);
#endif
}
std::vector<int32_t> tokens(int n, int start, int vocab) {
    std::vector<int32_t> result((size_t) n);
    for (int i = 0; i < n; ++i) result[(size_t) i] = 1 + (start + i * 17) % (vocab - 1);
    return result;
}
using Rows = std::vector<std::vector<float>>;
Rows picks(strata::core::Glm5Model& model, const std::vector<int32_t>& input) {
    Rows result;
    std::string error;
    for (int32_t token : input) {
        std::vector<float> logits;
        require(model.forward({token}, logits, error), error);
        result.push_back(std::move(logits));
    }
    return result;
}
void compare(const char* label, const Rows& got, const Rows& reference) {
    int agree = 0;
    require(got.size() == reference.size(), "comparison row count");
    for (size_t t = 0; t < got.size(); ++t) {
        require(got[t].size() == reference[t].size() && !got[t].empty(), "comparison vocabulary size");
        double squared = 0, norm = 0, max_abs = 0;
        int best = 0, ref_best = 0;
        for (size_t i = 0; i < got[t].size(); ++i) {
            const double a = got[t][i], b = reference[t][i];
            require(std::isfinite(a) && std::isfinite(b), "nonfinite logit");
            squared += (a-b)*(a-b); norm += b*b;
            max_abs = std::max(max_abs, std::abs(a-b));
            if (a > got[t][(size_t) best]) best = (int) i;
            if (b > reference[t][(size_t) ref_best]) ref_best = (int) i;
        }
        const double rel = std::sqrt(squared / std::max(norm, 1e-30));
        const double peak = got[t][(size_t) best], ref_peak = reference[t][(size_t) ref_best];
        double mass = 0, ref_mass = 0;
        for (size_t i = 0; i < got[t].size(); ++i) {
            mass += std::exp(got[t][i] - peak);
            ref_mass += std::exp(reference[t][i] - ref_peak);
        }
        const double logz = peak + std::log(mass), ref_logz = ref_peak + std::log(ref_mass);
        double kl = 0;
        for (size_t i = 0; i < got[t].size(); ++i) {
            const double logp = reference[t][i] - ref_logz, logq = got[t][i] - logz;
            kl += std::exp(logp) * (logp-logq);
        }
        agree += best == ref_best;
        std::printf("%s row %zu: rel_L2 %.7g max_abs %.7g KL(ref||got) %.7g top1 %d/%d\n",
                    label, t, rel, max_abs, kl, best, ref_best);
    }
    std::printf("%s: top1 agreement %d/%zu\n", label, agree, got.size());
}
Rows run(strata::core::Glm5Model& model, const std::vector<int32_t>& prefix,
         const std::vector<int32_t>& continuation, const std::vector<int32_t>& query,
         bool batched, bool split_calls) {
    model.reset();
    std::string error;
    const auto append = [&](const std::vector<int32_t>& input) {
        const int64_t before = model.position();
        if (batched) {
            // A second prefill call begins at a nonzero position. Partitioning
            // the prefix also changes expert groups without changing its tokens.
            const size_t mid = split_calls ? input.size()/2 : input.size();
            require(model.prefill(std::vector<int32_t>(input.begin(), input.begin() + (long) mid), error),
                    error.empty() ? "prefill unexpectedly fell back" : error);
            if (mid < input.size())
                require(model.prefill(std::vector<int32_t>(input.begin() + (long) mid, input.end()), error),
                        error.empty() ? "continuation prefill unexpectedly fell back" : error);
        } else {
            std::vector<float> ignored;
            require(model.forward(input, ignored, error), error);
        }
        require(model.position() == before + (int64_t) input.size(), "prefill position drift");
    };
    append(prefix);
    Rows result = picks(model, query);
    append(continuation);
    Rows later = picks(model, query);
    result.insert(result.end(), later.begin(), later.end());
    return result;
}
} // namespace
int main(int argc, char** argv) {
    require(argc == 4 || argc == 5, "usage: PACK CAPACITY PREFIX_TOKENS [SPLIT]");
    const int count = std::atoi(argv[3]);
    require(count >= 4, "prefix must contain at least four tokens");
    env("STRATA_GLM_FULL_PREFILL", "1"); env("STRATA_GLM_PREFILL", "128");
    env("STRATA_GLM_PREFILL_MIN", "1"); env("STRATA_GLM_PREFILL_TAIL_SKIP", "0");
    // The native CPU lane and CUDA dots have their own rounding differences.
    // Isolate GPU prefill arithmetic before testing placement-dependent lanes.
    env("STRATA_GLM_PREFILL_CPU", "0"); env("STRATA_GLM_PCIE_SHARE", "1");
    env("STRATA_GLM_CPU_LANE", "0");
    strata::core::Glm5Model model;
    std::string error;
    require(model.load_pack_env(argv[1], std::atoll(argv[2]), error, argc == 5 ? argv[4] : "auto"), error);
    require(model.geometry().full_dsa && model.prefill_chunk() > 0, "requires enabled full-GLM prefill");
    const auto& kinds = model.geometry().indexer_full;
    require(std::find(kinds.begin(), kinds.end(), 0) != kinds.end(), "requires shared-indexer layers");
    const int vocab = model.geometry().n_vocab;
    const auto prefix = tokens(count, 11, vocab), continuation = tokens(18, 119, vocab), query = tokens(4, 211, vocab);
    const Rows direct = run(model, prefix, continuation, query, false, false);
    const Rows batched = run(model, prefix, continuation, query, true, false);
    const Rows partitioned = run(model, prefix, continuation, query, true, true);
    // Partitioning controls causal continuation while exposing shape-dependent
    // accumulation differences. These are observations, not quality thresholds.
    compare("partitioned/batched", partitioned, batched);
    compare("batched/direct + continuation", batched, direct);
    compare("partitioned/direct + continuation", partitioned, direct);
    std::puts("glm_full_prefill_diagnostic: completed; numerical differences reported above are not a quality verdict");
}
