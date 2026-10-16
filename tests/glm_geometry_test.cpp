// Metadata parsing and cache sizing need no GPU calls. Exercise full GLM's
// normalization and shared-indexer contracts through the production parser.
#include "strata/core/glm_model.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {
template<class T> void append(std::vector<char>& bytes, T value) {
    const char* p = reinterpret_cast<const char*>(&value);
    bytes.insert(bytes.end(), p, p + sizeof(value));
}
void string(std::vector<char>& bytes, const std::string& value) {
    append<uint64_t>(bytes, value.size());
    bytes.insert(bytes.end(), value.begin(), value.end());
}
void require(bool ok, const std::string& message) {
    if (!ok) { std::fprintf(stderr, "glm_geometry_test: %s\n", message.c_str()); std::exit(1); }
}
struct Fixture {
    std::filesystem::path path;
    Fixture(bool indexer_override, float rms = 1e-5f, bool shared_indexers = false) {
        std::vector<char> bytes;
        append<uint32_t>(bytes, 0x46554747); // GGUF v3, no tensor payload
        append<uint32_t>(bytes, 3);
        append<uint64_t>(bytes, 0);
        append<uint64_t>(bytes, 2 + (indexer_override ? 1 : 0) + (shared_indexers ? 1 : 0));
        string(bytes, "general.architecture"); append<uint32_t>(bytes, 8); string(bytes, "glm-dsa");
        string(bytes, "glm-dsa.attention.layer_norm_rms_epsilon");
        append<uint32_t>(bytes, 6); append<float>(bytes, rms);
        if (indexer_override) {
            string(bytes, "glm-dsa.attention.layer_norm_epsilon");
            append<uint32_t>(bytes, 6); append<float>(bytes, 2.5e-4f);
        }
        if (shared_indexers) {
            string(bytes, "glm-dsa.attention.indexer.types");
            append<uint32_t>(bytes, 9); // bool[78], including flags beyond get()'s 64-item sample
            append<uint32_t>(bytes, 7); append<uint64_t>(bytes, 78);
            for (int l = 0; l < 78; ++l)
                append<uint8_t>(bytes, l < 3 || (l >= 6 && (l-6)%4 == 0));
        }
        bytes.resize(bytes.size() + 32, 0);
        path = std::filesystem::temp_directory_path() /
               ("maya-glm-geometry-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".gguf");
        std::ofstream out(path, std::ios::binary);
        out.write(bytes.data(), (std::streamsize) bytes.size());
        require(out.good(), "cannot write metadata fixture");
    }
    ~Fixture() { std::error_code err; std::filesystem::remove(path, err); }
};
void check(bool override, float rms) {
    Fixture fixture(override, rms);
    strata::GgufFile file(fixture.path.string());
    strata::core::Glm5Geometry geometry;
    std::string error;
    require(strata::core::Glm5Geometry::from_gguf(file, geometry, error), error);
    require(geometry.full_dsa, "fixture did not select full GLM");
    require(geometry.norm_eps == rms, "RMS epsilon changed while parsing LayerNorm");
    require(geometry.idx_norm_eps == (override ? 2.5e-4f : 1e-6f),
            "indexer LayerNorm epsilon did not use its independent default/override");
}
void cache_costs() {
    // Warlock's 78 trunk layers: 21 full indexers, 57 shared. Capacity must
    // charge exactly the state the allocator retains, including nonzero RoPE.
    Fixture fixture(false, 1e-5f, true);
    strata::GgufFile file(fixture.path.string());
    strata::core::Glm5Geometry g;
    std::string error;
    require(strata::core::Glm5Geometry::from_gguf(file, g, error), error);
    require(g.n_layers == 78 && g.indexer_full[74] && !g.indexer_full[77], "shared-indexer geometry");
    for (const int64_t context : {262144ll, 1048576ll}) {
        require(g.full_dsa_cache_bytes(2, context) == (uint64_t) context*1664,
                "full indexer must include FP16 latent/RoPE and F32 keys");
        require(g.full_dsa_cache_bytes(3, context) == (uint64_t) context*1152,
                "shared indexer must not allocate a key cache");
        uint64_t first = 0, last = 0;
        for (int l = 0; l < 78; ++l) {
            const uint64_t bytes = g.full_dsa_cache_bytes(l, context);
            if (l < 42) first += bytes;
            else last += bytes;
        }
        require(first == (uint64_t) context*54528 && last == (uint64_t) context*46080,
                "42/36 split state differs from the allocator geometry");
        require(first + last == (uint64_t) context*100608, "full model cache total");
    }
}
} // namespace
int main() {
    check(false, 1e-5f); // Warlock header: RMS key present, LayerNorm key absent
    check(true, 1e-5f);  // optional LayerNorm metadata overrides only the indexer
    check(false, 4e-5f); // changing RMS metadata must not alter indexer LayerNorm
    cache_costs();
    std::puts("glm_geometry_test: PASS (normalization metadata and shared-indexer cache costs)");
}
