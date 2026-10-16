// Metadata parsing needs no GPU calls. Exercise the full model's distinct
// normalization contracts through actual GGUF files and the production parser.
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
    Fixture(bool indexer_override, float rms = 1e-5f) {
        std::vector<char> bytes;
        append<uint32_t>(bytes, 0x46554747); // GGUF v3, no tensor payload
        append<uint32_t>(bytes, 3);
        append<uint64_t>(bytes, 0);
        append<uint64_t>(bytes, indexer_override ? 3 : 2);
        string(bytes, "general.architecture"); append<uint32_t>(bytes, 8); string(bytes, "glm-dsa");
        string(bytes, "glm-dsa.attention.layer_norm_rms_epsilon");
        append<uint32_t>(bytes, 6); append<float>(bytes, rms);
        if (indexer_override) {
            string(bytes, "glm-dsa.attention.layer_norm_epsilon");
            append<uint32_t>(bytes, 6); append<float>(bytes, 2.5e-4f);
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
} // namespace
int main() {
    check(false, 1e-5f); // Warlock header: RMS key present, LayerNorm key absent
    check(true, 1e-5f);  // optional LayerNorm metadata overrides only the indexer
    check(false, 4e-5f); // changing RMS metadata must not alter indexer LayerNorm
    std::puts("glm_geometry_test: PASS (independent RMS and indexer LayerNorm epsilon)");
}
