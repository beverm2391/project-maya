// Actual full-GLM IQ4_XS expert math, without loading the model or changing its
// runtime. CLI: glm_full_expert_oracle shard.gguf input.f32 [layer=3] [expert=229]
// Reads one expert, uses existing MMQ, and independently dequantizes its weights
// for CPU-double products. IQ4_XS MMQ's D4 activation scales are F32, not decode's
// FP16 q8_1 scales. Product error and input-quantization error are reported apart.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "ggml.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace mmq = strata::prefill::mmq;
namespace {
void require(bool yes, const char* why) {
    if (!yes) { std::fprintf(stderr, "expert oracle: %s\n", why); std::exit(1); }
}
void ck(cudaError_t e) { require(e == cudaSuccess, cudaGetErrorString(e)); }
template<class T> struct Device {
    T* p = nullptr;
    size_t n;
    explicit Device(size_t count) : n(count) { ck(cudaMalloc((void**) &p, n*sizeof(T))); }
    explicit Device(const std::vector<T>& v) : Device(v.size()) {
        ck(cudaMemcpy(p, v.data(), n*sizeof(T), cudaMemcpyHostToDevice));
    }
    ~Device() { cudaFree(p); }
    Device(const Device&) = delete;
    std::vector<T> get() const { std::vector<T> v(n); ck(cudaMemcpy(v.data(), p, n*sizeof(T), cudaMemcpyDeviceToHost)); return v; }
};
constexpr int E = 6144, FF = 2048, TYPE = GGML_TYPE_IQ4_XS;
struct Matrix { int rows, cols; std::vector<uint8_t> bytes; std::vector<float> values; };
Matrix read_expert(const strata::GgufFile& g, const std::string& path, const std::string& name,
                   int expert, int rows, int cols) {
    const auto* t = g.find(name);
    require(t && t->type == TYPE && t->shape.size() == 3, "missing or non-IQ4_XS expert tensor");
    require(t->shape[0] == (uint64_t) cols && t->shape[1] == (uint64_t) rows &&
            expert >= 0 && expert < (int) t->shape[2], "expert geometry mismatch");
    Matrix m{rows, cols, {}, {}};
    m.bytes.resize(ggml_row_size((ggml_type) TYPE, cols)*(size_t) rows);
    std::ifstream file(path, std::ios::binary);
    file.seekg((std::streamoff) (g.data_start()+t->offset+(uint64_t) expert*m.bytes.size()));
    file.read((char*) m.bytes.data(), (std::streamsize) m.bytes.size());
    require((bool) file, "short expert read");
    m.values.resize((size_t) rows*cols);
    ggml_get_type_traits((ggml_type) TYPE)->to_float(m.bytes.data(), m.values.data(), m.values.size());
    std::printf("source %s expert%d: %dx%d IQ4_XS, %zu bytes\n", name.c_str(), expert, rows, cols, m.bytes.size());
    return m;
}
// Pinned ggml MMQ D4: 128 codes + four F32 scales per 144-byte block, each
// scale applies to 32 codes. With one token the column blocks are contiguous.
std::vector<double> d4_values(const std::vector<uint8_t>& bytes, const std::vector<float>& source) {
    std::vector<double> out(source.size());
    require(source.size()%128 == 0 && mmq::q8_block_bytes() == 144, "unexpected MMQ D4 block geometry");
    for (size_t i = 0; i < source.size(); ++i) {
        const size_t block = i/128, lane = i%128;
        float scale;
        std::memcpy(&scale, bytes.data()+block*144+(lane/32)*4, 4);
        const int8_t code = *(const int8_t*) (bytes.data()+block*144+16+lane);
        out[i] = (double) scale*code;
        float amax = 0;
        for (size_t j = i/32*32; j < i/32*32+32; ++j) amax = std::max(amax, std::abs(source[j]));
        require(std::isfinite(out[i]) && std::abs(out[i]-source[i]) <= amax/254.0+1e-6*(1+amax),
                "MMQ input reconstruction exceeds nearest-code error bound");
    }
    return out;
}
std::vector<double> product(const Matrix& w, const std::vector<double>& x) {
    require((int) x.size() == w.cols, "CPU product input width");
    std::vector<double> out(w.rows);
    for (int r = 0; r < w.rows; ++r)
        for (int c = 0; c < w.cols; ++c) out[r] += (double) w.values[(size_t) r*w.cols+c]*x[c];
    return out;
}
void report(const char* label, const std::vector<float>& got, const std::vector<double>& want, bool assert_product) {
    require(got.size() == want.size(), "comparison shape");
    double ss = 0, ref = 0, max_error = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double error = std::abs(got[i]-want[i]);
        require(std::isfinite(got[i]) && std::isfinite(want[i]), "nonfinite product");
        ss += error*error; ref += want[i]*want[i]; max_error = std::max(max_error, error);
        if (assert_product) require(error <= 3e-4+3e-5*std::abs(want[i]), "MMQ product exceeds FP32 math tolerance");
    }
    std::printf("%s: n=%zu max_abs=%.9g relative_L2=%.9g%s\n", label, got.size(), max_error,
        std::sqrt(ss/std::max(ref, 1e-300)), assert_product ? " PASS" : " (includes activation quantization)");
}
std::vector<double> swiglu(const std::vector<double>& gate, const std::vector<double>& up) {
    std::vector<double> out(FF);
    for (int i = 0; i < FF; ++i) out[i] = gate[i]/(1+std::exp(-gate[i]))*up[i];
    return out;
}
mmq::Product gpu_product(const void* weights, int rows, int cols, const void* xq, const int* bounds,
                         const int* ids, float* out, int ld) {
    mmq::Product p;
    p.w = weights; p.type = TYPE; p.w_rows = rows; p.w_cols = cols;
    p.expert_bytes = mmq::matrix_bytes(TYPE, rows, cols); p.n = 1; p.xq = xq;
    p.bounds = bounds; p.ids = ids; p.total_rows = p.max_rows = 1; p.dst = out; p.ld_dst = ld;
    return p;
}
} // namespace
int main(int argc, char** argv) {
    require(argc >= 3 && argc <= 5, "usage: glm_full_expert_oracle shard.gguf input.f32 [layer=3] [expert=229]");
    const int layer = argc > 3 ? std::atoi(argv[3]) : 3, expert = argc > 4 ? std::atoi(argv[4]) : 229;
    strata::GgufFile source(argv[1]);
    const std::string prefix = "blk."+std::to_string(layer)+".";
    auto gate = read_expert(source, argv[1], prefix+"ffn_gate_exps.weight", expert, FF, E);
    auto up = read_expert(source, argv[1], prefix+"ffn_up_exps.weight", expert, FF, E);
    auto down = read_expert(source, argv[1], prefix+"ffn_down_exps.weight", expert, E, FF);
    std::vector<float> x(E);
    std::ifstream input(argv[2], std::ios::binary);
    input.read((char*) x.data(), E*sizeof(float));
    require((bool) input && input.peek() == std::char_traits<char>::eof(), "input must contain exactly one 6144-wide F32 row");
    require(mmq::built() && mmq::supported(TYPE) && mmq::glu_supported(TYPE, FF), "MMQ IQ4_XS GLU not available");
    std::vector<uint8_t> gu = gate.bytes;
    gu.insert(gu.end(), up.bytes.begin(), up.bytes.end());
    Device<uint8_t> dgu(gu), dd(down.bytes), dxq(mmq::q8_bytes(1,E)), dhq(mmq::q8_bytes(1,FF));
    Device<float> dx(x), dgu_out(2*FF), dh_out(2*FF), dout(E);
    Device<int> bounds(std::vector<int>{0,1}), ids(std::vector<int>{0});
    mmq::Context context;
    mmq::quantize(dx.p, nullptr, dxq.p, TYPE, E, E, 1, nullptr);
    auto gp = gpu_product(dgu.p, 2*FF, E, dxq.p, bounds.p, ids.p, dgu_out.p, 2*FF);
    context.run(gp, nullptr);
    gp.dst = dh_out.p;
    context.run_glu(gp, std::numeric_limits<float>::infinity(), nullptr);
    mmq::quantize(dh_out.p, nullptr, dhq.p, TYPE, FF, 2*FF, 1, nullptr);
    auto dp = gpu_product(dd.p, E, FF, dhq.p, bounds.p, ids.p, dout.p, E);
    context.run(dp, nullptr);
    ck(cudaDeviceSynchronize());
    const auto quant_x = d4_values(dxq.get(), x);
    const std::vector<double> raw_x(x.begin(),x.end());
    const auto qg = product(gate,quant_x), qu = product(up,quant_x);
    const auto raw_g = product(gate,raw_x), raw_u = product(up,raw_x);
    const auto gpu_gu = dgu_out.get();
    const std::vector<float> gpu_g(gpu_gu.begin(),gpu_gu.begin()+FF), gpu_u(gpu_gu.begin()+FF,gpu_gu.end());
    report("MMQ gate vs independent D4 product",gpu_g,qg,true);
    report("MMQ up vs independent D4 product",gpu_u,qu,true);
    report("MMQ gate vs unquantized F32 input math",gpu_g,raw_g,false);
    report("MMQ up vs unquantized F32 input math",gpu_u,raw_u,false);
    auto gpu_h = dh_out.get(); gpu_h.resize(FF);
    report("MMQ fused SwiGLU vs independent D4 math",gpu_h,swiglu(qg,qu),true);
    const auto qh = d4_values(dhq.get(),gpu_h);
    const auto gpu_down = dout.get();
    report("MMQ down vs independent D4 product",gpu_down,product(down,qh),true);
    report("MMQ chain vs unquantized expert math",gpu_down,product(down,swiglu(raw_g,raw_u)),false);
    std::puts("glm_full_expert_oracle: OK");
}
