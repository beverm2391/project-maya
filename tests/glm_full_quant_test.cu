// Full GLM's native quantized decode shapes, independently checked against ggml
// dequantization + CPU-double products. Unlike mv_nt parity, this catches a
// shape-dependent error shared by all GPU variants, without any model weights.
#include "strata/kernels/glm_fast.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace gf = strata::kernels::glmf;
using strata::kernels::f32_from_f16;
namespace {
void ck(cudaError_t e) {
    if (e != cudaSuccess) { std::fprintf(stderr, "full quant CUDA: %s\n", cudaGetErrorString(e)); std::exit(1); }
}
template<class T> struct Device {
    T* p = nullptr;
    size_t n;
    explicit Device(size_t count) : n(count) { ck(cudaMalloc((void**) &p, count*sizeof(T))); }
    explicit Device(const std::vector<T>& v) : Device(v.size()) { put(v); }
    ~Device() { cudaFree(p); }
    Device(const Device&) = delete;
    void put(const std::vector<T>& v) { ck(cudaMemcpy(p, v.data(), n*sizeof(T), cudaMemcpyHostToDevice)); }
    std::vector<T> get() const { std::vector<T> v(n); ck(cudaMemcpy(v.data(), p, n*sizeof(T), cudaMemcpyDeviceToHost)); return v; }
};
std::vector<float> random_values(size_t n, float scale, std::mt19937& rng) {
    std::normal_distribution<float> normal(0, scale);
    std::vector<float> x(n);
    for (auto& v : x) v = normal(rng);
    return x;
}
struct Weight {
    int type, cols, rows;
    std::vector<uint8_t> packed;
    std::vector<float> dequant;
    Weight(int t, int c, int r, float scale, std::mt19937& rng) : type(t), cols(c), rows(r) {
        auto values = random_values((size_t) c*r, scale, rng);
        packed.resize(gf::row_bytes(t, c)*(size_t) r);
        std::vector<float> imatrix(c, 1);
        const size_t wrote = ggml_quantize_chunk((ggml_type) t, values.data(), packed.data(), 0, r, c,
            ggml_quantize_requires_imatrix((ggml_type) t) ? imatrix.data() : nullptr);
        if (wrote != packed.size()) { std::fprintf(stderr, "quantized byte count mismatch\n"); std::exit(1); }
        dequant.resize(values.size());
        const auto* traits = ggml_get_type_traits((ggml_type) t);
        const size_t row = gf::row_bytes(t, c);
        for (int i = 0; i < r; ++i) traits->to_float(packed.data()+i*row, dequant.data()+(size_t) i*c, c);
    }
};
// Read the actual activation encoding, so the expected product is about kernel
// arithmetic, not differences between ggml CPU and GPU activation quantizers.
std::vector<double> q8_values(const std::vector<uint8_t>& bytes) {
    std::vector<double> out(bytes.size()/36*32);
    for (size_t b = 0; b < bytes.size()/36; ++b) {
        uint16_t d;
        std::memcpy(&d, bytes.data()+36*b, 2);
        const double scale = f32_from_f16(d);
        const auto* codes = (const int8_t*) (bytes.data()+36*b+4);
        for (int i = 0; i < 32; ++i) out[b*32+i] = scale*codes[i];
    }
    return out;
}
double dot(const float* row, const double* x, int n) {
    double result = 0;
    for (int i = 0; i < n; ++i) result += (double) row[i]*x[i];
    return result;
}
void close(const char* name, const std::vector<float>& got, const std::vector<double>& want,
           double atol = 3e-4, double rtol = 3e-5) {
    double maxerr = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        const double err = std::abs(got[i]-want[i]);
        maxerr = std::max(maxerr, err);
        if (!std::isfinite(got[i]) || err > atol+rtol*std::abs(want[i])) {
            std::fprintf(stderr, "%s[%zu] got %.9g want %.12g err %.6g\n", name, i, got[i], want[i], err);
            std::exit(1);
        }
    }
    std::printf("%s: %zu outputs, max absolute error %.6g\n", name, got.size(), maxerr);
}
void projection(int n, int m, std::mt19937& rng) {
    Weight w(GGML_TYPE_Q8_0, n, m, 0.05f, rng);
    auto x = random_values(n, 1, rng);
    Device<uint8_t> dw(w.packed), dxq((size_t) n/32*36);
    Device<float> dx(x), dy(m);
    gf::quantize_q8_1(dx.p, dxq.p, n, nullptr);
    gf::MvJob job{dw.p, dxq.p, dx.p, dy.p, nullptr, 1, GGML_TYPE_Q8_0, n, m};
    if (!gf::mv(&job, 1, nullptr)) { std::fprintf(stderr, "Q8_0 mv refused shape\n"); std::exit(1); }
    const auto q = q8_values(dxq.get());
    std::vector<double> want(m);
    for (int r = 0; r < m; ++r) want[r] = dot(w.dequant.data()+(size_t) r*n, q.data(), n);
    const std::string name = "Q8_0 projection "+std::to_string(n)+"x"+std::to_string(m);
    close(name.c_str(), dy.get(), want);
}
void experts(std::mt19937& rng) {
    constexpr int E = 6144, FF = 2048, K = 8;
    // Two distinct blobs exercised through eight plan entries. Entry 3 is absent
    // (the production null-pointer expert skip), all other plan weights differ.
    Weight gu0(GGML_TYPE_IQ4_XS, E, 2*FF, 0.1f, rng), gu1(GGML_TYPE_IQ4_XS, E, 2*FF, 0.1f, rng);
    Weight dn0(GGML_TYPE_IQ4_XS, FF, E, 0.05f, rng), dn1(GGML_TYPE_IQ4_XS, FF, E, 0.05f, rng);
    const size_t off = gu0.packed.size();
    std::vector<uint8_t> blob0 = gu0.packed, blob1 = gu1.packed;
    blob0.insert(blob0.end(), dn0.packed.begin(), dn0.packed.end());
    blob1.insert(blob1.end(), dn1.packed.begin(), dn1.packed.end());
    Device<uint8_t> db0(blob0), db1(blob1);
    std::vector<unsigned long long> ptr(K);
    std::vector<float> weights(K);
    for (int i = 0; i < K; ++i) {
        ptr[i] = i == 3 ? 0 : (unsigned long long) (i%2 ? db1.p : db0.p);
        weights[i] = 0.07f*(i+1);
    }
    Device<unsigned long long> dptr(ptr);
    Device<float> dw(weights);
    Device<int> flag(std::vector<int>{0});
    gf::MoeDev d;
    d.plan_ptr = dptr.p; d.plan_w = dw.p; d.cpu_flag = flag.p;
    auto x = random_values(E, 1, rng), shx = random_values(FF, 1, rng);
    Device<float> dx(x), dshx(shx), dout(E), dshout(E);
    Device<uint8_t> dxq(E/32*36), dhq((size_t) K*FF/32*36), dshq(FF/32*36);
    // Null entry's activation bytes must stay poisoned: gate/up may skip it and
    // down must never read it. The other entries must all be written fully.
    ck(cudaMemset(dhq.p, 0xff, dhq.n));
    gf::quantize_q8_1(dx.p, dxq.p, E, nullptr);
    gf::quantize_q8_1(dshx.p, dshq.p, FF, nullptr);
    Weight shared(GGML_TYPE_Q8_0, FF, E, 0.05f, rng);
    Device<uint8_t> dshared(shared.packed);
    gf::moe_gate_up(GGML_TYPE_IQ4_XS, d, K, E, FF, std::numeric_limits<float>::infinity(),
                    dxq.p, dhq.p, dshared.p, GGML_TYPE_Q8_0, dshq.p, FF, dshout.p, nullptr);
    gf::moe_down(GGML_TYPE_IQ4_XS, d, K, E, FF, off, dhq.p, dshout.p, dout.p, nullptr);
    const auto actual_hbytes = dhq.get();
    const auto qx = q8_values(dxq.get()), sh_qx = q8_values(dshq.get());
    // Validate each written h block against independent dequantized gate/up
    // products and UNCLAMPED SiLU. Allow one code's quantization error at a bin
    // boundary, not an arbitrary end-to-end tolerance that could hide clipping.
    int large_gate = 0;
    std::vector<double> h((size_t) K*FF);
    for (int e = 0; e < K; ++e) {
        if (ptr[e] == 0) continue;
        const auto& gu = e%2 ? gu1.dequant : gu0.dequant;
        std::vector<double> raw(FF);
        for (int i = 0; i < FF; ++i) {
            const double g = dot(gu.data()+(size_t) i*E, qx.data(), E);
            const double u = dot(gu.data()+(size_t) (FF+i)*E, qx.data(), E);
            raw[i] = g/(1+std::exp(-g))*u;
            large_gate += g > 10;
        }
        std::vector<uint8_t> bytes(actual_hbytes.begin()+(size_t) e*FF/32*36,
                                   actual_hbytes.begin()+(size_t) (e+1)*FF/32*36);
        const auto got = q8_values(bytes);
        for (int i = 0; i < FF; ++i) {
            double mx = 0;
            for (int j = i/32*32; j < i/32*32+32; ++j) mx = std::max(mx, std::abs(raw[j]));
            if (!std::isfinite(got[i]) || std::abs(got[i]-raw[i]) > mx/127*1.02+1e-4) {
                std::fprintf(stderr, "IQ4_XS GU route%d row%d got%.9g want%.9g\n", e, i, got[i], raw[i]); std::exit(1);
            }
            h[(size_t) e*FF+i] = got[i];
        }
    }
    if (large_gate == 0) { std::fprintf(stderr, "fixture never exercises unclamped gates\n"); std::exit(1); }
    std::printf("IQ4_XS GU6144x2048: all7active plan entries, %d gates >10, unclamped parity\n", large_gate);
    std::vector<double> shwant(E), want(E);
    for (int r = 0; r < E; ++r) {
        shwant[r] = dot(shared.dequant.data()+(size_t) r*FF, sh_qx.data(), FF);
        want[r] = shwant[r];
        for (int e = 0; e < K; ++e) if (ptr[e]) {
            const auto& down = e%2 ? dn1.dequant : dn0.dequant;
            want[r] += weights[e]*dot(down.data()+(size_t) r*FF, h.data()+(size_t) e*FF, FF);
        }
    }
    close("Q8_0 shared down2048x6144", dshout.get(), shwant);
    close("IQ4_XS routed down2048x6144 + shared", dout.get(), want, 8e-3, 8e-5);
}
}  // namespace
int main() {
    std::mt19937 rng(8192);
    projection(6144, 576, rng);    // KV-A includes separate RoPE key
    projection(6144, 2048, rng);   // Q-A and routed/shared GU input width
    projection(2048, 16384, rng);  // Q-B expands all64 heads
    projection(16384, 256, rng);   // attention output's input width, bounded output fixture
    experts(rng);
    ck(cudaDeviceSynchronize());
    if (gf::launch_errors()) return 1;
    std::puts("glm_full_quant_test: OK");
    return 0;
}
