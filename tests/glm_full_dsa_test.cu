// Full GLM DSA primitives against independent CPU-double formulas. The reference
// uses an explicit Hadamard matrix, stable_sort top-k, and decompressed K/V MLA.
// It deliberately avoids the absorbed/butterfly/radix implementation structure.
#include "strata/kernels/glm_full_dsa.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

namespace f = strata::kernels::full_glm_dsa;
using strata::kernels::f16_from_f32;
using strata::kernels::f32_from_f16;
namespace {
void ck(cudaError_t e) {
    if (e != cudaSuccess) { std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e)); std::exit(1); }
}
template<class T> struct Device {
    T* p = nullptr;
    size_t n;
    explicit Device(size_t count) : n(count) { ck(cudaMalloc((void**) &p, n*sizeof(T))); }
    explicit Device(const std::vector<T>& v) : Device(v.size()) { put(v); }
    ~Device() { cudaFree(p); }
    void put(const std::vector<T>& v) { ck(cudaMemcpy(p, v.data(), n*sizeof(T), cudaMemcpyHostToDevice)); }
    std::vector<T> get() { std::vector<T> v(n); ck(cudaMemcpy(v.data(), p, n*sizeof(T), cudaMemcpyDeviceToHost)); return v; }
};
std::vector<float> data(size_t n, float scale = 1) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = scale*(std::sin((double) i*0.379)+0.3*std::cos((double) i*0.173));
    return v;
}
void close(const char* name, const std::vector<float>& got, const std::vector<double>& want,
           double atol = 2e-5, double rtol = 2e-5) {
    double maxerr = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        const double err = std::abs(got[i]-want[i]);
        maxerr = std::max(maxerr, err);
        if (!std::isfinite(got[i]) || err > atol+rtol*std::abs(want[i])) {
            std::fprintf(stderr, "%s[%zu] got %.9g want %.12g err %.4g\n", name, i, got[i], want[i], err);
            std::exit(1);
        }
    }
    std::printf("%s: %zu values, max absolute error %.3g\n", name, want.size(), maxerr);
}
void rope(std::vector<double>& x, int offset, int dim, int pos, f::Rope r) {
    for (int i = 0; i < dim; i += 2) {
        const double ang = pos*std::pow((double) r.base, -(double) i/dim)*r.freq_scale;
        const double c = std::cos(ang)*r.attn_factor, s = std::sin(ang)*r.attn_factor;
        const double a = x[offset+i], b = x[offset+i+1];
        x[offset+i] = a*c-b*s; x[offset+i+1] = a*s+b*c;
    }
}
void test_norm() {
    for (int n : {2048, 6144, 6176}) {
        auto h = data(n), add = data(n, 0.2f), w = data(n, 0.8f);
        Device<float> dh(h), da(add), dw(w), dx(n);
        Device<unsigned char> xq(n/32*36);
        f::residual_norm(dh.p, da.p, dw.p, 1e-5f, n, dx.p, xq.p, nullptr);
        double ss = 0;
        std::vector<double> residual(n), want(n);
        for (int i = 0; i < n; ++i) { residual[i] = (float) (h[i]+add[i]); ss += residual[i]*residual[i]; }
        for (int i = 0; i < n; ++i) want[i] = residual[i]/std::sqrt(ss/n+1e-5f)*w[i];
        close("residual", dh.get(), residual);
        close("rms 6144/generic", dx.get(), want);
        // In-place head normalization and omitted affine weights are supported.
        dh.put(h);
        f::residual_norm(dh.p, nullptr, nullptr, 1e-5f, n, dh.p, nullptr, nullptr);
        ss = 0;
        for (float v : h) ss += (double) v*v;
        for (int i = 0; i < n; ++i) want[i] = h[i]/std::sqrt(ss/n+1e-5f);
        close("rms in-place", dh.get(), want);
        f::add_residual(dh.p, da.p, n, nullptr);
        for (int i = 0; i < n; ++i) want[i] += add[i];
        close("add residual", dh.get(), want);
    }
}
void test_prepare() {
    constexpr int H = 64, N = 192, L = 512, R = 64, POS = 7;
    auto q = data(H*(N+R)), kv = data(L+R), w = data(L, 0.7f);
    Device<float> dq(q), dkv(kv), dw(w);
    Device<uint16_t> dc(std::vector<uint16_t>((POS+1)*(L+R), 0x3555));
    const f::Rope r;
    f::prepare(dq.p, dkv.p, dw.p, dc.p, POS, H, N, L, R, 1e-5f, r, nullptr);
    std::vector<double> wantq(q.begin(), q.end()), wantcache(L+R);
    for (int h = 0; h < H; ++h) rope(wantq, h*(N+R)+N, R, POS, r);
    close("MLA adjacent RoPE q64x256", dq.get(), wantq);
    double ss = 0;
    for (int i = 0; i < L; ++i) ss += (double) kv[i]*kv[i];
    for (int i = 0; i < L; ++i) wantcache[i] = kv[i]/std::sqrt(ss/L+1e-5f)*w[i];
    std::vector<double> pe(kv.begin()+L, kv.end());
    rope(pe, 0, R, POS, r);
    for (int i = 0; i < R; ++i) wantcache[L+i] = pe[i];
    auto cache = dc.get();
    std::vector<float> got(L+R);
    for (int i = 0; i < POS*(L+R); ++i)
        if (cache[i] != 0x3555) { std::fprintf(stderr, "prepare overwrote old KV\n"); std::exit(1); }
    for (int i = 0; i < L+R; ++i) got[i] = f32_from_f16(cache[POS*(L+R)+i]);
    close("FP16 latent+RoPE cache", got, wantcache, 6e-4, 6e-4);
}
void hadamard(std::vector<double>& row) {
    const int n = (int) row.size();
    std::vector<double> out(n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            unsigned bits = (unsigned) (i & j), parity = 0;
            while (bits) { parity ^= bits & 1u; bits >>= 1; }
            out[i] += (parity ? -row[j] : row[j])/std::sqrt((double) n);
        }
    row = std::move(out);
}
void test_index(float eps) {
    constexpr int H = 32, D = 128, R = 64, POS = 3, C = POS+1;
    // Low key variance makes 1e-6 versus the trunk's 1e-5 observable.
    auto q = data(H*D), k = data(D, 0.001f), w = data(D, 0.8f), b = data(D, 0.2f);
    auto cache = data(C*D, 0.7f), iw = data(H, 0.3f);
    Device<float> dq(q), dk(k), dw(w), db(b), dc(cache), di(iw), ds(C);
    const f::Rope r;
    f::index_prepare(dq.p, dk.p, dw.p, db.p, dc.p, POS, H, D, R, eps, r, nullptr);
    std::vector<double> wantq(H*D), wantk(k.begin(), k.end());
    double mean = std::accumulate(k.begin(), k.end(), 0.0)/D, ss = 0;
    for (float v : k) ss += (v-mean)*(v-mean);
    for (int i = 0; i < D; ++i) wantk[i] = (k[i]-mean)/std::sqrt(ss/D+eps)*w[i]+b[i];
    rope(wantk, 0, R, POS, r); hadamard(wantk);
    for (int h = 0; h < H; ++h) {
        std::vector<double> row(q.begin()+h*D, q.begin()+(h+1)*D);
        rope(row, 0, R, POS, r); hadamard(row);
        std::copy(row.begin(), row.end(), wantq.begin()+h*D);
    }
    close("index Q RoPE then explicit Hadamard", dq.get(), wantq);
    close("index K LayerNorm affine+RoPE+Hadamard", dk.get(), wantk);
    auto gotcache = dc.get();
    for (int i = 0; i < POS*D; ++i)
        if (gotcache[i] != cache[i]) { std::fprintf(stderr, "index overwrote old cache\n"); std::exit(1); }
    // Score reference uses independently transformed q/k and includes negative
    // head weights, so applying ReLU after head summation would fail.
    f::score(dq.p, dc.p, di.p, D, H, C, ds.p, nullptr);
    std::vector<double> wants(C);
    for (int c = 0; c < C; ++c)
        for (int h = 0; h < H; ++h) {
            double dot = 0;
            for (int i = 0; i < D; ++i) dot += wantq[h*D+i]*(c == POS ? wantk[i] : cache[c*D+i]);
            wants[c] += std::max(dot, 0.0)*iw[h]/std::sqrt((double) D*H);
        }
    close("individual-cell score + raw weight scaling", ds.get(), wants, 5e-5, 5e-5);
}
void test_select() {
    for (int count : {1, 17, 2048, 2049, 4097, 32769}) {
        std::vector<float> scores(count);
        for (int i = 0; i < count; ++i) scores[i] = (float) ((i*7919)%97-48); // many cutoff ties
        scores[0] = -0.0f;
        for (int top : {1, 13, 2048}) {
            const int take = std::min(top, count);
            Device<float> ds(scores);
            Device<int> dc(take+1);
            const std::vector<int> sentinels(take+1, -77);
            dc.put(sentinels);
            f::select(ds.p, count, top, dc.p, nullptr);
            std::vector<int> order(count);
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return scores[a] > scores[b]; });
            auto got = dc.get();
            if (!std::equal(got.begin(), got.begin()+take, order.begin()) || got[take] != -77) {
                std::fprintf(stderr, "top-k failed count=%d top=%d\n", count, top); std::exit(1);
            }
        }
    }
    std::printf("individual-cell top2048: stable CPU sort parity, short and 32769-token cache\n");
}
uint16_t bf_bits(float x) { uint32_t u; std::memcpy(&u, &x, 4); return (uint16_t) (u >> 16); }
double bf_value(uint16_t h) { uint32_t u = (uint32_t) h << 16; float f; std::memcpy(&f, &u, 4); return f; }
void test_mla(int H, int N, int L, int R, int V, int S, bool all_padding = false) {
    const int C = S+2;
    auto q = data(H*(N+R), 0.11f), kw = data((size_t) H*L*N, 0.03f), vw = data((size_t) H*V*L, 0.07f);
    auto lat = data((size_t) C*(L+R), 0.4f);
    std::vector<uint16_t> kb(kw.size()), vb(vw.size()), cache(lat.size());
    for (size_t i = 0; i < kb.size(); ++i) kb[i] = bf_bits(kw[i]);
    for (size_t i = 0; i < vb.size(); ++i) vb[i] = bf_bits(vw[i]);
    for (size_t i = 0; i < cache.size(); ++i) cache[i] = f16_from_f32(lat[i]);
    std::vector<int> cells(S);
    for (int s = 0; s < S; ++s) cells[s] = all_padding || (s%7 == 5) ? -1 : S-s;
    Device<float> dq(q), dout(H*V);
    Device<uint16_t> dk(kb), dv(vb), dc(cache);
    Device<int> ds(cells);
    const float scale = 1/std::sqrt((float) (N+R));
    f::mla(dq.p, dk.p, dv.p, dc.p, ds.p, S, H, N, L, R, V, scale, dout.p, nullptr);
    std::vector<double> want(H*V);
    // Decompress each selected token's per-head K and V first. This directly
    // checks absorption and rejects confusing FP16 with BF16 projection bits.
    for (int h = 0; h < H; ++h) {
        std::vector<double> probs(S, -INFINITY), values((size_t) S*V);
        double mx = -INFINITY, denom = 0;
        for (int s = 0; s < S; ++s) if (cells[s] >= 0) {
            const size_t row = (size_t) cells[s]*(L+R);
            double dot = 0;
            for (int i = 0; i < N; ++i) {
                double key = 0;
                for (int c = 0; c < L; ++c) key += bf_value(kb[((size_t) h*L+c)*N+i])*f32_from_f16(cache[row+c]);
                dot += q[h*(N+R)+i]*key;
            }
            for (int i = 0; i < R; ++i) dot += q[h*(N+R)+N+i]*f32_from_f16(cache[row+L+i]);
            probs[s] = dot*scale;
            mx = std::max(mx, probs[s]);
            for (int v = 0; v < V; ++v)
                for (int c = 0; c < L; ++c)
                    values[(size_t) s*V+v] += bf_value(vb[((size_t) h*V+v)*L+c])*f32_from_f16(cache[row+c]);
        }
        for (int s = 0; s < S; ++s) {
            probs[s] = cells[s] < 0 ? 0 : std::exp(probs[s]-mx);
            denom += probs[s];
        }
        for (int v = 0; v < V; ++v)
            for (int s = 0; s < S; ++s)
                want[h*V+v] += denom > 0 ? probs[s]/denom*values[(size_t) s*V+v] : 0;
    }
    close("MLA vs decompressed K/V (BF16 weights, FP16 KV)", dout.get(), want, 3e-5, 3e-5);
}
}  // namespace

int main() {
    test_norm(); test_prepare(); test_index(1e-6f); test_index(2.5e-4f); test_select();
    test_mla(3, 192, 512, 64, 256, 11); // real full-model dimensions
    test_mla(2, 4, 8, 4, 6, 2048);     // full top-k, padding, distinct rope/latent terms
    test_mla(1, 4, 8, 4, 6, 13, true);
    ck(cudaDeviceSynchronize());
    std::puts("glm_full_dsa_test: OK");
    return 0;
}
