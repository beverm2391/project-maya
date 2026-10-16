// Full prefill boundary: independent CPU RMS plus batched-vs-tokenwise full DSA
// across causality, continuation and shared-indexer attention. No model needed.
#include "strata/kernels/glm_full_batch.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace f = strata::kernels::full_glm_dsa;
using strata::kernels::f16_from_f32;
using strata::kernels::f32_from_f16;
namespace {
void ck(cudaError_t e) {
    if (e != cudaSuccess) { std::fprintf(stderr, "batch CUDA: %s\n", cudaGetErrorString(e)); std::exit(1); }
}
template<class T> struct Device {
    T* p = nullptr;
    size_t n;
    explicit Device(size_t size) : n(size) { ck(cudaMalloc((void**) &p, n*sizeof(T))); }
    explicit Device(const std::vector<T>& v) : Device(v.size()) { put(v); }
    ~Device() { cudaFree(p); }
    Device(const Device&) = delete;
    void put(const std::vector<T>& v) { ck(cudaMemcpy(p, v.data(), n*sizeof(T), cudaMemcpyHostToDevice)); }
    std::vector<T> get() const { std::vector<T> v(n); ck(cudaMemcpy(v.data(), p, n*sizeof(T), cudaMemcpyDeviceToHost)); return v; }
};
std::vector<float> data(size_t n, float scale = 1, double offset = 0) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i)
        v[i] = scale*(std::sin((i+offset)*0.379)+0.3*std::cos((i+offset)*0.173));
    return v;
}
void close(const char* name, const std::vector<float>& got, const std::vector<double>& want,
           double tol = 3e-5) {
    double mx = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        const double error = std::abs(got[i]-want[i]);
        mx = std::max(mx, error);
        if (!std::isfinite(got[i]) || error > tol*(1+std::abs(want[i]))) {
            std::fprintf(stderr, "%s[%zu] got%.9g want%.12g\n", name, i, got[i], want[i]); std::exit(1);
        }
    }
    std::printf("%s: %zu values, max error %.4g\n", name, got.size(), mx);
}
void same(const char* name, const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || std::memcmp(a.data(), b.data(), a.size()*sizeof(float))) {
        std::fprintf(stderr, "%s: outputs are not bit-identical\n", name); std::exit(1);
    }
    close(name, a, std::vector<double>(b.begin(), b.end()), 0);
}
void norm_rows() {
    constexpr int T = 5;
    for (int E : {2048, 4096, 6144, 6176}) {
        auto h = data((size_t) T*E), add = data((size_t) T*E, 0.2f, 7), w = data(E, 0.7f, 11);
        Device<float> dh(h), da(add), dw(w), dx(h.size());
        Device<uint16_t> dx16(h.size());
        f::residual_norm_rows(dh.p, da.p, dw.p, 1e-5f, T, E, dx.p, dx16.p, nullptr);
        std::vector<double> residual(h.size()), want(h.size());
        for (int t = 0; t < T; ++t) {
            double ss = 0;
            for (int e = 0; e < E; ++e) {
                const size_t i = (size_t) t*E+e;
                residual[i] = (float) (h[i]+add[i]); ss += residual[i]*residual[i];
            }
            for (int e = 0; e < E; ++e) {
                const size_t i = (size_t) t*E+e;
                want[i] = residual[i]/std::sqrt(ss/E+1e-5f)*w[e];
            }
        }
        close("batched residual", dh.get(), residual);
        const auto got = dx.get();
        close("batched RMS incl 6144 and tail", got, want);
        const auto half = dx16.get();
        for (size_t i = 0; i < half.size(); ++i)
            if (half[i] != f16_from_f32(got[i])) { std::fprintf(stderr, "FP16 result mismatch\n"); std::exit(1); }
        // Omitted add/affine/FP16 output and h==x use the same row ownership.
        dh.put(h);
        f::residual_norm_rows(dh.p, nullptr, nullptr, 1e-5f, T, E, dh.p, nullptr, nullptr);
        for (int t = 0; t < T; ++t) {
            double ss = 0;
            for (int e = 0; e < E; ++e) { const double v = h[(size_t) t*E+e]; ss += v*v; }
            for (int e = 0; e < E; ++e) want[(size_t) t*E+e] = h[(size_t) t*E+e]/std::sqrt(ss/E+1e-5f);
        }
        close("batched RMS in-place", dh.get(), want);
        f::add_residual_rows(dh.p, da.p, T, E, nullptr);
        for (size_t i = 0; i < want.size(); ++i) want[i] += add[i];
        close("batched residual add", dh.get(), want);
    }
}
std::vector<uint16_t> bf16(const std::vector<float>& values) {
    std::vector<uint16_t> out(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        uint32_t u; std::memcpy(&u, &values[i], 4); out[i] = (uint16_t) (u >> 16);
    }
    return out;
}
void attention(int p0, int T, int topk) {
    constexpr int H = 2, N = 192, L = 512, R = 64, V = 256, IH = 32, IK = 128;
    const int C = p0+T+3;
    auto q = data((size_t) T*H*(N+R), 0.1f), kv = data((size_t) T*(L+R), 0.2f, 4);
    auto iq = data((size_t) T*IH*IK, 0.3f, 7), ik = data((size_t) T*IK, 0.2f, 13);
    auto iw = data((size_t) T*IH, 0.4f, 3), lnw = data(IK, 0.8f, 17), lnb = data(IK, 0.2f, 9);
    auto norm = data(L, 0.7f, 11), prefix_keys = data((size_t) C*IK, 0.3f, 8);
    auto kb = bf16(data((size_t) H*L*N, 0.03f, 2)), vb = bf16(data((size_t) H*V*L, 0.04f, 5));
    auto prefix_values = data((size_t) C*(L+R), 0.3f, 19);
    std::vector<uint16_t> prefix_lat(prefix_values.size());
    for (size_t i = 0; i < prefix_lat.size(); ++i) prefix_lat[i] = f16_from_f32(prefix_values[i]);
    // Every unwritten cache entry is poisoned. The batched path fills future
    // chunk positions before serving earlier queries; tokenwise fills on demand.
    for (int c = p0; c < C; ++c) {
        std::fill(prefix_lat.begin()+(size_t) c*(L+R), prefix_lat.begin()+(size_t) (c+1)*(L+R), 0x7e00);
        std::fill(prefix_keys.begin()+(size_t) c*IK, prefix_keys.begin()+(size_t) (c+1)*IK, 1e20f);
    }
    Device<float> bq(q), rq(q), bkv(kv), riq(iq), biq(iq), bik(ik), rik(ik), dw(norm), diw(iw), dlw(lnw), dlb(lnb);
    Device<float> bkeys(prefix_keys), rkeys(prefix_keys), bs(C), rs(C), bout((size_t) T*H*V), rout((size_t) T*H*V);
    Device<uint16_t> blat(prefix_lat), rlat(prefix_lat), dkb(kb), dvb(vb);
    Device<int> bc(std::vector<int>((size_t) T*topk, -99)), rc(std::vector<int>((size_t) T*topk, -1));
    const f::Rope rope;
    f::prepare_rows(bq.p, bkv.p, dw.p, blat.p, p0, T, H, N, L, R, 1e-5f, rope, nullptr);
    f::index_rows(biq.p, bik.p, dlw.p, dlb.p, bkeys.p, diw.p, p0, T, IH, IK, R, topk,
                  1e-5f, rope, bs.p, bc.p, nullptr);
    f::mla_rows(bq.p, dkb.p, dvb.p, blat.p, bc.p, p0, T, topk, H, N, L, R, V, 0.0625f, bout.p, nullptr);
    for (int t = 0; t < T; ++t) {
        const int pos = p0+t, ns = std::min(pos+1, topk);
        f::prepare(rq.p+(size_t) t*H*(N+R), bkv.p+(size_t) t*(L+R), dw.p, rlat.p, pos, H, N, L, R, 1e-5f, rope, nullptr);
        f::index_prepare(riq.p+(size_t) t*IH*IK, rik.p+(size_t) t*IK, dlw.p, dlb.p, rkeys.p,
                         pos, IH, IK, R, 1e-5f, rope, nullptr);
        f::score(riq.p+(size_t) t*IH*IK, rkeys.p, diw.p+(size_t) t*IH, IK, IH, pos+1, rs.p, nullptr);
        f::select(rs.p, pos+1, topk, rc.p+(size_t) t*topk, nullptr);
        f::mla(rq.p+(size_t) t*H*(N+R), dkb.p, dvb.p, rlat.p, rc.p+(size_t) t*topk,
                ns, H, N, L, R, V, 0.0625f, rout.p+(size_t) t*H*V, nullptr);
    }
    const auto selected = bc.get(), expected = rc.get();
    if (selected != expected) { std::fprintf(stderr, "batched selection differs from causal tokenwise\n"); std::exit(1); }
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < topk; ++i)
            if (selected[(size_t) t*topk+i] > p0+t) { std::fprintf(stderr, "future key selected\n"); std::exit(1); }
    same("causal batch MLA vs tokenwise", bout.get(), rout.get());
    same("batch indexer transforms vs tokenwise", biq.get(), riq.get());
    // A following shared-indexer layer owns different queries and KV values,
    // and reuses the earlier layer's selection without scoring anything again.
    auto shared_q = data(q.size(), 0.2f, 31), shared_kv = data(kv.size(), 0.4f, 41);
    bq.put(shared_q); rq.put(shared_q); bkv.put(shared_kv);
    blat.put(prefix_lat); rlat.put(prefix_lat);
    f::prepare_rows(bq.p, bkv.p, dw.p, blat.p, p0, T, H, N, L, R, 1e-5f, rope, nullptr);
    f::mla_rows(bq.p, dkb.p, dvb.p, blat.p, bc.p, p0, T, topk, H, N, L, R, V, 0.0625f, bout.p, nullptr);
    for (int t = 0; t < T; ++t) {
        f::prepare(rq.p+(size_t) t*H*(N+R), bkv.p+(size_t) t*(L+R), dw.p, rlat.p, p0+t,
                    H, N, L, R, 1e-5f, rope, nullptr);
        f::mla(rq.p+(size_t) t*H*(N+R), dkb.p, dvb.p, rlat.p, rc.p+(size_t) t*topk,
                std::min(p0+t+1, topk), H, N, L, R, V, 0.0625f, rout.p+(size_t) t*H*V, nullptr);
    }
    same("shared-indexer batch MLA vs tokenwise", bout.get(), rout.get());
    if (bc.get() != selected) { std::fprintf(stderr, "shared layer changed selections\n"); std::exit(1); }
    std::printf("causal/shared proof: p0=%d T=%d topk=%d, future rows poisoned\n", p0, T, topk);
}

template<class Call> double time_ms(Call run) {
    constexpr int repeats = 5;
    for (int i = 0; i < 2; ++i) run();
    cudaEvent_t start, stop;
    ck(cudaEventCreate(&start)); ck(cudaEventCreate(&stop));
    ck(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i) run();
    ck(cudaEventRecord(stop)); ck(cudaEventSynchronize(stop));
    float ms = 0;
    ck(cudaEventElapsedTime(&ms, start, stop));
    ck(cudaEventDestroy(start)); ck(cudaEventDestroy(stop));
    return ms/repeats;
}
void bench_rows(const char* capture_path = nullptr) {
    constexpr int H = 64, N = 192, L = 512, R = 64, V = 256, K = 2048, P0 = 2047;
    auto kb = bf16(data((size_t) H*L*N, 0.03f, 2)), vb = bf16(data((size_t) H*V*L, 0.04f, 5));
    Device<uint16_t> dkb(kb), dvb(vb);
    FILE* capture = capture_path ? std::fopen(capture_path, "wb") : nullptr;
    if (capture_path && !capture) { std::perror("MLA row capture"); std::exit(1); }
    std::puts("full MLA row-launch timing: 2 warmups, 5 repetitions, CUDA events, no transfers");
    std::puts("rows selected serial_ms batch_ms speedup batch_ms_per_row");
    for (int T : {32, 128}) {
        auto q = data((size_t) T*H*(N+R), 0.1f, 7), values = data((size_t) (P0+T+1)*(L+R), 0.4f, 11);
        std::vector<uint16_t> cache(values.size());
        for (size_t i = 0; i < cache.size(); ++i) cache[i] = f16_from_f32(values[i]);
        std::vector<int> cells((size_t) T*K);
        for (int t = 0; t < T; ++t)
            for (int s = 0; s < K; ++s) cells[(size_t) t*K+s] = t+(s*13)%K;
        Device<float> dq(q), serial_out((size_t) T*H*V), batch_out((size_t) T*H*V);
        Device<uint16_t> dc(cache);
        Device<int> ds(cells);
        auto serial = [&] {
            for (int t = 0; t < T; ++t)
                f::mla(dq.p+(size_t) t*H*(N+R), dkb.p, dvb.p, dc.p, ds.p+(size_t) t*K,
                    K, H, N, L, R, V, 0.0625f, serial_out.p+(size_t) t*H*V, nullptr);
        };
        auto batch = [&] {
            f::mla_rows(dq.p, dkb.p, dvb.p, dc.p, ds.p, P0, T, K, H, N, L, R, V,
                         0.0625f, batch_out.p, nullptr);
        };
        serial(); batch();
        const auto serial_values = serial_out.get(), batch_values = batch_out.get();
        same("row-batched MLA vs per-row launches", batch_values, serial_values);
        if (capture && std::fwrite(batch_values.data(), sizeof(float), batch_values.size(), capture) != batch_values.size()) {
            std::perror("MLA row capture write"); std::exit(1);
        }
        const double old_ms = time_ms(serial), new_ms = time_ms(batch);
        std::printf("%d %d %.4f %.4f %.3f %.5f\n", T, K, old_ms, new_ms, old_ms/new_ms, new_ms/T);
    }
    if (capture && std::fclose(capture)) { std::perror("MLA row capture close"); std::exit(1); }
}
}  // namespace
int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--bench") == 0) { bench_rows(); return 0; }
    if (argc == 3 && std::strcmp(argv[1], "--bench-save") == 0) { bench_rows(argv[2]); return 0; }
    norm_rows();
    attention(0, 4, 3);
    attention(3, 4, 4);
    attention(2050, 2, 2048);
    ck(cudaDeviceSynchronize());
    std::puts("glm_full_batch_test: OK");
    return 0;
}
