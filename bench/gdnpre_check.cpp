// gdnpre_check: GGML_OP_GDN_PRE against the unfused op chain it replaces, same data, per backend.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-vulkan.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static const int64_t C = 10240, H = 48, HK = 16, D = 128;

struct Out { std::vector<float> x, g, b, st; };

static Out run(ggml_backend_t be, bool fused, const std::vector<std::vector<float>> & in) {
    ggml_init_params ip = { 64u*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 256, false);
    ggml_tensor * qkv = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C);
    ggml_tensor * cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3*C, 3);
    ggml_tensor * conv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, C);
    ggml_tensor * alpha = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
    ggml_tensor * beta = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
    ggml_tensor * dt = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
    ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
    ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_tensor * ins[] = { qkv, cache, conv, alpha, beta, dt, a };
    ggml_tensor * x, * g, * b, * st;
    const float eps = 1e-6f;
    auto l2 = [&](ggml_tensor * t) { return ggml_scale(ctx, ggml_rms_norm(ctx, t, eps/D), 1.0f/sqrtf((float) D)); };
    if (fused) {
        ggml_tensor * pre = ggml_gdn_pre(ctx, qkv, cache, conv, alpha, beta, dt, a, idx, 0, 2*D*HK, eps);
        x = ggml_view_1d(ctx, pre, C, 0);
        g = ggml_view_1d(ctx, pre, H, C*4);
        b = ggml_view_1d(ctx, pre, H, (C+64)*4);
        st = nullptr;
    } else {
        ggml_tensor * state = ggml_reshape_3d(ctx, ggml_get_rows(ctx, cache, idx), 3, C, 1);
        ggml_tensor * ci = ggml_concat(ctx, state, ggml_transpose(ctx, ggml_reshape_2d(ctx, qkv, C, 1)), 0);
        ggml_tensor * cv = ggml_silu(ctx, ggml_ssm_conv(ctx, ci, conv));
        ggml_tensor * q = l2(ggml_view_4d(ctx, cv, D, HK, 1, 1, D*4, C*4, C*4, 0));
        ggml_tensor * k = l2(ggml_view_4d(ctx, cv, D, HK, 1, 1, D*4, C*4, C*4, D*HK*4));
        ggml_tensor * v = ggml_view_4d(ctx, cv, D, H, 1, 1, D*4, C*4, C*4, 2*D*HK*4);
        x = ggml_concat(ctx, ggml_concat(ctx, ggml_reshape_1d(ctx, ggml_cont(ctx, q), D*HK), ggml_reshape_1d(ctx, ggml_cont(ctx, k), D*HK), 0),
                        ggml_reshape_1d(ctx, ggml_cont(ctx, v), D*H), 0);
        g = ggml_mul(ctx, ggml_softplus(ctx, ggml_add(ctx, alpha, dt)), a);
        b = ggml_sigmoid(ctx, beta);
        // new state = last 3 columns of ci per channel
        st = ggml_cont(ctx, ggml_view_3d(ctx, ci, 3, C, 1, ci->nb[1], ci->nb[2], 4));
        st = ggml_reshape_1d(ctx, st, 3*C);
    }
    ggml_build_forward_expand(gf, x); ggml_build_forward_expand(gf, g); ggml_build_forward_expand(gf, b); if (st) ggml_build_forward_expand(gf, st);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    for (int i = 0; i < 7; i++) ggml_backend_tensor_set(ins[i], in[i].data(), 0, in[i].size()*4);
    int32_t ix = 2; ggml_backend_tensor_set(idx, &ix, 0, 4);
    ggml_backend_graph_compute(be, gf);
    Out o;
    auto rd = [&](ggml_tensor * t, std::vector<float> & v) { v.resize(ggml_nelements(t)); ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t)); };
    rd(x, o.x); rd(g, o.g); rd(b, o.b);
    if (fused) { // dst cell 0 of the cache
        o.st.resize(3*C); ggml_backend_tensor_get(cache, o.st.data(), 0, 3*C*4);
    } else {
        rd(st, o.st);
    }
    ggml_backend_buffer_free(buf); ggml_free(ctx);
    return o;
}

static void cmp(const char * nm, const std::vector<float> & a, const std::vector<float> & b) {
    double mx = 0, ref = 0;
    for (size_t i = 0; i < a.size(); i++) { mx = std::max(mx, (double) fabs(a[i] - b[i])); ref = std::max(ref, (double) fabs(b[i])); }
    printf("  %-6s n=%zu max|d|=%.3e  max|ref|=%.3e  rel=%.3e\n", nm, a.size(), mx, ref, mx/ref);
}

int main() {
    std::mt19937 rng(7); std::normal_distribution<float> nd(0, 1);
    auto fill = [&](size_t n, float s, float off = 0) { std::vector<float> v(n); for (auto & f : v) f = off + s*nd(rng); return v; };
    std::vector<std::vector<float>> in = { fill(C, 1.0f), fill(3*C*3, 1.0f), fill(4*C, 0.4f), fill(H, 1.0f), fill(H, 1.0f), fill(H, 0.3f), fill(H, 0.2f, -1.5f) };
    ggml_backend_t vk = ggml_backend_vk_init(0), cpu = ggml_backend_cpu_init();
    Out cu = run(cpu, false, in), cf = run(cpu, true, in), vu = run(vk, false, in), vf = run(vk, true, in);
    printf("cpu fused vs cpu unfused\n"); cmp("x", cf.x, cu.x); cmp("gate", cf.g, cu.g); cmp("beta", cf.b, cu.b); cmp("state", cf.st, cu.st);
    printf("vk fused vs cpu unfused\n");  cmp("x", vf.x, cu.x); cmp("gate", vf.g, cu.g); cmp("beta", vf.b, cu.b); cmp("state", vf.st, cu.st);
    printf("vk unfused vs cpu unfused\n"); cmp("x", vu.x, cu.x); cmp("gate", vu.g, cu.g); cmp("beta", vu.b, cu.b); cmp("state", vu.st, cu.st);
    return 0;
}
