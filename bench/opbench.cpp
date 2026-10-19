// opbench: steady-state per-op cost on Vulkan0 for the ops in one Flash-Next decode layer (n_tokens=1).
// Each case builds a graph of N independent copies of one op and reports wall us/op (includes CPU record).
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-vulkan.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

static ggml_backend_t be;
struct Case { std::string name; std::function<ggml_tensor*(ggml_context*, int)> mk; int n; };

static double run(const Case & c) {
    const int N = c.n;
    ggml_init_params ip = { 256u*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * g = ggml_new_graph_custom(ctx, 4096, false);
    std::vector<ggml_tensor*> outs;
    for (int i = 0; i < N; i++) { ggml_tensor * o = c.mk(ctx, i); if (!o) { ggml_free(ctx); return -1; } ggml_build_forward_expand(g, o); }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) { ggml_free(ctx); return -2; }
    // fill inputs with small deterministic data (weights quantized types: zeros are fine for timing)
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        if (t->op == GGML_OP_NONE && t->buffer) {
            std::vector<uint8_t> z(ggml_nbytes(t), 0);
            if (t->type == GGML_TYPE_F32) { float * f = (float*)z.data(); for (size_t k = 0; k < z.size()/4; k++) f[k] = 0.01f * ((k % 17) - 8); }
            if (t->type == GGML_TYPE_I32) { int32_t * f = (int32_t*)z.data(); for (size_t k = 0; k < z.size()/4; k++) f[k] = (int32_t)((k*37) % 500); }
            ggml_backend_tensor_set(t, z.data(), 0, z.size());
        }
    }
    for (int w = 0; w < 5; w++) ggml_backend_graph_compute(be, g);
    const int R = 30;
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < R; r++) ggml_backend_graph_compute(be, g);
    double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    ggml_backend_buffer_free(buf); ggml_free(ctx);
    return us / R / N;
}

static ggml_tensor * T(ggml_context * c, ggml_type t, int64_t a, int64_t b = 1, int64_t d = 1, int64_t e = 1) { return ggml_new_tensor_4d(c, t, a, b, d, e); }

int main(int argc, char ** argv) {
    be = ggml_backend_vk_init(0);
    if (!be) { fprintf(stderr, "no vk\n"); return 1; }
    const char * filt = argc > 1 ? argv[1] : "";
    std::vector<Case> cs;
    auto mm = [&](const char * nm, ggml_type t, int64_t k, int64_t m, int n_unused = 0) {
        (void)n_unused;
        size_t bytes = ggml_row_size(t, k) * m;
        int npool = (int)std::min<size_t>(64, std::max<size_t>(4, (size_t)(96u << 20) / bytes + 1));
        std::string s = std::string("MUL_MAT ") + nm + " " + ggml_type_name(t) + " k=" + std::to_string(k) + " m=" + std::to_string(m) + " (" + std::to_string(bytes/1024) + " KiB x" + std::to_string(npool) + ")";
        cs.push_back({s, [=](ggml_context * c, int i) {
            static thread_local std::vector<ggml_tensor*> pool; static thread_local ggml_context * pc = nullptr;
            if (i == 0 || pc != c) { pool.clear(); pc = c; }
            while ((int)pool.size() < npool) pool.push_back(T(c, t, k, m));
            static thread_local ggml_tensor * x = nullptr; if (i == 0) x = T(c, GGML_TYPE_F32, k);
            return ggml_mul_mat(c, pool[i % pool.size()], x); }, npool});
    };
    mm("hc_down", GGML_TYPE_BF16, 10240, 320);
    mm("hc_up", GGML_TYPE_BF16, 320, 10240);
    mm("hc_inject", GGML_TYPE_BF16, 10240, 4);
    mm("alpha/beta", GGML_TYPE_BF16, 2560, 48);
    mm("router", GGML_TYPE_BF16, 2560, 512);
    mm("shexp_gate", GGML_TYPE_BF16, 2560, 1);
    mm("qkv", GGML_TYPE_IQ4_XS, 2560, 10240);
    mm("z", GGML_TYPE_IQ3_S, 2560, 6144);
    mm("attn_out", GGML_TYPE_Q4_K, 6144, 2560);
    mm("shexp_gate_proj", GGML_TYPE_IQ4_XS, 2560, 640);
    mm("shexp_up_proj", GGML_TYPE_IQ3_S, 2560, 640);
    mm("shexp_down", GGML_TYPE_IQ4_NL, 640, 2560);
    // MUL_MAT_ID experts: 10 of 512
    auto mid = [&](const char * nm, ggml_type t, int64_t k, int64_t m, bool down, int nu = 10) {
        cs.push_back({std::string("MUL_MAT_ID ") + nm + " " + ggml_type_name(t) + (nu != 10 ? " nu=" + std::to_string(nu) : ""), [=](ggml_context * c, int i) {
            static thread_local ggml_tensor * ids = nullptr, * x = nullptr, * xd = nullptr; static thread_local std::vector<ggml_tensor*> pool; static thread_local ggml_context * pc = nullptr;
            if (i == 0 || pc != c) { pool.clear(); pc = c; ids = T(c, GGML_TYPE_I32, nu, 1); x = T(c, GGML_TYPE_F32, k, 1, 1); xd = T(c, GGML_TYPE_F32, k, nu, 1); }
            while ((int)pool.size() < 12) pool.push_back(T(c, t, k, m, 512));
            return ggml_mul_mat_id(c, pool[i % pool.size()], down ? xd : x, ids); }, 12});
    };
    mid("gate", GGML_TYPE_IQ2_XXS, 2560, 640, false);
    mid("down", GGML_TYPE_Q2_0, 640, 2560, true);
    for (int nu : {1, 2, 5, 20}) mid("gate", GGML_TYPE_IQ2_XXS, 2560, 640, false, nu);
    mid("gate", GGML_TYPE_IQ2_XS, 2560, 640, false);
    mid("gate", GGML_TYPE_IQ2_S, 2560, 640, false);
    mid("gate", GGML_TYPE_IQ3_XXS, 2560, 640, false);
    mid("gate", GGML_TYPE_IQ3_S, 2560, 640, false);
    mid("down", GGML_TYPE_IQ4_NL, 640, 2560, true);
    // elementwise / small
    auto ew = [&](const char * nm, std::function<ggml_tensor*(ggml_context*, ggml_tensor*, ggml_tensor*)> f, int64_t a, int64_t b = 1) {
        cs.push_back({std::string("EW ") + nm, [=](ggml_context * c, int i) {
            static thread_local ggml_tensor * x = nullptr, * y = nullptr; if (i == 0) { x = T(c, GGML_TYPE_F32, a, b); y = T(c, GGML_TYPE_F32, a, b); }
            return f(c, x, y); }, 60});
    };
    ew("add 2560x1", [](ggml_context * c, ggml_tensor * x, ggml_tensor * y) { return ggml_add(c, x, y); }, 2560);
    ew("mul 2560x4", [](ggml_context * c, ggml_tensor * x, ggml_tensor * y) { return ggml_mul(c, x, y); }, 2560, 4);
    ew("scale 320", [](ggml_context * c, ggml_tensor * x, ggml_tensor *) { return ggml_scale(c, x, 0.25f); }, 320);
    ew("silu 320", [](ggml_context * c, ggml_tensor * x, ggml_tensor *) { return ggml_silu(c, x); }, 320);
    ew("sigmoid 4", [](ggml_context * c, ggml_tensor * x, ggml_tensor *) { return ggml_sigmoid(c, x); }, 4);
    ew("softplus 48", [](ggml_context * c, ggml_tensor * x, ggml_tensor *) { return ggml_softplus(c, x); }, 48);
    ew("rms_norm 2560x4", [](ggml_context * c, ggml_tensor * x, ggml_tensor *) { return ggml_rms_norm(c, x, 1e-6f); }, 2560, 4);
    ew("rms_norm 128x16", [](ggml_context * c, ggml_tensor * x, ggml_tensor *) { return ggml_rms_norm(c, x, 1e-6f); }, 128, 16);
    ew("swiglu 640x10", [](ggml_context * c, ggml_tensor * x, ggml_tensor * y) { return ggml_swiglu_split(c, x, y); }, 640, 10);
    ew("softmax 512", [](ggml_context * c, ggml_tensor * x, ggml_tensor *) { return ggml_soft_max(c, x); }, 512);
    ew("argsort 512", [](ggml_context * c, ggml_tensor * x, ggml_tensor *) { return ggml_argsort(c, x, GGML_SORT_ORDER_DESC); }, 512);
    ew("sum_rows 10", [](ggml_context * c, ggml_tensor * x, ggml_tensor *) { return ggml_sum_rows(c, x); }, 10);
    ew("cont 3x10240", [](ggml_context * c, ggml_tensor * x, ggml_tensor *) { return ggml_cont(c, ggml_view_2d(c, x, 3, 10240, x->nb[1], 0)); }, 5, 10240);
    ew("ssm_conv", [](ggml_context * c, ggml_tensor * x, ggml_tensor * y) { return ggml_ssm_conv(c, x, y); }, 5, 10240);
    cs.back().mk = [](ggml_context * c, int i) { static thread_local ggml_tensor * x = nullptr, * w = nullptr; if (i == 0) { x = T(c, GGML_TYPE_F32, 5, 10240); w = T(c, GGML_TYPE_F32, 4, 10240); } return ggml_ssm_conv(c, x, w); };
    cs.push_back({"HC_MIX fused 2560x4 (pool 8)", [](ggml_context * c, int i) {
        static thread_local std::vector<ggml_tensor*> wd, wu, wi; static thread_local ggml_context * pc = nullptr;
        static thread_local ggml_tensor * x = nullptr, * g = nullptr;
        if (i == 0 || pc != c) { wd.clear(); wu.clear(); wi.clear(); pc = c; x = T(c, GGML_TYPE_F32, 2560, 4, 1); g = T(c, GGML_TYPE_F32, 2560, 4, 1);
            for (int k = 0; k < 8; k++) { wd.push_back(T(c, GGML_TYPE_BF16, 10240, 320)); wu.push_back(T(c, GGML_TYPE_BF16, 320, 10240)); wi.push_back(T(c, GGML_TYPE_BF16, 10240, 4)); } }
        return ggml_hc_mix(c, x, g, wd[i % 8], wu[i % 8], wi[i % 8], 1e-6f); }, 40});
    cs.push_back({"HC_PRE_gated 2560x4", [](ggml_context * c, int i) { static thread_local ggml_tensor * x = nullptr, * g = nullptr; if (i == 0) { x = T(c, GGML_TYPE_F32, 2560, 4, 1); g = T(c, GGML_TYPE_F32, 2560, 4, 1); } return ggml_dsv4_hc_pre_gated(c, x, g, 0.25f); }, 40});
    cs.push_back({"HC_POST 2560x4", [](ggml_context * c, int i) { static thread_local ggml_tensor * x = nullptr, * r = nullptr, * w = nullptr; if (i == 0) { x = T(c, GGML_TYPE_F32, 2560, 1); r = T(c, GGML_TYPE_F32, 2560, 4, 1); w = T(c, GGML_TYPE_F32, 4, 1); } return ggml_dsv4_hc_post(c, x, r, w, nullptr); }, 40});
    cs.push_back({"GATED_DELTA_NET", [](ggml_context * c, int i) {
        static thread_local ggml_tensor *q, *k, *v, *g, *b, *s; if (i == 0) { q = T(c, GGML_TYPE_F32, 128, 16, 1, 1); k = T(c, GGML_TYPE_F32, 128, 16, 1, 1); v = T(c, GGML_TYPE_F32, 128, 48, 1, 1); g = T(c, GGML_TYPE_F32, 1, 48, 1, 1); b = T(c, GGML_TYPE_F32, 1, 48, 1, 1); s = T(c, GGML_TYPE_F32, 128, 128, 48, 1); }
        return ggml_gated_delta_net(c, q, k, v, g, b, s, 1); }, 20});
    cs.push_back({"CPY state 786432", [](ggml_context * c, int i) { static thread_local ggml_tensor * x = nullptr, * y = nullptr; if (i == 0) { x = T(c, GGML_TYPE_F32, 786432); y = T(c, GGML_TYPE_F32, 786432); } return ggml_cpy(c, x, y); }, 30});
    for (auto & c : cs) {
        if (*filt && c.name.find(filt) == std::string::npos) continue;
        double us = run(c);
        printf("%-48s %8.2f us/op\n", c.name.c_str(), us); fflush(stdout);
    }
    return 0;
}
