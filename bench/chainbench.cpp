// chainbench: cost of a *dependent* chain of N identical small ops (each reads the previous output).
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-vulkan.h"
#include <chrono>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>
static ggml_backend_t be;
static double run(const std::string & name, int N, std::function<ggml_tensor*(ggml_context*, ggml_tensor*, int)> step, std::function<ggml_tensor*(ggml_context*)> mkin) {
    ggml_init_params ip = { 256u*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * g = ggml_new_graph_custom(ctx, 4096, false);
    ggml_tensor * x = mkin(ctx); ggml_set_input(x);
    ggml_tensor * cur = x;
    for (int i = 0; i < N; i++) cur = step(ctx, cur, i);
    ggml_set_output(cur);
    ggml_build_forward_expand(g, cur);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) if (t->op == GGML_OP_NONE && t->buffer) {
        std::vector<uint8_t> z(ggml_nbytes(t), 0);
        if (t->type == GGML_TYPE_F32) { float * f = (float*)z.data(); for (size_t k = 0; k < z.size()/4; k++) f[k] = 0.01f * ((k % 17) - 8); }
        ggml_backend_tensor_set(t, z.data(), 0, z.size()); }
    for (int w = 0; w < 5; w++) ggml_backend_graph_compute(be, g);
    const int R = 50;
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < R; r++) ggml_backend_graph_compute(be, g);
    double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    printf("%-44s %8.2f us/op (dependent chain of %d)\n", name.c_str(), us / R / N, N); fflush(stdout);
    ggml_backend_buffer_free(buf); ggml_free(ctx);
    return us / R / N;
}
int main() {
    be = ggml_backend_vk_init(0);
    auto F = [](int64_t a, int64_t b = 1) { return [=](ggml_context * c) { return ggml_new_tensor_2d(c, GGML_TYPE_F32, a, b); }; };
    run("scale 4", 64, [](ggml_context * c, ggml_tensor * x, int) { return ggml_scale(c, x, 0.999f); }, F(4));
    run("scale 2560", 64, [](ggml_context * c, ggml_tensor * x, int) { return ggml_scale(c, x, 0.999f); }, F(2560));
    run("scale 10240", 64, [](ggml_context * c, ggml_tensor * x, int) { return ggml_scale(c, x, 0.999f); }, F(10240));
    run("rms_norm 2560x4", 64, [](ggml_context * c, ggml_tensor * x, int) { return ggml_rms_norm(c, x, 1e-6f); }, F(2560, 4));
    run("silu 2560", 64, [](ggml_context * c, ggml_tensor * x, int) { return ggml_silu(c, x); }, F(2560));
    static std::vector<ggml_tensor*> W;
    auto mmchain = [&](const char * nm, ggml_type t, int64_t k) {
        run(std::string("mul_mat ") + nm + " " + std::to_string(k) + "x" + std::to_string(k), 32,
            [&, t, k](ggml_context * c, ggml_tensor * x, int i) { static ggml_context * pc = nullptr; static std::vector<ggml_tensor*> pool; if (pc != c) { pc = c; pool.clear(); } if ((int)pool.size() < 4) pool.push_back(ggml_new_tensor_2d(c, t, k, k)); return ggml_mul_mat(c, pool[i % pool.size()], x); }, F(k));
    };
    mmchain("bf16", GGML_TYPE_BF16, 320);
    mmchain("bf16", GGML_TYPE_BF16, 2560);
    mmchain("iq4_xs", GGML_TYPE_IQ4_XS, 2560);
    mmchain("q4_K", GGML_TYPE_Q4_K, 2560);
    mmchain("iq3_s", GGML_TYPE_IQ3_S, 640);
    return 0;
}
