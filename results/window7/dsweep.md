| model                          |       size |     params | backend    | ngl | n_ubatch |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --: | --------------: | -------------------: |
| qwen35 27B IQ4_XS - 4.25 bpw   |  13.26 GiB |    27.32 B | Vulkan     |  99 |       64 |   0 |            tg32 |         28.14 ± 0.00 |
| qwen35 27B IQ4_XS - 4.25 bpw   |  13.26 GiB |    27.32 B | Vulkan     |  99 |       64 |   0 |       tg32 @ d8 |         16.81 ± 0.00 |
| qwen35 27B IQ4_XS - 4.25 bpw   |  13.26 GiB |    27.32 B | Vulkan     |  99 |       64 |   0 |      tg32 @ d16 |         28.41 ± 0.00 |
| qwen35 27B IQ4_XS - 4.25 bpw   |  13.26 GiB |    27.32 B | Vulkan     |  99 |       64 |   0 |      tg32 @ d24 |         16.78 ± 0.00 |
| qwen35 27B IQ4_XS - 4.25 bpw   |  13.26 GiB |    27.32 B | Vulkan     |  99 |       64 |   0 |      tg32 @ d32 |         27.79 ± 0.00 |
| qwen35 27B IQ4_XS - 4.25 bpw   |  13.26 GiB |    27.32 B | Vulkan     |  99 |       64 |   0 |      tg32 @ d40 |         16.38 ± 0.00 |
| qwen35 27B IQ4_XS - 4.25 bpw   |  13.26 GiB |    27.32 B | Vulkan     |  99 |       64 |   0 |      tg32 @ d48 |         27.90 ± 0.00 |
| qwen35 27B IQ4_XS - 4.25 bpw   |  13.26 GiB |    27.32 B | Vulkan     |  99 |       64 |   0 |      tg32 @ d56 |         16.24 ± 0.00 |
| qwen35 27B IQ4_XS - 4.25 bpw   |  13.26 GiB |    27.32 B | Vulkan     |  99 |       64 |   0 |      tg32 @ d64 |         28.04 ± 0.00 |
| qwen35 27B IQ4_XS - 4.25 bpw   |  13.26 GiB |    27.32 B | Vulkan     |  99 |       64 |   0 |      tg32 @ d80 |         16.48 ± 0.00 |
| qwen35 27B IQ4_XS - 4.25 bpw   |  13.26 GiB |    27.32 B | Vulkan     |  99 |       64 |   0 |      tg32 @ d96 |         27.64 ± 0.00 |
| qwen35 27B IQ4_XS - 4.25 bpw   |  13.26 GiB |    27.32 B | Vulkan     |  99 |       64 |   0 |     tg32 @ d112 |         16.35 ± 0.00 |
| qwen35 27B IQ4_XS - 4.25 bpw   |  13.26 GiB |    27.32 B | Vulkan     |  99 |       64 |   0 |     tg32 @ d128 |         28.21 ± 0.00 |
| qwen35 27B IQ4_XS - 4.25 bpw   |  13.26 GiB |    27.32 B | Vulkan     |  99 |       64 |   0 |     tg32 @ d192 |         16.34 ± 0.00 |
| qwen35 27B IQ4_XS - 4.25 bpw   |  13.26 GiB |    27.32 B | Vulkan     |  99 |       64 |   0 |     tg32 @ d256 |         16.51 ± 0.00 |
| qwen35 27B IQ4_XS - 4.25 bpw   |  13.26 GiB |    27.32 B | Vulkan     |  99 |       64 |   0 |     tg32 @ d512 |         16.55 ± 0.00 |

build: 547208d (1)
