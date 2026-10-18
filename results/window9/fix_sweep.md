| model                          |       size |     params | backend    | ngl | n_ubatch |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --: | --------------: | -------------------: |
| qwen35 27B IQ4_XS - 4.25 bpw   |  13.26 GiB |    27.32 B | Vulkan     |  99 |       64 |   0 |            tg32 |         28.40 ± 0.00 |
| qwen35 27B IQ4_XS - 4.25 bpw   |  13.26 GiB |    27.32 B | Vulkan     |  99 |       64 |   0 |       tg32 @ d8 |         28.00 ± 0.00 |
| qwen35 27B IQ4_XS - 4.25 bpw   |  13.26 GiB |    27.32 B | Vulkan     |  99 |       64 |   0 |      tg32 @ d16 |         28.28 ± 0.00 |
| qwen35 27B IQ4_XS - 4.25 bpw   |  13.26 GiB |    27.32 B | Vulkan     |  99 |       64 |   0 |      tg32 @ d24 |         28.21 ± 0.00 |
| qwen35 27B IQ4_XS - 4.25 bpw   |  13.26 GiB |    27.32 B | Vulkan     |  99 |       64 |   0 |      tg32 @ d40 |         28.10 ± 0.00 |
| qwen35 27B IQ4_XS - 4.25 bpw   |  13.26 GiB |    27.32 B | Vulkan     |  99 |       64 |   0 |     tg32 @ d128 |         27.83 ± 0.00 |
| qwen35 27B IQ4_XS - 4.25 bpw   |  13.26 GiB |    27.32 B | Vulkan     |  99 |       64 |   0 |     tg32 @ d256 |         27.95 ± 0.00 |
| qwen35 27B IQ4_XS - 4.25 bpw   |  13.26 GiB |    27.32 B | Vulkan     |  99 |       64 |   0 |     tg32 @ d512 |         27.68 ± 0.00 |

build: 547208d (1)
