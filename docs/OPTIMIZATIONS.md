# 优化工作清单

本清单描述累计源码差异和选定配置，不把各轮独立实验百分比相加，也不把组件结果当作端到端性能。源码补丁见 `engines/strata/c4/replay/patches/`。

| 范围 | 内容 | 归属与边界 |
| --- | --- | --- |
| gfx928 平台 | HIP 兼容层、wave64、dot4、DTK 编译与链接 | K100AI 适配；不是 NVIDIA 二进制直接迁移 |
| ggml / llama.cpp | common.cuh、vendors/hip.h 和 gfx928 MMQ 配置 | 独立依赖补丁，不能只复制 Strata src |
| MMVQ | Q6/Q8 packed、Q6 wave packing 及准备路径 | 按实际量化类型和形状命中 |
| MoE 调度 | 专家布局、resident sort、fused/MMQ 路径 | 驻留与输入形状影响收益 |
| Prefill gate/up | exact 展开、KGU、部分 IQ2_S 专用计算路径 | 不等于覆盖所有 IQ2_XS/IQ2_XXS 张量 |
| Prefill down | down GEMM、Q2_0 staging、IQ4_NL exact preexpand | XXS 下仍须按类型检查命中，不能误称 Q2_0 完全没有支持 |
| Dense GEMM | rocBLAS 步长 padding、hipBLASLt 配方 | 形状相关，不承诺所有长度都更快 |
| Prefill 临时空间 | lazy workspace、条件 scratch、embedding 缓冲复用 | 减少临时空间及冗余分配 |
| Prefill 融合 | SiLU×Mul 与量化融合，减少中间读写 | 单一路径 exact 不代表整套 FP16 策略无损 |
| Decode | GDN head、消除无用写回、HC split、down prefetch | 累计代码修改与已有路径调优 |
| PLE / Attention | prelaunch、lane/cell、gfx928 计算路径 | 与兼容分支和负载共同验证 |
| MTP | HIP 分支衔接、独立 MTP pack/runtime、草稿参数调优 | 基于上游投机解码；收益依赖接受率 |
| 长上下文 | int8 KV、32K 驻留窗口、KV prefetch | 上游能力的 K100AI 配置与实测，不全部归为新增内核 |
| 缓存与部署 | prompt/会话缓存、prefill 分块、显存预留、服务配置 | 工程调优；缓存槽数不是推理并发数 |
| 复现与交付 | 固定引擎哈希、依赖提交、源码补丁、文件清单、小安装包 | 安装已验证不等于干净源码重建已通过 |

Strata 累计补丁为 21 个修改文件和 4 个新增实现头文件；依赖补丁为 2 个修改文件与 1 个新增配置。历史备份文件不进入发布补丁。

数值政策沿用历史已接受的 FP16 中间结果策略；FP16 Dm 等路径会改变数值结果，因此不宣传为“近似无损”或“全部位级一致”。不能以确定性测试代替相对于原版的质量测试。

历史 c4 曾有 32K context / prefill30602 的 47 请求、4 次进程启动验证；当前默认是 prefill20480 / max_context262144，不能把前者当后者的完整验收。首版不发布无上下文和负载限定的 tok/s 倍数。

XXS 专用实验及长上下文候选与本 c4 冻结产物分开，未混入本 Release。后续版本需独立记录新增内容和模型覆盖。
