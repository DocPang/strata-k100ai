# 优化内容与格式覆盖

c4 累计改动覆盖平台适配、量化计算、MoE 分派、prefill、decode 和工作区管理。源码补丁位于 `engines/strata/c4/replay/patches/`。

## 实现范围

| 范围 | 实现 |
| --- | --- |
| gfx928 平台 | HIP 兼容层、wave64、dot4、DTK 编译和链接适配 |
| ggml / llama.cpp | HIP 通用头、供应商接口及 gfx928 MMQ 配置 |
| MMVQ | Q6/Q8 packed、Q6 wave packing 与准备路径 |
| MoE 分派 | 专家布局、resident sort、fused/MMQ 路径调整 |
| Prefill gate/up | exact 展开、KGU、部分 IQ2_S 专用计算 |
| Prefill down | down GEMM、Q2_0 staging、IQ4_NL exact preexpand |
| Dense GEMM | rocBLAS 步长 padding、hipBLASLt 支持 |
| 工作区 | lazy workspace、条件 scratch、embedding 缓冲复用 |
| 融合 | SiLU×Mul 与量化融合，减少中间读写 |
| Decode | GDN head、无用写回消除、HC split、down prefetch |
| PLE / Attention | prelaunch、lane/cell 与 gfx928 计算路径 |
| MTP | HIP 分支衔接、独立 runtime、草稿参数配置 |

Strata 补丁包含 21 个修改文件和 4 个新增实现头文件；llama.cpp/ggml 补丁包含 2 个修改文件和 1 个新增配置。源文件回放及构建状态见 [SOURCE_BUILD.md](SOURCE_BUILD.md)。

## 部署调优

INT8 KV、KV streaming、prompt/会话缓存及 MTP 基于上游能力。本项目提供 K100AI 环境下的参数组合与部署记录。prefill 分块、显存预留和专家驻留策略分别管理，容量关系见 [MEMORY.md](MEMORY.md)。

## IQ3_XXS 覆盖

同一发布系列中，IQ3_S 与 IQ3_XXS 的混合量化布局不同：

| 部位 | IQ3_S | IQ3_XXS | c4 适用范围 |
| --- | --- | --- | --- |
| gate/up | IQ2_S、IQ3_XXS、IQ3_S、IQ4_XS | 增加 IQ2_XS、IQ2_XXS | 新类型使用通用路径；19/48 层不进入现有 exact gate/up 分支 |
| down | IQ4_NL 39 层、Q2_0 9 层 | IQ4_NL 18 层、Q2_0 30 层 | 已有两类实现，实际分派仍受形状约束 |
| 目标输出头 | Q6_K | Q5_K | Q6 packed 优化不直接覆盖 Q5_K 目标头 |

层数为静态类型统计，不代表时间占比。本版本新增 XXS 的配置及模型校验，不包含独立实验引擎的专用优化。

## 数值与性能记录

c4 保留已接受的 FP16 中间结果策略。exact 子路径的性质不代表整套计算位级一致。量化质量差异、实现误差与采样差异应分别评估。

性能比较需固定设备、权重、输入、上下文、缓存和 MTP 接受率条件。组件或 profiler 数据单独记录，整模结果使用无仪器的固定负载测试。本版本不提供跨负载累计加速比。
