# 干净重建报告（2026-10-20）

用**已公开的材料**从零重建 `43002a72…`，验证重建路径可用、且重建产物与发布二进制行为等价。

## 结论

| 项 | 结果 |
| --- | --- |
| 源码重放 | **通过**：2,426 个 Strata 文件 + 3,609 个 llama.cpp 文件逐文件 SHA-256 全部匹配，补丁 25 + 3 个 |
| rc.6 增量覆盖 | **通过**：7 个文件覆盖后与 `manifest.json` **逐字节一致（BAD 0）** |
| 干净全量编译 | **通过**：`CLEAN_BUILD_COMPLETE`，退出码 0，约 6 分钟（48 核 / 32 job） |
| 字节可复现 | **否**：同为 29,290,960 B，sha 为 `bdf03ad4ba91…` ≠ `43002a72…`，1,940,181 字节不同 |
| 行为等价 | **通过**：250K `output_sha256 = 08d2c11d70446fe6`，两二进制一致，且与发布记录一致 |

⇒ `clean_rebuild_verified` 由 `false` 改为 **`true`**，但仅指**行为**可重建，**不是**字节可重建。

## 行为等价证据（卡4，renderD132，suite ctx250，249,986 token，同一份 env）

| 臂 | 引擎 | prefill | 250K `output_sha256` | 30-token `output_sha256` | decode |
| --- | --- | --- | --- | --- | --- |
| 发布版 | `43002a72…` | 136,029.1 / 136,036.5 ms（1,837.7 / 1,837.6 tok/s） | `08d2c11d70446fe6` | `6fcfdfac85e2a381` | 349.3 / 319.5 ms |
| 重建版 | `bdf03ad4…` | 136,294.8 / 136,147.6 ms（1,834.2 / 1,836.6 tok/s） | `08d2c11d70446fe6` | `6fcfdfac85e2a381` | 349.1 / 318.8 ms |
| 差异 | — | +0.20% / +0.08%（噪声内） | **一致** | **一致** | 一致 |

两臂 `first_logits_nonfinite = 0`，`retrieval_answer_match` 在两臂取值相同。

## 重建步骤（与 `docs/SOURCE_BUILD.md` 一致）

```bash
# 1) 恢复 c4 基础源码（逐文件校验 2,426 + 3,609）
python3 engines/strata/c4/replay/prepare.py \
  --strata-source pristine-strata --llama-source pristine-llama --out prepared
# → prepared/SOURCE_REPLAY_VERIFIED.json: status=source_replay_verified

# 2) 打 rc.5 快照补丁
cd prepared/strata
git apply -p1 ../../engines/strata/xxs-20261016/patches/c4-to-xxs-20261016.patch

# 3) 覆盖 rc.6 的 7 个文件并逐文件校验
#    结果：7/7 与 manifest.json 一致

# 4) 干净全量编译
bash prepared/recipe/build-in-container.sh prepared     # DTK 容器内
# → CLEAN_BUILD_COMPLETE, prepared/build/strata
```

## 踩到的坑（复用历史归档时必看）

- 远端保留的 `strata_replay_c4_20261015/inputs.tar.gz` 标称"pristine inputs"，**实际是已打过补丁的树**：Strata 有 21 个文件、llama.cpp 有 2 个文件持有的是**补丁后**内容，直接喂给 `prepare.py` 会报 `pristine base mismatch`。
- 处置：对该树反向应用 `strata-c4.patch` / `llama-gfx928.patch` 后，2,422 + 3,608 个基线文件全部匹配，再走正常流程。**不要把这份 inputs.tar.gz 当作原版输入**。
- 补丁脚本 `patch_yf16_*.py` 未参与重建（它们只解释 `GEMM_Y_F16` 的成因），重建靠的是覆盖 7 个文件。

## 复现性边界

1,940,181 / 29,290,960 字节（6.6%）不同，说明构建不是 hermetic 的：路径、时间戳、链接顺序都会改变布局。因此：
- **不要**用重建产物的 sha 去核对发布包；
- 判断等价要看 `output_sha256` + 固定样本，而不是 `cmp`。
