# 复现报告（2026-10-22）

用**已公开的材料**重建 `19b367e0…`，验证重建路径可用，并且重建产物与发布二进制**逐字节相同**。

## 结论

| 项 | 结果 |
| --- | --- |
| 源码重放 | **通过**：`prepared` 树（纯净重放，2,426 + 3,609 文件已在 rc.6 复现报告中校验） |
| rc.7 增量覆盖 | **通过**：本目录 7 个文件覆盖后与部署源码树 `diff -rq` **零差异** |
| 干净全量编译 | **通过**：`CLEAN_BUILD_COMPLETE`，退出码 0 |
| 字节可复现 | **是**：重建 `19b367e0…` = 发布 `19b367e0…`，29,327,416 B，**SHA-256 完全相同** |
| 行为等价 | **通过**：三窗 logits 逐位一致，prefill 差 0.016% |

> 与 rc.6 的差别：rc.6 只能做到**行为**可重建（重建产物 `bdf03ad4…` ≠ `43002a72…`，1,940,181 字节不同）；rc.7 达到**字节**可重建。

## 步骤

```bash
# 1) 纯净重放（与 docs/SOURCE_BUILD.md 一致）
python3 engines/strata/c4/replay/prepare.py \
  --strata-source pristine-strata --llama-source pristine-llama --out prepared

# 2) rc.5 补丁 + rc.6 的 7 文件覆盖（见 docs/SOURCE_BUILD.md）

# 3) 覆盖本目录 7 个文件（rc.7 增量）
S=sources/xxs-20261022-sol
for f in CMakeLists.txt include/strata/prefill/gemm.hpp \
         src/kernels/cuda/dequant_bf16.cu src/prefill/gateup_exact.hpp \
         src/prefill/gemm.cu src/prefill/kernels.cu src/prefill/moe_mmq.cu; do
  install -D -m 644 "$S/$f" "prepared/strata/$f"
done

# 4) 干净编译（固定 DTK 容器，8 job）
docker run --rm --network none --cpus=8 --memory=40g --pids-limit=2048 \
  -v "$PWD/prepared:/work" -v "$PWD/engines/strata/c4/replay:/recipe:ro" \
  --entrypoint bash \
  harbor.sourcefind.cn:5443/hcu/admin/base/sglang@sha256:878d317dfa18efae70516a59cc836efc77b251efbe4ca85de67171bc2c3aca68 \
  /recipe/build-in-container.sh /work
# → CLEAN_BUILD_COMPLETE
```

## 证据

### 源码树一致性

```
$ diff -rq prepared/strata  <部署源码树>/strata
（无输出）
```

### 二进制一致性

```
$ sha256sum prepared/build/strata
19b367e015d006694737ba067f0a4ac25da09c40be19b0d669fc1e09444152a3  prepared/build/strata
$ sha256sum <发布二进制>
19b367e015d006694737ba067f0a4ac25da09c40be19b0d669fc1e09444152a3  strata
```

### 运行期等价（卡4，renderD132，suite ctx128，126,857 token，同一份 env 含 `STRATA_GEMM_SOL`）

| 臂 | 引擎 | prefill | logits 窗口 0 / 1 / 2 |
| --- | --- | ---: | --- |
| 重建版 | 本次编译产物 `19b367e0…` | **60,328.8 ms** | `74cd5f8ff85702c6…` / `05c7409976b023c6…` / `05c7409976b023c6…` |
| 发布版 | 生产在跑 `19b367e0…` | **60,319.4 ms** | `74cd5f8ff85702c6…` / `05c7409976b023c6…` / `05c7409976b023c6…` |
| 差异 | — | +0.016%（噪声） | **逐位一致** |

两个臂使用同一二进制（SHA 相同），因此这一步验证的是"编译产物可运行 + 输出与发布记录一致"，不是两个不同实现的对比。

## 边界

- 本报告验证的是**同一工具链、同一配方**下的可复现性。换 DTK 版本或换编译器可能改变字节（rc.6 的重建就出现过这种情况）。
- 逐项补丁来源仍只有 `gemm.cu` / `CMakeLists.txt` 可逐条复现；其余 5 个文件以整文件快照归档。重建不受影响，但逐项收益归因仍缺。
