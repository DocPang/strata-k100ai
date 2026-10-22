# 源码恢复与构建

v0.1.0-rc.7 直接分发已测的 `19b367e0…` 冻结二进制，并且**可从公开材料字节级重建**；上一版 rc.6 分发 `43002a7262e5`，再上一版 rc.5 分发 `d87a4678a21b`。源码恢复与日常安装是两条独立流程，运行包使用者不需要自行编译。

**rc.6 相对 rc.5 的增量源码已公开**，见 [sources/xxs-20261016-gold](../sources/xxs-20261016-gold/)：7 个变更文件、4 个 fail-closed 补丁脚本（按 manifest 顺序应用）、逐文件 SHA-256，以及相对上游 `fb58e0db` 的累计差异。该目录只是增量，不是完整引擎树；更早的平台与内核累计工作见 [c4→xxs-20261016 补丁](../engines/strata/xxs-20261016/patches/c4-to-xxs-20261016.patch)。从这些材料重编译**不保证**得到逐字节相同的二进制，因此 `clean_rebuild_verified` 仍为 `false`（见 [rc.6 引擎清单](../engines/strata/xxs-20261016-gold/manifest.json)）。

## 重建 rc.6 的正确顺序（三步，缺一不可）

> **注意：只应用补丁是重建不出 rc.6 的。** `c4-to-xxs-20261016.patch` 里 `TOPK_H16`、`TOPK_REG_MAX`、`Q6_WAVE_PACK_LDS` 的出现次数都是 **0**——rc.6 这三个开关的实现只存在于 `sources/xxs-20261016-gold/` 公开的源码文件里。4 个 `patch_yf16_*.py` 只是 `GEMM_Y_F16` 的历史成因，**不是**重建手段。

### 第一步：恢复 c4 基础源码

按本文件下半部分（[c4 基础源码恢复](#c4-基础源码恢复)）执行 `prepare.py`，得到 `prepared/strata` 与 `prepared/llama`。

### 第二步：打上 rc.5 快照补丁

```bash
R=/absolute/path/strata-k100ai
git -C prepared/strata apply --check "$R/engines/strata/xxs-20261016/patches/c4-to-xxs-20261016.patch"
git -C prepared/strata apply         "$R/engines/strata/xxs-20261016/patches/c4-to-xxs-20261016.patch"
```

该补丁恢复 10 个变更文件，逐文件摘要见 [恢复清单](../engines/strata/xxs-20261016/source-recovery.json)。

### 第三步：按路径覆盖 rc.6 的 7 个源码文件，并逐个校验

```bash
R=/absolute/path/strata-k100ai
V="$R/sources/xxs-20261016-gold"
for f in \
  include/strata/prefill/gemm.hpp \
  include/strata/prefill/kernels.hpp \
  src/prefill/gemm.cu \
  src/prefill/kernels.cu \
  src/prefill/prefill.cpp \
  src/kernels/cuda/qsa_select.cu \
  src/kernels/cuda/native_mmvq.cu ; do
  install -m 644 "$V/$f" "prepared/strata/$f"
done

# 逐文件校验：必须与 manifest.json 完全一致
python3 - <<'PYEOF'
import hashlib, json, pathlib
V = pathlib.Path("/absolute/path/strata-k100ai/sources/xxs-20261016-gold")
man = json.loads((V / "manifest.json").read_text())
bad = 0
for e in man["files"]:
    if not e["path"].startswith(("include/", "src/")):
        continue
    p = pathlib.Path("prepared/strata") / e["path"]
    got = hashlib.sha256(p.read_bytes()).hexdigest()
    ok = got == e["sha256"]
    bad += 0 if ok else 1
    print(("OK  " if ok else "FAIL"), e["path"], got[:16])
raise SystemExit(1 if bad else 0)
PYEOF
```

7 个文件全部 `OK` 才继续；任何一个 `FAIL` 说明前两步的输入或补丁版本不对。

### 叠加 rc.7 增量（`sources/xxs-20261022-sol/`）

在 rc.6 的 7 个文件覆盖完成、校验通过之后，再覆盖 rc.7 的 7 个文件（其中 `gemm.hpp`、`gemm.cu`、`kernels.cu` 与 rc.6 同名，覆盖即替换）：

```bash
S=sources/xxs-20261022-sol
for f in CMakeLists.txt include/strata/prefill/gemm.hpp \
         src/kernels/cuda/dequant_bf16.cu src/prefill/gateup_exact.hpp \
         src/prefill/gemm.cu src/prefill/kernels.cu src/prefill/moe_mmq.cu; do
  install -D -m 644 "$S/$f" "prepared/strata/$f"
done

# 逐文件校验（与 rc.6 同一套逻辑，换目录即可）
python3 - <<'PYEOF'
import hashlib, json, pathlib
V = pathlib.Path("/absolute/path/strata-k100ai/sources/xxs-20261022-sol")
man = json.loads((V / "manifest.json").read_text())
bad = 0
for e in man["files"]:
    p = pathlib.Path("prepared/strata") / e["path"]
    got = hashlib.sha256(p.read_bytes()).hexdigest()
    ok = got == e["sha256"]
    bad += 0 if ok else 1
    print(("OK  " if ok else "FAIL"), e["path"], got[:16])
raise SystemExit(1 if bad else 0)
PYEOF
```

7 个文件全部 `OK` 后，`prepared/strata` 与发布二进制对应的源码树即**逐字节一致**（本次已验证：与部署树 `diff -rq` 零差异），随后干净编译产物的 SHA-256 与发布资产**完全相同**。见 [REBUILD_REPORT.md](../sources/xxs-20261022-sol/REBUILD_REPORT.md)。

> 注意：`sources/xxs-20261022-sol/patches/` 的三个脚本（`patch_gemsol.py`、`patch_gemsol2.py`、`patch_cmake_rocblas.py`）**不需要执行**——它们的作用是说明 pin hook 是怎么加进去的；直接覆盖上面 7 个文件即可得到等价结果。

### 关于 4 个 `patch_yf16_*.py`

它们的路径契约是 `<workdir>/source/{include,src}/...`（`W = argv[1]`，然后拼 `W/"source/..."`），与本文件的 `prepared/strata/...` 布局不同。若要复跑这些脚本，必须传入一个含有 `source/` 子目录的 workdir，例如 `ln -s prepared/strata workdir/source`。它们的存在是为了说明 `GEMM_Y_F16` 是怎么一步步改出来的（含 `f16_inplace` 漏改导致的 NaN 修复），**重建时不需要执行**。

### 编译与验收

```bash
bash engines/strata/c4/replay/build-in-container.sh /work   # 全量编译、链接（容器内）
# 之后必须做：模型加载 + 固定样本输出比对（不是只看编译通过）
```

历史构建曾复用已有静态对象；**从源码重编译不保证得到逐字节相同的二进制**（编译器、DTK 版本、构建目录都会影响）。字节身份以 `manifest.json` 的 `engine_sha256` 核对发布包，而不是核对重编译产物。在完成"干净编译 + 固定样本输出一致"之前，`clean_rebuild_verified` 保持 `false`。

# c4 基础源码恢复

**以下是 c4 基础源码的恢复与构建步骤；当前 XXS 版本还需应用上文的附加补丁。完整干净编译、链接和新产物 GPU 等价验证尚未完成。** 历史构建曾复用旧目标文件，不能用旧增量构建脚本证明别人能够全量重建。

## 固定输入

- Strata：`https://github.com/Niko1221/Strata.git`，`fb58e0dbc8399662c0e47c76578c6e878b14f6cf`。
- llama.cpp：`https://github.com/ggml-org/llama.cpp.git`，`3cf03257f219afbe7334045ff7c6a06ac68c627d`。
- DTK：使用 [INSTALL.md](INSTALL.md) 的固定镜像。
- 历史引擎 SHA-256：`c4be512a7d257d9b56bc325136431e5feffd2b6d76f52378051829bfe73fc903`。

## 恢复源码

在新工作目录执行，RECIPE 指向本仓库内配方绝对路径。开发者源码恢复工具需要 Python 3；普通安装用户不需要宿主 Python。

```bash
git clone https://github.com/Niko1221/Strata.git strata-original
git -C strata-original checkout --detach fb58e0dbc8399662c0e47c76578c6e878b14f6cf
git clone https://github.com/ggml-org/llama.cpp.git llama-original
git -C llama-original checkout --detach 3cf03257f219afbe7334045ff7c6a06ac68c627d
RECIPE=/absolute/path/strata-k100ai/engines/strata/c4/replay
python3 "$RECIPE/prepare.py" \
  --strata-source "$PWD/strata-original" \
  --llama-source "$PWD/llama-original" \
  --out "$PWD/prepared"
```

脚本校验原版输入、补丁与应用后文件的哈希，在新的输出目录中恢复源码，输入 checkout 保持只读。源码恢复成功覆盖 2,426 个 Strata 文件和 3,609 个 llama.cpp 文件。补丁为累计源码快照，逐次实验提交历史未纳入该快照。

## 在固定 DTK 容器中构建

```bash
BASE_IMAGE=harbor.sourcefind.cn:5443/hcu/admin/base/sglang@sha256:878d317dfa18efae70516a59cc836efc77b251efbe4ca85de67171bc2c3aca68
docker run --rm --network none --cpus=8 --memory=40g --pids-limit=2048 \
  -v "$PWD/prepared:/work" -v "$RECIPE:/recipe:ro" \
  --entrypoint bash "$BASE_IMAGE" /recipe/build-in-container.sh /work
```

构建使用 CPU 和源码输入，创建全新 build 目录。只检查配置可加 `-e K100AI_CONFIGURE_ONLY=1`；配置生成、编译链接与运行验收分别记录。

关键配方包括 `STRATA_HIP_GFX906=ON`、`STRATA_ENABLE_HIP=OFF`、目标 `gfx928`、gfx928 dot4、hipBLASLt、early-inline/no-function-calls。命名中的 gfx906 是该项目的 wave64 兼容后端，不代表目标硬件是 gfx906。脚本只在一次性构建容器中修正 HSA include 链接。

构建后需要重新做模型加载、数值/文本、固定负载 A/B/A、长上下文与缓存、服务接口和稳定性门槛。重新编译二进制 SHA 不同不自动判错；必须记录工具链和实际运行结果。不要覆盖正在使用的历史 Release 资产。
