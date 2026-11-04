#!/usr/bin/env python3
"""Create the deterministic DeepSeek-K100AI benchmark prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys


CHINESE = """
分布式推理服务需要同时处理模型计算、专家权重读取、跨卡通信和请求调度。
当输入很长时，专家访存和注意力阶段会占据大部分时间，而 decode 阶段更依赖
矩阵向量乘法和每次 token 的同步开销。优化时必须先测量真实热点，再判断瓶颈
来自算术吞吐、内存带宽、PCIe 数据搬运还是主机侧调度。一个看似有利的融合
内核可能减少 launch 次数，却增加寄存器压力或降低占用率，因此必须回到完整
模型上验证。发布版本还应保留原路径和明确开关，记录设备、权重、源码提交、
输入长度、重复次数以及中位数，避免把缓存命中或启动预热误认为算子收益。
"""

CODE = """
def route_tokens(hidden, router, experts, top_k):
    logits = router(hidden)
    scores, selected = logits.softmax(dim=-1).topk(top_k, dim=-1)
    result = hidden.new_zeros(hidden.shape)
    for index in range(top_k):
        expert_id = selected[:, index]
        weight = scores[:, index : index + 1]
        result += weight * experts(expert_id, hidden)
    return result


def batch_experts(rows, assignments, capacity):
    rows_by_expert = [[] for _ in range(capacity)]
    for row, expert in enumerate(assignments):
        rows_by_expert[expert].append(row)
    return rows_by_expert
"""

MIXED = """
性能分析先从端到端样本开始。下面的伪代码把每层拆成 attention、router、
gate/up expert、activation、down expert 和 combine；每个阶段既记录耗时，
也记录理论工作量。

```python
for layer in model.layers:
    h = layer.attention(h)
    ids, weights = layer.router(h)
    gate_up = grouped_gemm(h, layer.experts.gate_up, ids)
    act = swiglu(gate_up)
    expert_out = grouped_gemm(act, layer.experts.down, ids)
    h = layer.combine(h, expert_out, weights)
```

每次只改变一个主要因素，例如 GEMM tile、activation layout、融合边界、
共享内存复用或通信重叠。微基准必须使用真实形状，随后检查 API 输出、完整
模型吞吐和固定质量集；如果整模型没有收益，即使局部 kernel 更快也撤回。
"""


CORPORA = {"zh": CHINESE, "code": CODE, "mixed": MIXED}

SUFFIXES = {
    "zh": "\n\n请仅根据以上内容，用三点说明最值得优先验证的性能瓶颈。",
    "code": "\n\n请根据以上代码，用三点指出可能影响扩展性和稳定性的风险。",
    "mixed": "\n\n请根据以上材料，用三点给出按收益优先排序的优化验证顺序。",
}


def load_tokenizer(pack: pathlib.Path):
    from tools.strata_tokenizer import Tokenizer

    tokenizer_dir = pack / "tokenizer"
    ids = json.loads((tokenizer_dir / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(ids)
    for token, index in ids.items():
        tokens[index] = token
    merges = (tokenizer_dir / "merges.txt").read_text(encoding="utf-8").splitlines()
    token_types = json.loads((tokenizer_dir / "token_type.json").read_text(encoding="utf-8"))
    config = json.loads((tokenizer_dir / "tokenizer.json").read_text(encoding="utf-8"))
    return Tokenizer(
        tokens,
        merges,
        token_types,
        config.get("pre", "deepseek3"),
        config.get("special_ids"),
    )


def make_prompt(tokenizer, kind: str, target: int) -> dict:
    suffix = SUFFIXES[kind]
    suffix_ids = tokenizer.encode(suffix)
    budget = max(1, target - len(suffix_ids))
    paragraph = CORPORA[kind].strip()
    repetitions = max(1, (budget * 4) // max(1, len(paragraph)))
    body = "\n\n".join([paragraph] * repetitions)
    ids = tokenizer.encode(body)[:budget]
    text = tokenizer.decode(ids).rstrip() + suffix
    final_ids = tokenizer.encode(text)
    return {
        "name": f"{kind}-{target // 1024}k",
        "kind": kind,
        "target_tokens": target,
        "raw_token_count": len(final_ids),
        "text": text,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    tokenizer = load_tokenizer(args.pack)
    prompts = [make_prompt(tokenizer, kind, target)
               for target in (4096, 8192, 32768)
               for kind in ("zh", "code", "mixed")]
    payload = {
        "schema": 1,
        "pack": str(args.pack),
        "targets": [4096, 8192, 32768],
        "prompts": prompts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    for prompt in prompts:
        print(prompt["name"], prompt["raw_token_count"], prompt["sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
