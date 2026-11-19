#!/usr/bin/env python3
"""生成本报告 §2 使用的长文档（固定种子，可复现）。

用法: python3 gen-longdoc.py <词数> <输出文件>
115,000 词 ≈ 115,093 tokens；480,000 词 ≈ 480,093 tokens（本引擎 tokenizer）。
needle 置于 90% 深度，结尾为检索问题。
"""
import random
import sys

random.seed(20261119)
WORDS = ("the of and to in that is was he for it with as his on be at by i this had not are but from or have an "
         "they which one you were her all she there would their we him been has when who will no more if out so "
         "said what up its about into than them can only other new some could time these two may then do first "
         "any my now such like our over man me even most made after also did many before must through back years "
         "where much your way well down should because each just those people mr how too little state good very "
         "make world still own see men work long get here between both life being under never day same another "
         "know while last might us great old year off come since against go came right used take three").split()

NEEDLE = "\n\nThe vault access code for the archive is KILO-4472-VV. Keep it confidential.\n\n"
QUESTION = "\n\nQuestion: What is the vault access code? Answer with the code only."


def para(n: int) -> str:
    out = []
    while len(out) < n:
        out += random.sample(WORDS, random.randint(12, 22))
        out.append(".")
    return " ".join(out[:n])


def main() -> None:
    n = int(sys.argv[1])
    path = sys.argv[2]
    doc = para(n)
    cut = int(len(doc) * 0.9)
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc[:cut] + NEEDLE + doc[cut:] + QUESTION)
    print(f"wrote {n} words to {path}")


if __name__ == "__main__":
    main()
