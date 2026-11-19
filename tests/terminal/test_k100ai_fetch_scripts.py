"""K100AI Qwen downloads prepare their pinned helper before the multi-GB shards."""

from __future__ import annotations

import hashlib
import os
import runpy
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "k100ai"
COMMIT = "3cf03257f219afbe7334045ff7c6a06ac68c627d"


class FetchScripts(unittest.TestCase):
    def test_iq3_xxs_model_uses_the_repository_subdirectory(self):
        script = (SCRIPTS / "fetch-qwen-iq3xxs.sh").read_text(encoding="utf-8")
        self.assertIn('"$base/IQ3_XXS/$file"', script)
        self.assertNotIn('"$base/$file"', script)

    def test_both_qwen_scripts_prepare_gguf_py_before_model_downloads(self):
        for name in ("fetch-qwen-iq3s.sh", "fetch-qwen-iq3xxs.sh"):
            script = (SCRIPTS / name).read_text(encoding="utf-8")
            prepare = script.index("fetch-llama-gguf-py.sh")
            download = script.index("downloading $file")
            self.assertLess(prepare, download, name)

    def test_release_packager_ships_download_dependencies_and_requires_ggml(self):
        module = runpy.run_path(str(SCRIPTS / "package-release.py"))
        scripts = set(module["SCRIPT_FILES"])
        self.assertIn("fetch-llama-gguf-py.sh", scripts)
        self.assertIn("fetch-qwen-iq3xxs.sh", scripts)
        source = (SCRIPTS / "package-release.py").read_text(encoding="utf-8")
        self.assertIn('parser.add_argument("--ggml-dir", type=Path, required=True)', source)

    def test_helper_restores_gguf_py_from_a_verified_archive(self):
        with tempfile.TemporaryDirectory(prefix="gguf-py-test-") as temporary:
            root = Path(temporary)
            archive = root / "llama.zip"
            prefix = f"llama.cpp-{COMMIT}"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr(f"{prefix}/gguf-py/gguf/__init__.py", "")
                bundle.writestr(
                    f"{prefix}/gguf-py/gguf/constants.py",
                    "class GGMLQuantizationType:\n    Q2_0 = 42\n",
                )
                bundle.writestr(f"{prefix}/LICENSE", "MIT\n")

            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            env = dict(
                os.environ,
                STRATA_LLAMA_CPP_ZIP=str(archive),
                STRATA_LLAMA_CPP_ZIP_SHA256=digest,
            )
            result = subprocess.run(
                ["bash", str(SCRIPTS / "fetch-llama-gguf-py.sh"), str(root)],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("gguf-py ready", result.stdout)
            restored = root / "third_party/llama.cpp/gguf-py/gguf/constants.py"
            self.assertTrue(restored.is_file())
            self.assertIn("Q2_0 = 42", restored.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
