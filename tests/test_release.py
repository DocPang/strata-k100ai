import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('entrypoint', ROOT / 'containers/strata/entrypoint.py')
entrypoint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entrypoint)


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.profile = json.loads((ROOT / 'profiles/next-iq3s-single/profile.json').read_text())
        self.settings = json.loads((ROOT / 'profiles/next-iq3s-single/settings.json').read_text())
        self.env = {'SHARD1': 'first.gguf', 'SHARD2': 'second.gguf'}

    def test_default_and_override_reach_engine(self):
        config = entrypoint.render(self.profile, self.settings, self.env)
        self.assertNotIn('${', json.dumps(config))
        self.assertEqual(config['args'][config['args'].index('--prefill') + 1], '20480')
        self.assertNotIn('--control-vector-scaled', config['args'])
        self.settings['prefill'] = 8192
        config = entrypoint.render(self.profile, self.settings, self.env)
        self.assertEqual(config['args'][config['args'].index('--prefill') + 1], '8192')
        self.assertEqual(config['env']['STRATA_SH_STREAM'], '0')

    def test_control_vector_is_explicit(self):
        self.env['CONTROL_VECTOR'] = '/extras/vector.gguf'
        args = entrypoint.render(self.profile, self.settings, self.env)['args']
        self.assertIn('/extras/vector.gguf:1.0', args)

    def test_invalid_settings_fail_instead_of_silent_fallback(self):
        for name, value in [('prefill', -1), ('spec', 1.5), ('unknown', 1), ('spec_min_p', float('nan'))]:
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    entrypoint.render(self.profile, {**self.settings, name: value}, self.env)
        with self.assertRaises(ValueError):
            entrypoint.render(self.profile, {**self.settings, 'max_context': 8192}, self.env)
        with self.assertRaises(ValueError):
            entrypoint.render(self.profile, self.settings, {**self.env, 'SHARD1': '../private.gguf'})


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for name in ['install.sh', 'deployment.env.example']:
            shutil.copy(ROOT / name, self.root / name)
        for name in ['containers', 'engines', 'profiles']:
            shutil.copytree(ROOT / name, self.root / name)
        (self.root / 'deployment.env').write_text('existing-settings\n')
        bin_dir = self.root / 'bin'; bin_dir.mkdir()
        (bin_dir / 'uname').write_text('#!/bin/sh\ncase "$1" in -s) echo Linux;; -m) echo x86_64;; esac\n')
        (bin_dir / 'docker').write_text('''#!/bin/sh
printf '%s\\n' "$*" >> "$CALLS"
case "$*" in
 "info") exit 0;;
 "image inspect --format "*) echo "${MOCK_BASE_ID:-sha256:caeff0e9e6efef918a31359c4764b18bf2d428f2b76e3580c788f4cc8b0e93ad}";;
 "image inspect "*) test "${MISSING_BASE:-0}" != 1;;
 "buildx version") exit 0;;
 "buildx inspect default --format "*) echo docker;;
 "tag "*|"build "*) exit 0;;
 *) echo "Unexpected docker operation" >&2; exit 99;;
esac
''')
        for p in bin_dir.iterdir(): p.chmod(0o755)
        self.env = {**os.environ, 'PATH': str(bin_dir) + ':' + os.environ['PATH'], 'CALLS': str(self.root / 'calls')}

    def tearDown(self):
        self.temp.cleanup()

    def run_install(self, *args):
        return subprocess.run(['bash', str(self.root / 'install.sh'), *args], env=self.env,
                              capture_output=True, text=True)

    def test_local_only_missing_base_never_pulls(self):
        self.env['MISSING_BASE'] = '1'
        result = self.run_install('--local-only')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('prohibits pulling', result.stderr)
        self.assertNotIn('pull ', (self.root / 'calls').read_text())

    def test_wrong_base_is_rejected(self):
        self.env['MOCK_BASE_ID'] = 'sha256:wrong'
        result = self.run_install('--local-only')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Unsupported base', result.stderr)

    def test_tampered_bundle_never_builds(self):
        bad = self.root / 'bad.tar.gz'; bad.write_bytes(b'bad')
        result = self.run_install('--local-only', '--bundle', str(bad))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('checksum mismatch', result.stderr)
        self.assertNotIn('build --', (self.root / 'calls').read_text())

    def test_offline_install_contract(self):
        bundle = ROOT / 'local-artifacts/strata-c4-linux-amd64.tar.gz'
        if not bundle.exists(): self.skipTest('Release bundle is checked separately on the publishing host')
        result = self.run_install('--local-only', '--bundle', str(bundle))
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = (self.root / 'calls').read_text()
        self.assertNotIn('pull ', calls)
        self.assertIn('--network=none', calls)
        self.assertNotIn('run ', calls)
        self.assertEqual((self.root / 'deployment.env').read_text(), 'existing-settings\n')


class EnvironmentTests(unittest.TestCase):
    def test_environment_file_is_not_executed(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / 'deployment.env').write_text('PORT=$(touch should-not-exist)\n')
            result = subprocess.run(['bash', '-c', 'source "$1"; load_deployment_env; printf "%s" "$PORT"',
                                     '_', str(ROOT / 'scripts/config.sh')], cwd=folder, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, '$(touch should-not-exist)')
            self.assertFalse((folder / 'should-not-exist').exists())

    def test_unrecognized_environment_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / 'deployment.env').write_text('PATH=unexpected\n')
            result = subprocess.run(['bash', '-c', 'source "$1"; load_deployment_env',
                                     '_', str(ROOT / 'scripts/config.sh')], cwd=folder, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Unknown deployment setting', result.stderr)


if __name__ == '__main__':
    unittest.main()
