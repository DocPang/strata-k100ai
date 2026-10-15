#!/usr/bin/env python3
"""Generate a container-local Strata config, or run packaged model tools."""
import json
import os
from pathlib import Path
import sys

ROOT = Path('/opt/k100ai')
APP = ROOT / 'app'
SETTINGS = {
    'prefill': ('--prefill', 20480, 1, 262144),
    'max_context': ('--max-context', 262144, 1024, 262144),
    'expert_cache': ('--expert-cache', 24576, 1, 24576),
    'kv_resident': ('--kv-resident', 32768, 0, 262144),
    'spec': ('--spec', 4, 0, 16),
    'spec_min_p': ('--spec-min-p', 0.70, 0, 1),
    'vram_reserve_mib': ('--vram-reserve-mib', 2048, 0, 65536),
    'conversation_cache_mib': ('--conversation-cache-mib', 16384, 0, 1048576),
    'conversation_cache_slots': ('--conversation-cache-slots', 4, 0, 64),
}


def render(profile, settings, env):
    unknown = set(settings) - set(SETTINGS)
    if unknown:
        raise ValueError('Unknown settings: ' + ', '.join(sorted(unknown)))
    args = list(profile['arguments'])
    resolved = {}
    for name, (flag, default, low, high) in SETTINGS.items():
        baseline = type(default)(args[args.index(flag) + 1])
        value = settings.get(name, baseline)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(name + ' must be numeric')
        if isinstance(default, int) and not isinstance(value, int):
            raise ValueError(name + ' must be an integer')
        if not low <= value <= high:
            raise ValueError(name + ' is outside the supported range')
        resolved[name] = value
        args[args.index(flag) + 1] = str(value)
    if resolved['prefill'] > resolved['max_context']:
        raise ValueError('prefill must not exceed max_context')
    if 0 < resolved['kv_resident'] < 20480:
        raise ValueError('kv_resident must be 0 or at least 20480 for c4')
    bindings = {'PACK': '/pack', 'MODEL_SHARD_1': '/models/' + env.get('SHARD1', ''),
                'MODEL_SHARD_2': '/models/' + env.get('SHARD2', ''),
                'MTP_RUNTIME': '/mtp', 'EXPERT_PROFILE': str(APP / 'data/expert-profile.bin')}
    for key in ('SHARD1', 'SHARD2'):
        name = env.get(key, '')
        if not name or Path(name).name != name or name in ('.', '..'):
            raise ValueError(key + ' must be a filename directly inside MODEL_DIR')
    for i, value in enumerate(args):
        for key, path in bindings.items():
            value = value.replace('${' + key + '}', path)
        if '${' in value:
            raise ValueError('Unresolved argument placeholder')
        args[i] = value
    vector = env.get('CONTROL_VECTOR', '')
    if vector:
        if not vector.startswith('/extras/') or '..' in Path(vector).parts:
            raise ValueError('CONTROL_VECTOR must be a file under /extras/')
        args += ['--control-vector-scaled', vector + ':1.0', '--control-vector-layer-range', '4', '44',
                 '--cvec-mode', 'project', '--cvec-dir', 'per-layer']
    engine_env = dict(profile['environment'])
    engine_env.update(HIP_VISIBLE_DEVICES='0', ROCR_VISIBLE_DEVICES='0', HSA_FORCE_FINE_GRAIN_PCIE='1',
                      LD_LIBRARY_PATH='/opt/hyhal/lib:/opt/dtk/lib:/opt/dtk/llvm/lib:/opt/dtk/.hyhal/lib')
    return {'exe': str(ROOT / 'bin/strata'), 'args': args, 'cwd': str(APP),
            'tokenizer': '/pack/tokenizer', 'model_name': 'qwen3.8-flash-next-strata',
            'backend': 'hip', 'gpu': 0, 'env': engine_env, 'open_browser': False,
            'sampling': {'temperature': 0}, 'engine_silence_s': 1800}


def main():
    os.chdir(APP)
    command = sys.argv[1:]
    if command and command[0] == 'tool':
        allowed = {'iq_pack.py', 'mtp_fetch.py', 'mtp_pack.py', 'mtp_rt.py'}
        if len(command) < 2 or command[1] not in allowed:
            raise SystemExit('tool requires one of: ' + ', '.join(sorted(allowed)))
        os.execv(sys.executable, [sys.executable, str(APP / 'tools' / command[1])] + command[2:])
    if command != ['serve']:
        raise SystemExit('Supported commands: serve | tool SCRIPT [ARGS]')
    key = os.environ.get('STRATA_API_KEY', '').strip()
    if len(key) < 16 or key == 'CHANGE_ME_GENERATE_A_PRIVATE_KEY':
        raise SystemExit('Set STRATA_API_KEY to a private key of at least 16 characters in deployment.env')
    profile_id = os.environ.get('PROFILE', 'next-iq3s-single')
    if profile_id not in ('next-iq3s-single', 'next-iq3xxs-single'):
        raise SystemExit('Unsupported PROFILE')
    profile = json.loads((ROOT / 'profiles' / profile_id / 'profile.json').read_text())
    settings = json.loads(Path('/config/settings.json').read_text())
    config = render(profile, settings, os.environ)
    for path in ['/pack/tokenizer', '/mtp', '/opt/hyhal/lib',
                 '/models/' + os.environ['SHARD1'], '/models/' + os.environ['SHARD2']]:
        if not Path(path).exists():
            raise SystemExit('Missing required input: ' + path)
    if os.environ.get('CONTROL_VECTOR') and not Path(os.environ['CONTROL_VECTOR']).is_file():
        raise SystemExit('Missing CONTROL_VECTOR file')
    output = Path('/tmp/k100ai-serving.json')
    output.write_text(json.dumps(config, indent=2) + '\n')
    os.chmod(output, 0o600)
    Path('/var/log/hylog').mkdir(parents=True, exist_ok=True)
    os.execv(sys.executable, [sys.executable, '-m', 'serve.server', '--engine', 'strata',
                            '--config', str(output), '--host', '0.0.0.0', '--port', '8080'])


if __name__ == '__main__':
    main()
