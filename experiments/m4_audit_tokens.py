#!/usr/bin/env python3
"""Run a separate tokenizer diagnostic using an archived M4 trial's model and libraries."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shlex

from m3_preflight import Recorder


def checked(result):
    if result['returncode'] != 0:
        raise RuntimeError(f"{result['name']} failed: {result['stderr']}")
    return result['stdout']


def verify_hashes(output, expected):
    observed = {}
    for line in output.splitlines():
        digest, name = line.split(maxsplit=1)
        observed[name.strip()] = digest
    for name, digest in expected.items():
        if observed.get(name) != digest:
            raise ValueError('device artifact hash mismatch: ' + name)
    return {name: observed[name] for name in expected}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('trial', type=Path)
    p.add_argument('--serial', required=True)
    a = p.parse_args()
    m = json.loads((a.trial / 'manifest.json').read_text())
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%H%M%S-%f')
    out = a.trial.parent / (a.trial.name + '-token-audit-' + stamp)
    out.mkdir(exist_ok=False)
    r = Recorder(out)
    adb = ['adb', '-s', a.serial]
    remote = m['remote_dir']
    binary = Path('tmp/m4-build/bundle/m4_token_audit')
    source = Path('experiments/m4_token_audit.cc')
    (out / source.name).write_bytes(source.read_bytes())
    binary_hash = hashlib.sha256(binary.read_bytes()).hexdigest()
    expected = {'/data/local/tmp/litertlm/model.litertlm': m['model_observed_sha256']}
    expected.update({remote + '/' + name: digest for name, digest in m['bundle_sha256'].items()
                     if name.endswith('.so')})
    expected[remote + '/budget70-1.json'] = m['inputs_sha256']['budget70-1.json']
    checks = checked(r.run('verify_parent_artifacts', adb + ['shell',
                     'sha256sum ' + shlex.join(list(expected))], 180))
    observed = verify_hashes(checks, expected)
    checked(r.run('push_token_audit', adb + ['push', str(binary), remote + '/m4_token_audit']))
    binary_check = checked(r.run('verify_token_audit', adb + ['shell',
                           'sha256sum ' + shlex.quote(remote + '/m4_token_audit')]))
    observed.update(verify_hashes(binary_check, {remote + '/m4_token_audit': binary_hash}))
    command = ('cd ' + shlex.quote(remote) + ' && LD_PRELOAD=' + shlex.quote(remote + '/libLiteRt.so')
               + ' LD_LIBRARY_PATH=' + shlex.quote(remote)
               + ' ./m4_token_audit /data/local/tmp/litertlm/model.litertlm '
               + shlex.quote(remote + '/budget70-1.json'))
    result = r.run('audit', adb + ['shell', command], 180)
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    checked(result)
    for name in ['rendered.txt', 'part-0.txt', 'part-1.txt', 'part-2.txt']:
        checked(r.run('pull_' + name, adb + ['pull', remote + '/' + name, str(out / name)]))
    meta = {'parent_trial': a.trial.name,
            'native_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'native_binary_sha256': binary_hash,
            'model_sha256': observed['/data/local/tmp/litertlm/model.litertlm'],
            'source_commit': m['source_commit'], 'device_artifacts_sha256': observed,
            'raw_sha256': {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                           for f in out.iterdir() if f.is_file()}}
    (out / 'manifest.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(out, result['returncode'], result['stdout'])


if __name__ == '__main__':
    main()
