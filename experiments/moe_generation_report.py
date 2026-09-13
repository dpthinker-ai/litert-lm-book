#!/usr/bin/env python3
"""Read-only audit/summary of D21-D25 MoE generation records (stdlib only).

Inputs are experiment directories or individual run directories. No model is
loaded. Existing summary.json files are never used as measurement evidence.
Missing or invalid records are excluded and cause exit 2; valid runs remain
visible. A missing archive manifest is reported, never treated as verified.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


class InvalidRecord(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise InvalidRecord(message)


def read_json(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def number(value, name, minimum=0):
    require(type(value) in (int, float) and math.isfinite(value) and value >= minimum,
            f'{name}: missing, non-finite, or invalid number')
    return value


def integer(value, name, minimum=0):
    number(value, name, minimum)
    require(type(value) is int, f'{name}: expected integer')
    return value


def stats(values):
    values = sorted(values)
    if not values:
        return {'n': 0, 'median': None, 'min': None, 'max': None}
    return {'n': len(values), 'median': statistics.median(values),
            'min': values[0], 'max': values[-1]}


def checked_files(root, entries):
    require(isinstance(entries, dict) and entries, 'file hash map missing or empty')
    for name, value in entries.items():
        path = (root / name).resolve()
        require(path.is_relative_to(root.resolve()), f'hash path escapes directory: {name}')
        digest = value.get('sha256') if isinstance(value, dict) else value
        require(isinstance(digest, str) and len(digest) == 64, f'invalid SHA-256: {name}')
        require(path.is_file() and sha(path) == digest, f'missing file or SHA-256 mismatch: {name}')
        if isinstance(value, dict) and 'bytes' in value:
            require(path.stat().st_size == value['bytes'], f'file size mismatch: {name}')


def request_record(item, index, report, worker):
    require(item.get('index') == index, 'request index/order mismatch')
    for key in ('generation_completed', 'conversation_closed', 'stream_final'):
        require(item.get(key) is True, f'request {index}: {key} not true')
    require('stream_error' in item and item['stream_error'] is None, 'stream error or missing error field')
    events = item['events']
    require(isinstance(events, list) and events, 'missing stream events')
    parts, arrivals = [], []
    previous = 0
    for position, event in enumerate(events):
        elapsed = number(event.get('elapsed_seconds'), 'event elapsed_seconds')
        require(elapsed >= previous, 'non-monotonic callback timestamps')
        previous = elapsed
        require('error' in event and event['error'] is None, 'stream event error')
        require(type(event.get('is_final')) is bool, 'missing final flag')
        require(event['is_final'] == (position == len(events) - 1), 'final event missing or followed by events')
        require(isinstance(event.get('message'), str), 'missing event message')
        texts = []
        if event['message']:
            message = json.loads(event['message'])
            require(isinstance(message, dict) and isinstance(message.get('content'), list), 'invalid stream message')
            texts = [c['text'] for c in message['content']
                     if c.get('type') == 'text' and c.get('text')]
            require(all(isinstance(t, str) for t in texts), 'non-string text')
        if texts:
            parts.extend(texts)
            arrivals.append(elapsed)
    text = ''.join(parts)
    require(text and item.get('text') == text, 'missing text or reconstructed text mismatch')
    require(item.get('first_text_callback_seconds') == arrivals[0], 'first text timestamp mismatch')
    wall = number(item.get('generation_wall_seconds'), 'generation_wall_seconds')
    require(wall >= events[-1]['elapsed_seconds'], 'wall time precedes final callback')
    benchmark = item['benchmark']
    prefill = integer(benchmark.get('last_prefill_token_count'), 'prefill count', 1)
    decode = integer(benchmark.get('last_decode_token_count'), 'decode count', 2)
    rate = number(benchmark.get('last_decode_tokens_per_second'), 'runtime decode rate', 1e-30)
    cap = integer(report.get('max_output_tokens'), 'output cap', 1)
    require(decode <= cap, 'decode count exceeds configured output cap')
    multi = 'turns_per_history' in report
    session = integer(item.get('history') if multi else index, 'session index')
    turn = integer(item.get('turn') if multi else 0, 'turn index')
    after = integer(item.get('conversation_token_count'), 'conversation count')
    if multi:
        before = integer(item.get('conversation_token_count_before'), 'conversation count before')
        require(after == before + prefill + decode, 'conversation increment mismatch')
    else:
        before = None
    gaps = [(right - left) * 1000 for left, right in zip(arrivals, arrivals[1:])]
    answer_matches = None
    if 'expected' in item:
        try:
            answer_matches = json.loads(text) == item['expected']
        except json.JSONDecodeError:
            answer_matches = False
        require(item.get('state_matches') is answer_matches, 'saved JSON comparison differs from recomputation')
    return dict(index=index, session=session, turn=turn,
                engine_request='first' if index == 0 else 'subsequent',
                session_request='first' if turn == 0 else 'subsequent',
                prompt=item.get('prompt', worker.get('prompt')),
                prefill_tokens=prefill, decode_tokens=decode, output_cap=cap,
                reached_output_cap=decode == cap, text=text,
                first_text_ms=arrivals[0] * 1000, generation_wall_ms=wall * 1000,
                runtime_decode_tokens_per_second=rate,
                text_callback_intervals_ms=stats(gaps),
                text_callback_max_gap_ms=max(gaps) if gaps else None,
                conversation_count_before=before, conversation_count_after=after,
                json_state_matches=answer_matches,
                answer_contains_marker=('AMBER-42' in text.upper()) if 'answer_contains_marker' in item else None)


def measurements(requests):
    return dict(requests=len(requests),
                first_text_ms=stats([r['first_text_ms'] for r in requests]),
                generation_wall_ms=stats([r['generation_wall_ms'] for r in requests]),
                runtime_decode_tokens_per_second=stats([r['runtime_decode_tokens_per_second'] for r in requests]),
                prefill_tokens=stats([r['prefill_tokens'] for r in requests]),
                decode_tokens=stats([r['decode_tokens'] for r in requests]),
                reached_output_cap=sum(r['reached_output_cap'] for r in requests),
                json_state_checks=sum(r['json_state_matches'] is not None for r in requests),
                json_state_matches=sum(r['json_state_matches'] is True for r in requests),
                marker_checks=sum(r['answer_contains_marker'] is not None for r in requests),
                marker_matches=sum(r['answer_contains_marker'] is True for r in requests))


def summarize_run(directory, archive):
    report = read_json(directory / 'report.json')
    worker = read_json(directory / 'worker.json')
    require(report.get('worker_result') == worker, 'report and worker.json disagree')
    checked_files(directory, report.get('files'))
    require(all(name in report['files'] for name in ('worker.json', 'stderr.log', 'resource-guard.json')),
            'run hashes do not cover worker/log/resource records')
    require(report.get('returncode') == 0 and report.get('status') == 'GENERATION_COMPLETED'
            and report.get('stop_reason') is None, 'run failed, incomplete, or stopped by guard')
    require(worker.get('phase') == 'completed', 'worker not completed')
    for key in ('engine_created', 'engine_closed', 'generation_completed'):
        require(worker.get(key) is True, f'{key} not true')
    require(report.get('backend_errors') == [], 'backend errors missing or nonempty')
    native = (directory / 'stderr.log').read_text(errors='replace')
    bad_markers = ('validation error:', 'binding entry buffer not set', 'unresolved value',
                   'out of memory', 'failed to execute', 'fatal error')
    require(not any(marker in native.lower() for marker in bad_markers), 'backend error in native log')
    backend = dict(artisan='backend: GPU_ARTISAN' in native,
                   metal='llm_metal_runner' in native,
                   f16='CalculationsPrecision::F16' in native)
    require(all(backend.values()), 'Artisan/Metal/F16 execution evidence missing; outside D21-D25 scope')
    repeats = integer(report.get('repetitions_in_engine'), 'repetitions', 1)
    turns = integer(report.get('turns_per_history', 1), 'turns per history', 1)
    require(len(worker['generations']) == repeats * turns, 'request count differs from configured repetitions')
    requests = [request_record(g, i, report, worker) for i, g in enumerate(worker['generations'])]
    sessions = []
    for session in range(repeats):
        rows = [r for r in requests if r['session'] == session]
        require([r['turn'] for r in rows] == list(range(turns)), 'session/turn sequence mismatch')
        if turns > 1:
            require(rows[0]['conversation_count_before'] == 0, 'new session count not zero')
            for left, right in zip(rows, rows[1:]):
                require(left['conversation_count_after'] == right['conversation_count_before'], 'history count discontinuity')
        sessions.append(dict(session=session, engine_first_request=session == 0,
                             first=measurements(rows[:1]), subsequent=measurements(rows[1:]),
                             final_conversation_count=rows[-1]['conversation_count_after']))
    samples = read_json(directory / 'resource-guard.json')
    require(isinstance(samples, list) and samples, 'resource samples missing')
    for sample in samples:
        integer(sample.get('pressure'), 'system pressure')
        require(not sample['pressure'] & 4, 'critical pressure sample in completed run')
        require(not sample.get('error') and not sample.get('rusage_error'), 'resource capture error')
    stages = worker.get('memory_stages', [])
    memory = 'memory_stages' in worker
    if memory:
        names = ['before_engine', 'engine_created', 'short_conversation_created',
                 'short_generation_finished', 'short_conversation_closed', 'long_conversation_created',
                 'long_generation_finished', 'long_conversation_closed', 'engine_closed']
        require([s['stage'] for s in stages] == names, 'memory stages missing or reordered')
        for stage in stages:
            require(not stage.get('capture_error'), 'memory capture error')
            for field in ('resident_size', 'phys_footprint', 'wired_size', 'pageins'):
                integer(stage.get(field), field)
            if stage['stage'] in ('engine_created', 'short_generation_finished', 'long_generation_finished', 'engine_closed'):
                require(stage.get('vmmap_returncode') == 0, 'vmmap failed or missing')
                require(stage['stage'] + '-vmmap.txt' in report['files'], 'vmmap file not covered by hashes')
    condition_keys = ('model', 'library_sha256', 'machine', 'platform', 'packages', 'requested_backend',
                      'context_capacity', 'max_output_tokens', 'benchmark_enabled', 'thinking',
                      'speculative_decoding', 'top_k', 'temperature', 'seed')
    require(all(key in report for key in condition_keys), 'run conditions missing')
    require(isinstance(report['model'].get('sha256'), str), 'model hash missing')
    conditions = {key: report[key] for key in condition_keys}
    conditions['model'] = {key: report['model'][key] for key in ('sha256', 'bytes')}
    if memory:
        conditions['input_raw_token_budget'] = integer(report.get('input_raw_token_budget'), 'input raw token budget')
    return dict(directory=str(directory), manifest=archive, run_file_hashes='verified',
                report_sha256=sha(directory / 'report.json'), worker_sha256=sha(directory / 'worker.json'),
                collector_sha256=report.get('script_sha256'), conditions=conditions,
                kind='memory' if memory else ('multiturn' if turns > 1 else 'fresh_sessions'),
                backend_log_evidence=backend, engine_stage_ms=number(worker.get('engine_create_seconds'), 'engine stage') * 1000,
                requests=requests, sessions=sessions, memory_stages=stages,
                pressure_counts=dict(Counter(str(s['pressure']) for s in samples)),
                sampled_memory_max_bytes={key: max((s[key] for s in samples if key in s), default=None)
                                          for key in ('resident_size', 'phys_footprint')})


def summarize(paths):
    runs, errors, duplicates, archives = [], [], [], []
    seen_paths, seen_workers = set(), set()
    for argument in paths:
        root = Path(argument).resolve()
        try:
            require(root.is_dir(), 'input must be a run or experiment directory')
            manifest_path = root / 'manifest.json'
            archive = dict(directory=str(root), status='not_present')
            if manifest_path.exists():
                entries = read_json(manifest_path)['files']
                checked_files(root, entries)
                archive.update(status='verified', manifest_sha256=sha(manifest_path), files=len(entries))
            archives.append(archive)
            reports = [root / 'report.json'] if (root / 'report.json').exists() else sorted(root.rglob('report.json'))
            require(reports, 'no report.json found')
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            errors.append(dict(directory=str(root), error=str(exc)))
            continue
        for path in reports:
            directory = path.parent
            if directory in seen_paths:
                duplicates.append(str(directory))
                continue
            seen_paths.add(directory)
            try:
                if archive['status'] == 'verified':
                    require(str(path.relative_to(root)) in entries, 'report is not covered by archive manifest')
                run = summarize_run(directory, archive)
                if run['worker_sha256'] in seen_workers:
                    duplicates.append(str(directory))
                    continue
                seen_workers.add(run['worker_sha256'])
                runs.append(run)
            except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
                errors.append(dict(directory=str(directory), error=str(exc)))
    grouped = defaultdict(list)
    for run in runs:
        if run['kind'] == 'memory':
            continue
        for request in run['requests']:
            key = json.dumps(dict(conditions=run['conditions'], kind=run['kind'], prompt=request['prompt'],
                                  engine_request=request['engine_request'], session_request=request['session_request']), sort_keys=True)
            grouped[key].append((run, request))
    groups = []
    for key, items in sorted(grouped.items()):
        groups.append(dict(group=json.loads(key), processes=len({r['directory'] for r, _ in items}),
                           run_directories=sorted({r['directory'] for r, _ in items}),
                           **measurements([request for _, request in items])))
    memory_groups = defaultdict(list)
    for run in runs:
        if run['kind'] == 'memory':
            workload = [{key: request[key] for key in ('session', 'turn', 'prompt', 'prefill_tokens')}
                        for request in run['requests']]
            key = json.dumps(dict(conditions=run['conditions'], workload=workload), sort_keys=True)
            memory_groups[key].append(run)
    memory = []
    for key, items in sorted(memory_groups.items()):
        names = [s['stage'] for s in items[0]['memory_stages']]
        rows = []
        for name in names:
            rows.append(dict(stage=name, **{field + '_gib': stats([next(s[field] for s in r['memory_stages'] if s['stage'] == name) / 2**30 for r in items])
                                           for field in ('resident_size', 'phys_footprint')}))
        memory.append(dict(**json.loads(key), processes=len(items), stages=rows))
    return dict(schema_version=1, valid=not errors, errors=errors, duplicate_runs_excluded=duplicates,
                archives=archives, included_processes=len(runs),
                performance_processes=sum(r['kind'] != 'memory' for r in runs),
                performance_requests=sum(len(r['requests']) for r in runs if r['kind'] != 'memory'),
                memory_processes=sum(r['kind'] == 'memory' for r in runs),
                memory_requests=sum(len(r['requests']) for r in runs if r['kind'] == 'memory'),
                runs=runs, groups=groups, memory_groups=memory,
                methods=['Only report/worker/events/logs and file hashes are evidence; existing summary files are not used.',
                         'First/subsequent refer separately to engine requests and session turns; later sessions share an engine.',
                         'Decode rate is the saved runtime benchmark rate, not inferred from callback count or wall time.',
                         'Engine stage can include pre-call checkpoint overhead; callback timing starts before request setup.',
                         'Repeated requests are not independent process replications; n in metric summaries counts requests.',
                         'Memory runs are excluded from performance groups; stage values are sampled process accounting, not GPU allocation.',
                         'JSON state/marker checks and reaching an output cap are separate from successful stream completion.',
                         'Manifest verification checks archived bytes, not authenticity, model contents, or source/binary equivalence.'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, help='New JSON file; existing files are never overwritten')
    args = parser.parse_args(argv)
    if args.output and any(args.output.resolve().is_relative_to(p.resolve()) for p in args.directories):
        parser.error('--output must be outside input directories')
    result = summarize(args.directories)
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if args.output:
        try:
            with args.output.open('x') as stream:
                stream.write(payload)
        except OSError as exc:
            parser.error(str(exc))
    else:
        print(payload, end='')
    return 0 if result['valid'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
