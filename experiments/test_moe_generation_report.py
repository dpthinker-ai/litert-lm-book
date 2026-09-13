"""Read-only regression checks; no native runtime import or model execution."""
import contextlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import moe_generation_report as report


BASE = Path(__file__).resolve().parent / 'data/2026-09-13'
DIRECTORIES = ['moe-full-model-retry', 'moe-context', 'moe-context-2048',
               'moe-context-4096', 'moe-long-output', 'moe-multiturn', 'moe-memory']
SOURCE = BASE / 'moe-full-model-retry/stream-ram-r2'


class ArchivedEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = report.summarize([BASE / d for d in DIRECTORIES])

    def test_complete_archive_counts_and_integrity(self):
        result = self.result
        self.assertTrue(result['valid'], result['errors'])
        self.assertEqual((result['performance_processes'], result['performance_requests']), (15, 60))
        self.assertEqual((result['memory_processes'], result['memory_requests']), (3, 6))
        self.assertTrue(all(a['status'] == 'verified' for a in result['archives']))

    def test_engine_reuse_and_known_timing(self):
        english = [g for g in self.result['groups']
                   if g['group']['conditions']['context_capacity'] == 128]
        self.assertEqual(len(english), 2)
        first = next(g for g in english if g['group']['engine_request'] == 'first')
        reused = next(g for g in english if g['group']['engine_request'] == 'subsequent')
        self.assertEqual((first['processes'], first['requests']), (3, 3))
        self.assertEqual((reused['processes'], reused['requests']), (3, 6))
        self.assertAlmostEqual(first['first_text_ms']['median'], 2612, delta=1)
        self.assertAlmostEqual(reused['first_text_ms']['median'], 120.4, delta=.1)
        self.assertAlmostEqual(reused['runtime_decode_tokens_per_second']['median'], 82.61, delta=.01)

    def test_long_output_does_not_become_quality_success(self):
        groups = [g for g in self.result['groups'] if g['group']['conditions']['max_output_tokens'] in (128, 256)]
        self.assertEqual(sum(g['requests'] for g in groups), 6)
        self.assertEqual(sum(g['reached_output_cap'] for g in groups), 6)
        reused = next(g for g in groups if g['group']['conditions']['max_output_tokens'] == 256
                      and g['group']['engine_request'] == 'subsequent')
        self.assertAlmostEqual(reused['runtime_decode_tokens_per_second']['median'], 51.70, delta=.01)
        self.assertAlmostEqual(reused['first_text_ms']['median'], 171.1, delta=.1)

    def test_multiturn_sessions_are_one_process(self):
        run = next(r for r in self.result['runs'] if r['kind'] == 'multiturn')
        self.assertEqual(len(run['sessions']), 3)
        self.assertEqual([s['engine_first_request'] for s in run['sessions']], [True, False, False])
        self.assertEqual([s['final_conversation_count'] for s in run['sessions']], [2516, 2512, 2509])
        self.assertEqual(sum(r['json_state_matches'] is True for r in run['requests']), 18)
        self.assertEqual(sum(r['engine_request'] == 'first' for r in run['requests']), 1)
        self.assertEqual(sum(r['session_request'] == 'first' for r in run['requests']), 3)

    def test_memory_recomputed_and_not_in_performance(self):
        self.assertEqual(len(self.result['memory_groups']), 1)
        group = self.result['memory_groups'][0]
        self.assertEqual(group['processes'], 3)
        stages = {r['stage']: r for r in group['stages']}
        self.assertAlmostEqual(stages['engine_created']['resident_size_gib']['median'], 5.012, delta=.001)
        self.assertAlmostEqual(stages['long_generation_finished']['phys_footprint_gib']['median'], 3.184, delta=.001)
        self.assertTrue(all(g['group']['kind'] != 'memory' for g in self.result['groups']))
        pressures = {}
        for run in self.result['runs']:
            if run['kind'] == 'memory':
                for key, value in run['pressure_counts'].items():
                    pressures[key] = pressures.get(key, 0) + value
        self.assertEqual(pressures, {'1': 364, '2': 16})


class DamagedEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / 'run'
        shutil.copytree(SOURCE, self.run)

    def rewrite(self, worker_change=None, report_change=None, log_append=None):
        original = report.read_json(self.run / 'report.json')
        worker = report.read_json(self.run / 'worker.json')
        if worker_change:
            worker_change(worker)
        (self.run / 'worker.json').write_text(json.dumps(worker))
        original['worker_result'] = worker
        if log_append:
            with (self.run / 'stderr.log').open('a') as stream:
                stream.write(log_append)
        original['files'] = {name: report.sha(self.run / name) for name in original['files']}
        if report_change:
            report_change(original)
        (self.run / 'report.json').write_text(json.dumps(original))

    def assert_excluded(self, contains):
        result = report.summarize([self.run])
        self.assertFalse(result['valid'])
        self.assertEqual(result['included_processes'], 0)
        self.assertIn(contains, result['errors'][0]['error'])

    def test_new_run_does_not_claim_archive_manifest_verification(self):
        result = report.summarize([self.run])
        self.assertTrue(result['valid'])
        self.assertEqual(result['archives'][0]['status'], 'not_present')
        self.assertEqual(result['runs'][0]['run_file_hashes'], 'verified')

    def test_memory_workloads_do_not_mix(self):
        # Keep process configuration equal except for one input characteristic.
        for changed in ('budget', 'prompt', 'prefill'):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                for name in ('run1', 'run2'):
                    shutil.copytree(BASE / 'moe-memory' / name, root / name)
                directory = root / 'run2'
                saved = report.read_json(directory / 'report.json')
                worker = report.read_json(directory / 'worker.json')
                if changed == 'budget':
                    saved['input_raw_token_budget'] += 1
                elif changed == 'prompt':
                    worker['generations'][1]['prompt'] += '\nDifferent input.'
                else:
                    worker['generations'][1]['benchmark']['last_prefill_token_count'] += 1
                (directory / 'worker.json').write_text(json.dumps(worker))
                saved['worker_result'] = worker
                saved['files']['worker.json'] = report.sha(directory / 'worker.json')
                (directory / 'report.json').write_text(json.dumps(saved))
                result = report.summarize([root])
                self.assertTrue(result['valid'], result['errors'])
                self.assertEqual(len(result['memory_groups']), 2)
                self.assertEqual([g['processes'] for g in result['memory_groups']], [1, 1])
                self.assertTrue(all('workload' in g and 'input_raw_token_budget' in g['conditions']
                                    for g in result['memory_groups']))

    def test_manifest_detects_modified_file(self):
        manifest = {'files': {str(p.relative_to(self.root)): report.sha(p)
                              for p in self.run.iterdir() if p.is_file()}}
        (self.root / 'manifest.json').write_text(json.dumps(manifest))
        (self.run / 'stderr.log').write_text('changed')
        result = report.summarize([self.root])
        self.assertFalse(result['valid'])
        self.assertIn('SHA-256 mismatch', result['errors'][0]['error'])

    def test_missing_worker_is_not_skipped_as_success(self):
        (self.run / 'worker.json').unlink()
        self.assert_excluded('worker.json')

    def test_required_metric_missing(self):
        self.rewrite(lambda w: w['generations'][0]['benchmark'].pop('last_decode_tokens_per_second'))
        self.assert_excluded('runtime decode rate')

    def test_failure_is_excluded_even_with_a_response(self):
        self.rewrite(report_change=lambda r: r.update(status='STOPPED_BY_GUARD', stop_reason='timeout'))
        self.assert_excluded('failed, incomplete, or stopped')

    def test_backend_error_overrides_success_status(self):
        self.rewrite(log_append='\nValidation error: invalid command buffer\n')
        self.assert_excluded('backend error in native log')

    def test_final_event_is_required_independently_of_saved_flags(self):
        self.rewrite(lambda w: w['generations'][0]['events'][-1].update(is_final=False))
        self.assert_excluded('final event missing')

    def test_saved_timestamp_cannot_replace_raw_event_timestamp(self):
        self.rewrite(lambda w: w['generations'][0].update(first_text_callback_seconds=999))
        self.assert_excluded('first text timestamp mismatch')

    def test_report_worker_disagreement(self):
        worker = report.read_json(self.run / 'worker.json')
        worker['phase'] = 'incomplete'
        (self.run / 'worker.json').write_text(json.dumps(worker))
        self.assert_excluded('report and worker.json disagree')

    def test_overlapping_inputs_and_copies_not_double_counted(self):
        shutil.copytree(self.run, self.root / 'copy')
        result = report.summarize([self.root, self.run])
        self.assertTrue(result['valid'])
        self.assertEqual(result['performance_requests'], 3)
        self.assertEqual(len(result['duplicate_runs_excluded']), 2)

    def test_existing_summary_is_not_measurement_input(self):
        (self.run / 'summary.json').write_text('{"requests":999999}')
        result = report.summarize([self.run])
        self.assertTrue(result['valid'])
        self.assertEqual(result['performance_requests'], 3)

    def test_output_is_new_and_outside_inputs(self):
        output = self.root / 'new-summary.json'
        self.assertEqual(report.main([str(self.run), '--output', str(output)]), 0)
        initial = output.read_bytes()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            report.main([str(self.run), '--output', str(output)])
        self.assertEqual(output.read_bytes(), initial)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            report.main([str(self.run), '--output', str(self.run / 'new.json')])
        self.assertFalse((self.run / 'new.json').exists())

    def test_cli_emits_json_and_nonzero_for_invalid_record(self):
        (self.run / 'worker.json').unlink()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(report.main([str(self.run)]), 2)
        self.assertFalse(json.loads(output.getvalue())['valid'])


if __name__ == '__main__':
    unittest.main()
