import unittest
from m4_audit_tokens import verify_hashes, checked


class AuditIdentityTests(unittest.TestCase):
    def test_replaced_model_and_missing_library_are_rejected(self):
        expected = {'/device/model': 'a' * 64, '/device/lib.so': 'b' * 64}
        for output in ['c' * 64 + '  /device/model\n' + 'b' * 64 + '  /device/lib.so\n',
                       'a' * 64 + '  /device/model\n']:
            with self.assertRaises(ValueError):
                verify_hashes(output, expected)

    def test_verified_artifacts_return_observed_identity(self):
        expected = {'/device/model': 'a' * 64, '/device/lib.so': 'b' * 64}
        output = '\n'.join(digest + '  ' + name for name, digest in expected.items())
        self.assertEqual(verify_hashes(output, expected), expected)

    def test_failed_or_timed_out_command_cannot_continue(self):
        for code in [1, None]:
            with self.assertRaises(RuntimeError):
                checked({'returncode': code, 'name': 'push', 'stderr': 'failure', 'stdout': ''})
