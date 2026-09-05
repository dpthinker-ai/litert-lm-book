import unittest
from m3_summarize import field, span, describe
from m3_run import thermal_status

class MeasurementSemantics(unittest.TestCase):
    def test_missing_memory_is_not_zero(self):
        self.assertIsNone(field('permission denied', 'VmRSS'))
        self.assertEqual(field('VmRSS:\t0 kB\n', 'VmRSS'),0)
    def test_units_are_not_silently_guessed(self):
        self.assertIsNone(field('VmRSS: 1024 bytes\n','VmRSS'))
    def test_clock_subtraction_stays_in_device_domain(self):
        events=[{'event':'start','ns':5_000_000_000},{'event':'end','ns':5_123_000_000}]
        self.assertEqual(span(events,'start','end'),123)
        self.assertIsNone(span(events,'start','missing'))
    def test_absent_thermal_status_is_not_normal(self):
        self.assertIsNone(thermal_status('adb: device offline'))
        self.assertEqual(thermal_status('Thermal Status: 3\n'),3)
    def test_no_samples_have_no_estimate(self):
        self.assertIsNone(describe([])['median'])

if __name__=='__main__': unittest.main()
