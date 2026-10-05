"""Offline actual-notebook regressions. Run: python -m unittest discover -s tests

Requires numpy and scipy, as does the notebook. No Colab, microphone or network.
"""
import ast
import contextlib
import gc
import io
import json
from pathlib import Path
import re
import tempfile
import time
import unittest

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.interpolate import interp1d
from scipy.io import wavfile
from scipy.signal import savgol_filter

NOTEBOOK = Path(__file__).resolve().parents[1] / 'Frequency_Response_analysis.ipynb'


class RecordingAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.nb = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
        self.ns = {name: globals()[name] for name in (
            'np', 'rfft', 'rfftfreq', 'interp1d', 'wavfile', 'savgol_filter',
            'gc', 're', 'time')}
        source = '\n'.join(line for line in ''.join(self.nb['cells'][1]['source']).splitlines()
                           if not line.startswith('!'))
        # Execute unchanged notebook definitions, omitting Colab/install/UI imports.
        for node in ast.parse(source).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(NOTEBOOK), 'exec'), self.ns)
        self.ns['fft_cache'] = self.ns['EnhancedCache'](10)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data = np.random.default_rng(7).normal(size=8192).astype(np.float32)

    def process(self, ref, test, ref_rate=48000, test_rate=48000):
        root = Path(self.temp.name)
        wavfile.write(root / 'ref.wav', ref_rate, ref)
        wavfile.write(root / 'test.wav', test_rate, test)
        with contextlib.redirect_stdout(io.StringIO()):
            return self.ns['process_microphone'](str(root / 'ref.wav'), str(root / 'test.wav'))

    def main_cell(self, ref, test, ref_rate=48000, test_rate=48000):
        self.ns.update(data_ref_unclipped=ref, data_test_unclipped=test,
                       samplerate_ref=ref_rate, samplerate_test=test_rate)
        exec(''.join(self.nb['cells'][4]['source']), self.ns)
        return self.ns['calculate_frequency_response'](
            self.ns['normalize_for_pink_noise'](self.ns['freqs_test'], self.ns['magnitude_test']),
            self.ns['normalize_for_pink_noise'](self.ns['freqs_ref'], self.ns['magnitude_ref']))

    def assert_flat(self, result, expected=0):
        for key in ('original', 'medium', 'heavy'):
            np.testing.assert_allclose(result[key], expected, atol=1e-5)
        self.assertTrue(np.all(np.diff(result['freqs']) > 0))

    def test_processor_same_length_control(self):
        self.assert_flat(self.process(self.data, self.data))

    def test_processor_shorter_test(self):
        self.assert_flat(self.process(self.data, self.data[:4096]))

    def test_processor_longer_test(self):
        self.assert_flat(self.process(self.data[:4096], self.data))

    def test_processor_unequal_lengths_gain_preserved(self):
        self.assert_flat(self.process(self.data, self.data[:4096] * 2), 20*np.log10(2))

    def test_processor_rejects_different_rates_before_fft(self):
        def unexpected(*args, **kwargs):
            self.fail('FFT executed before sample-rate validation')
        self.ns['optimized_fft_analysis'] = unexpected
        with self.assertRaisesRegex(ValueError, 'same sample rate'):
            self.process(self.data, self.data, test_rate=44100)

    def test_processor_reverse_rate_mismatch(self):
        with self.assertRaisesRegex(ValueError, 'same sample rate'):
            self.process(self.data, self.data, ref_rate=44100)

    def test_processor_cache_separates_sample_rates(self):
        self.process(self.data, self.data, ref_rate=48000, test_rate=48000)
        result = self.process(self.data, self.data, ref_rate=32000, test_rate=32000)
        self.assert_flat(result)
        self.assertLessEqual(result['freqs'][-1], 16000)
        self.assertEqual(len(self.ns['fft_cache'].cache), 4)

    def test_processor_cache_separates_truncated_lengths(self):
        self.process(self.data, self.data)
        result = self.process(self.data, self.data[:4096])
        self.assert_flat(result)
        self.assertEqual(len(self.ns['fft_cache'].cache), 4)

    def test_main_same_rate_control(self):
        response = self.main_cell(self.data, self.data)
        np.testing.assert_allclose(response[1:], 0, atol=1e-5)

    def test_main_shorter_test_control(self):
        response = self.main_cell(self.data, self.data[:4096])
        np.testing.assert_allclose(response[1:], 0, atol=1e-5)

    def test_main_longer_test_control(self):
        response = self.main_cell(self.data[:4096], self.data)
        np.testing.assert_allclose(response[1:], 0, atol=1e-5)

    def test_main_rejects_different_rates_before_fft(self):
        def unexpected(*args, **kwargs):
            self.fail('FFT executed before sample-rate validation')
        self.ns['fft_analysis'] = unexpected
        with self.assertRaisesRegex(ValueError, 'same sample rate'):
            self.main_cell(self.data, self.data, test_rate=44100)


if __name__ == '__main__':
    unittest.main()
