"""Run with python -m unittest discover -s tests -v (NumPy and SciPy).

Executes the notebook's actual definitions and FFT cell, without Colab/UI.
NOTEBOOK_PATH can select an unmodified notebook for regression verification.
"""
import ast
import contextlib
import gc
import io
import json
import os
from pathlib import Path
import re
import tempfile
import time
import unittest
import wave

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.interpolate import interp1d
from scipy.io import wavfile
from scipy.signal import savgol_filter


class WavScalingTests(unittest.TestCase):
    def setUp(self):
        path = Path(os.environ.get('NOTEBOOK_PATH',
            Path(__file__).resolve().parents[1] / 'Frequency_Response_analysis.ipynb'))
        self.cells = json.loads(path.read_text())['cells']
        source = '\n'.join(line for line in ''.join(self.cells[1]['source']).splitlines()
                           if not line.startswith('!'))
        tree = ast.parse(source)
        definitions = ast.Module(body=[node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))], type_ignores=[])
        self.env = dict(np=np, wavfile=wavfile, rfft=rfft, rfftfreq=rfftfreq,
                        interp1d=interp1d, savgol_filter=savgol_filter,
                        re=re, time=time, gc=gc)
        exec(compile(definitions, str(path), 'exec'), self.env)
        self.env['fft_cache'] = self.env['EnhancedCache'](10)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def wav(self, name, samples):
        path = str(Path(self.temp.name) / (name + '.wav'))
        wavfile.write(path, 48000, samples)
        return path

    def load(self, samples):
        rate, data = self.env['load_audio'](self.wav('input', samples))
        self.assertEqual(rate, 48000)
        return data

    def test_signed_integer_full_scale(self):
        for dtype in [np.int16, np.int32, np.int64]:
            with self.subTest(dtype=dtype):
                scale = 2 ** (np.iinfo(dtype).bits - 1)
                samples = np.array([-scale, -scale//2, 0, scale//2, scale-1], dtype=dtype)
                np.testing.assert_array_equal(self.load(samples), samples.astype(float)/scale)

    def test_uint8_midpoint_and_full_scale(self):
        samples = np.array([0, 64, 128, 192, 255], dtype=np.uint8)
        np.testing.assert_array_equal(self.load(samples), [-1, -.5, 0, .5, 127/128])

    def test_float_amplitudes_unchanged_including_above_full_scale(self):
        for dtype in [np.float32, np.float64]:
            with self.subTest(dtype=dtype):
                samples = np.array([-1.5, -.5, 0, .25, 1.25], dtype=dtype)
                actual = self.load(samples)
                np.testing.assert_array_equal(actual, samples)
                self.assertEqual(actual.dtype, samples.dtype)

    def test_stereo_pcm_normalized_before_downmix(self):
        samples = np.array([[-32768, 0], [16384, 32766], [0, 0]], dtype=np.int16)
        np.testing.assert_array_equal(self.load(samples), (samples.astype(float)/32768).mean(axis=1))

    def test_stereo_uint8_midpoint_before_downmix(self):
        samples = np.array([[0, 128], [128, 128], [192, 64]], dtype=np.uint8)
        np.testing.assert_array_equal(self.load(samples), [-.5, 0, 0])

    def test_packed24_left_justified_int32(self):
        # Write actual packed 24-bit RIFF PCM, rather than mocking wavfile.read.
        values = [-8388608, -4194304, 0, 4194304, 8388607]
        path = str(Path(self.temp.name) / 'packed24.wav')
        with wave.open(path, 'wb') as output:
            output.setnchannels(1)
            output.setsampwidth(3)
            output.setframerate(48000)
            output.writeframes(b''.join((v & 0xffffff).to_bytes(3, 'little') for v in values))
        _, decoded = wavfile.read(path)
        self.assertEqual(decoded.dtype, np.dtype('int32'))
        np.testing.assert_array_equal(decoded.astype(np.int64), np.array(values, dtype=np.int64)*256)
        _, actual = self.env['load_audio'](path)
        np.testing.assert_array_equal(actual, np.array(values)/8388608)

    def compare(self, ref, test):
        env = self.env
        env['samplerate_ref'], env['data_ref_unclipped'] = env['load_audio'](ref)
        env['samplerate_test'], env['data_test_unclipped'] = env['load_audio'](test)
        exec(compile(''.join(self.cells[4]['source']), 'actual-main-fft-cell', 'exec'), env)
        response = env['calculate_frequency_response'](env['magnitude_test'], env['magnitude_ref'])
        main = response[(env['freqs_ref'] >= 20) & (env['freqs_ref'] <= 20000)]
        with contextlib.redirect_stdout(io.StringIO()):
            optional = env['process_microphone'](ref, test)
        return float(np.median(main)), float(np.median(optional['original']))

    def signals(self):
        samples = np.random.default_rng(1701).integers(-64, 64, 4096) * 2
        # All samples exactly representable in every encoding; no quantization ambiguity.
        return {
            'uint8': (samples + 128).astype(np.uint8),
            'pcm16': (samples * 256).astype(np.int16),
            'pcm32': (samples * 16777216).astype(np.int32),
            'float32': (samples / 128).astype(np.float32),
        }

    def test_mixed_encodings_identical_waveform_zero_response(self):
        paths = {name: self.wav(name, data) for name, data in self.signals().items()}
        for name, path in paths.items():
            with self.subTest(encoding=name):
                for db in self.compare(paths['pcm16'], path):
                    self.assertAlmostEqual(db, 0, places=6)

    def test_real_gain_difference_preserved_across_encodings(self):
        signals = self.signals()
        ref = self.wav('ref', signals['pcm16'])
        for name, data in [('pcm16', signals['pcm16']//2), ('float32', signals['float32']/2)]:
            with self.subTest(encoding=name):
                for db in self.compare(ref, self.wav(name, data)):
                    self.assertAlmostEqual(db, 20*np.log10(.5), places=6)


if __name__ == '__main__':
    unittest.main()
