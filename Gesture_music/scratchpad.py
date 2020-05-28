#!/usr/bin/env python3
"""Play a sine signal."""
import argparse
import sys

import numpy as np
import sounddevice as sd


start_idx = 0
frequency=1000
amplitude = 1


samplerate = sd.query_devices(sd.default.device, 'output')['default_samplerate']
def callback(outdata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    global start_idx
    t = (start_idx + np.arange(frames)) / samplerate
    t = t.reshape(-1, 1)
    outdata[:] = amplitude * np.sin(2 * np.pi * frequency * t)
    start_idx += frames

sd.OutputStream(device=sd.default.device, channels=1, callback=callback,
                     samplerate=samplerate)
