"""Rende un MIDI di pianoforte in audio con un semplice sintetizzatore (armoniche + decadimento).
Non è un vero pianoforte: serve solo a sentire note e ritmo senza programmi esterni.
Uso: python ascolta.py file.mid uscita.wav"""
import sys

import numpy as np
import pretty_midi
import soundfile as sf

SR = 22050


def tone(freq, dur, vel):
    n = int((dur + 0.6) * SR)
    t = np.arange(n) / SR
    decay = np.exp(-t * (1.2 + freq / 400))
    y = sum((0.6 ** k) * np.sin(2 * np.pi * freq * (k + 1) * t) * np.exp(-t * k * 0.8) for k in range(5))
    env = decay * np.minimum(1, t / 0.005)
    rel = np.ones(n)
    r0 = int(dur * SR)
    rel[r0:] = np.exp(-(np.arange(n - r0)) / (0.08 * SR))
    return (vel / 127) * y * env * rel


def render(mid, wav):
    pm = pretty_midi.PrettyMIDI(mid)
    end = pm.get_end_time() + 1
    out = np.zeros(int(end * SR) + SR)
    for inst in pm.instruments:
        for n in inst.notes:
            y = tone(pretty_midi.note_number_to_hz(n.pitch), n.end - n.start, n.velocity)
            s = int(n.start * SR)
            out[s:s + len(y)] += y[: len(out) - s]
    out /= np.abs(out).max() + 1e-9
    sf.write(wav, 0.9 * out, SR)


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2])
