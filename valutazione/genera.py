"""Genera brani sintetici con risposta nota (accordi, melodia, basso, tempo, tonalità).

Ogni brano ha voce e base separate (con un po' di "sporco" incrociato, come dopo una vera
separazione), più il mix. Serve a misurare la precisione dell'analisi in modo ripetibile.

Uso: python valutazione/genera.py <cartella> [numero_brani] [seed]
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 22050
NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
MAJ = [0, 2, 4, 5, 7, 9, 11]
MIN = [0, 2, 3, 5, 7, 8, 10]
# giri tipici (gradi 1-7) per modo maggiore e minore
GIRI_MAJ = [[1, 5, 6, 4], [1, 6, 4, 5], [6, 4, 1, 5], [1, 4, 5, 4], [1, 5, 4, 4], [4, 1, 5, 6], [1, 3, 4, 5], [2, 5, 1, 6]]
GIRI_MIN = [[1, 6, 3, 7], [1, 4, 7, 3], [1, 6, 7, 1], [6, 7, 1, 1], [1, 4, 5, 1], [1, 7, 6, 7]]


def diatonic_chord(tonic, mode, degree):
    scale = MAJ if mode == "maggiore" else MIN
    r = (tonic + scale[degree - 1]) % 12
    third = (scale[(degree + 1) % 7] - scale[degree - 1]) % 12
    q = "m" if third == 3 else ""
    if mode == "minore" and degree == 5 and np.random.rand() < 0.5:
        q = ""  # dominante maggiore
    if mode == "maggiore" and degree == 7:
        q = "m"  # semplifica il diminuito
    return r, q


def tone(freq, dur, kind, vel=0.6):
    n = int(dur * SR)
    t = np.arange(n) / SR
    if kind == "piano":
        y = sum((0.55 ** k) * np.sin(2 * np.pi * freq * (k + 1) * t) for k in range(6))
        env = np.exp(-t * 2.5) * np.minimum(1, t / 0.004)
    elif kind == "basso":
        y = np.sin(2 * np.pi * freq * t) + 0.4 * np.sin(4 * np.pi * freq * t) + 0.15 * np.sin(6 * np.pi * freq * t)
        env = np.exp(-t * 1.5) * np.minimum(1, t / 0.01)
    elif kind == "pad":
        y = sum((0.5 ** k) * np.sin(2 * np.pi * freq * (k + 1) * t + k) for k in range(4))
        env = np.minimum(1, t / 0.15)
    env = env * np.minimum(1, (n - np.arange(n)) / (0.02 * SR))
    return vel * y * env


def voice(f_path, amp_path):
    """Voce sintetica: armoniche con formanti approssimate, da una curva di frequenza."""
    phase = 2 * np.pi * np.cumsum(f_path) / SR
    y = np.zeros_like(f_path)
    for k in range(1, 12):
        fk = k * f_path
        # formanti approssimate di una "a" (700, 1200, 2600 Hz)
        g = sum(np.exp(-((fk - F) / B) ** 2) for F, B in ((700, 200), (1200, 250), (2600, 400))) + 0.08
        y += g * np.sin(k * phase) / k ** 0.5
    y += 0.01 * np.random.randn(len(y))  # respiro
    return y * amp_path


def drums(n_beats, spb, n):
    y = np.zeros(n)
    for b in range(n_beats):
        s = int(b * spb * SR)
        if b % 2 == 0:  # cassa
            t = np.arange(int(0.15 * SR)) / SR
            k = np.sin(2 * np.pi * (50 + 80 * np.exp(-t * 30)) * t) * np.exp(-t * 20)
            y[s:s + len(k)] += 0.9 * k[: n - s]
        else:  # rullante
            t = np.arange(int(0.12 * SR)) / SR
            k = np.random.randn(len(t)) * np.exp(-t * 25)
            y[s:s + len(k)] += 0.35 * k[: n - s]
        for h in (0, 0.5):  # charleston in crome
            hs = int((b + h) * spb * SR)
            t = np.arange(int(0.03 * SR)) / SR
            k = np.diff(np.random.randn(len(t) + 1)) * np.exp(-t * 120)
            y[hs:hs + len(k)] += 0.12 * k[: max(0, n - hs)]
    return y


def make_song(i, out, rng):
    tonic = int(rng.integers(12))
    mode = "maggiore" if rng.random() < 0.65 else "minore"
    bpm = float(rng.uniform(85, 150))
    spb = 60 / bpm
    giro = (GIRI_MAJ if mode == "maggiore" else GIRI_MIN)[rng.integers(len(GIRI_MAJ if mode == "maggiore" else GIRI_MIN))]
    n_bars = 32
    beats_per_chord = 4 if rng.random() < 0.6 else 2
    lead_in = 1.0  # secondi di silenzio iniziale
    chords = []
    b = 0
    while b < n_bars * 4:
        deg = giro[(b // beats_per_chord) % len(giro)]
        r, q = diatonic_chord(tonic, mode, deg)
        sev = rng.random() < 0.25  # alcune settime nel voicing (la risposta resta la triade)
        chords.append({"root": r, "q": q, "settima": bool(sev), "beat": b, "beats": beats_per_chord})
        b += beats_per_chord
    total = lead_in + n_bars * 4 * spb + 2
    n = int(total * SR)
    instr = np.zeros(n)
    scale = MAJ if mode == "maggiore" else MIN
    # accompagnamento: accordi (piano o pad), basso
    comp = "piano" if rng.random() < 0.6 else "pad"
    bass_notes = []
    for c in chords:
        t0 = lead_in + c["beat"] * spb
        third = 3 if c["q"] == "m" else 4
        ints = [0, third, 7] + ([10 if c["q"] == "m" or rng.random() < 0.7 else 11] if c["settima"] else [])
        base = 55 + ((c["root"] - 7) % 12)  # voicing tra Sol3 e Fa#4
        inv = rng.integers(3)
        pitches = sorted(base + iv + (12 if j < inv else 0) for j, iv in enumerate(ints))
        hits = [(k, 1.0) for k in range(c["beats"])] if comp == "piano" else [(0, c["beats"])]
        for k, d in hits:
            for p in pitches:
                y = tone(440 * 2 ** ((p - 69) / 12), d * spb, comp, 0.18)
                s = int((t0 + k * spb) * SR)
                instr[s:s + len(y)] += y[: n - s]
        root = 36 + (c["root"] - 0) % 12
        if root > 45:
            root -= 12
        for k in range(c["beats"]):
            p = root + (7 if (k % 2 == 1 and rng.random() < 0.3) else 0)
            y = tone(440 * 2 ** ((p - 69) / 12), spb * 0.9, "basso", 0.5)
            s = int((t0 + k * spb) * SR)
            instr[s:s + len(y)] += y[: n - s]
            bass_notes.append({"pitch": int(p), "start": t0 + k * spb, "end": t0 + (k + 0.9) * spb})
    dr = drums(n_bars * 4, spb, n - int(lead_in * SR))
    instr[int(lead_in * SR):] += dr
    # melodia: note dell'accordo sui tempi forti, note di passaggio della scala
    center = 64 if rng.random() < 0.5 else 57  # voce femminile o maschile
    mel = []
    cur = center + scale[0] - (center % 12) + tonic
    while cur > center + 6:
        cur -= 12
    pos = 0.0
    while pos < n_bars * 4 - 1:
        if rng.random() < 0.15:
            pos += rng.choice([0.5, 1.0, 2.0])  # pausa
            continue
        dur = float(rng.choice([0.5, 0.5, 1.0, 1.0, 1.5, 2.0]))
        c = next(c for c in chords if c["beat"] <= pos < c["beat"] + c["beats"])
        tones = [(c["root"] + iv) % 12 for iv in (0, 3 if c["q"] == "m" else 4, 7)]
        pool = tones if pos % 1 == 0 else [(tonic + s) % 12 for s in scale]
        cands = [p for p in range(cur - 5, cur + 6) if p % 12 in pool and center - 7 <= p <= center + 12]
        if not cands:
            cands = [cur]
        cur = int(rng.choice(cands))
        mel.append({"pitch": cur, "start": lead_in + pos * spb, "end": lead_in + (pos + dur) * spb})
        pos += dur
    f = np.full(n, 200.0)
    amp = np.zeros(n)
    t_all = np.arange(n) / SR
    vib = 1 + 0.015 * np.sin(2 * np.pi * 5.5 * t_all)
    for j, m in enumerate(mel):
        s, e = int(m["start"] * SR), int((m["end"] - 0.02) * SR)
        hz = 440 * 2 ** ((m["pitch"] - 69) / 12)
        f[s:e] = hz
        # attacco con portamento dalla nota precedente (60 ms)
        if j and mel[j - 1]["end"] >= m["start"] - 0.01:
            prev = 440 * 2 ** ((mel[j - 1]["pitch"] - 69) / 12)
            k = min(int(0.06 * SR), e - s)
            f[s:s + k] = np.linspace(prev, hz, k)
        L = e - s
        a = np.minimum(1, np.arange(L) / (0.03 * SR)) * np.minimum(1, (L - np.arange(L)) / (0.04 * SR))
        amp[s:e] = 0.25 * a
    voc = voice(f * vib, amp)
    # separazione imperfetta: un po' di base nella voce e viceversa
    voc_stem = voc + 0.06 * instr
    ins_stem = instr + 0.04 * voc
    mix = voc + instr
    norm = np.abs(mix).max() + 1e-9
    name = f"sint{i:02d}"
    (out / "input").mkdir(parents=True, exist_ok=True)
    (out / "stems").mkdir(parents=True, exist_ok=True)
    (out / "verita").mkdir(parents=True, exist_ok=True)
    wav = out / "input" / f"{name}.wav"
    sf.write(wav, 0.9 * mix / norm, SR)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav), "-b:a", "192k", str(out / "input" / f"{name}.mp3")], check=True)
    wav.unlink()
    sf.write(out / "stems" / f"{name}_vocals.wav", 0.9 * voc_stem / norm, SR)
    sf.write(out / "stems" / f"{name}_instrumental.wav", 0.9 * ins_stem / norm, SR)
    truth = {
        "brano": name, "tonica": NOTE_NAMES[tonic], "modo": mode, "tempo_bpm": bpm, "lead_in": lead_in,
        "accordi": [{"accordo": NOTE_NAMES[c["root"]] + c["q"], "root_pc": c["root"], "qualita": c["q"],
                     "start": lead_in + c["beat"] * spb, "end": lead_in + (c["beat"] + c["beats"]) * spb} for c in chords],
        "melodia": mel, "basso": bass_notes,
    }
    (out / "verita" / f"{name}.json").write_text(json.dumps(truth, indent=1))
    return name


if __name__ == "__main__":
    out = Path(sys.argv[1])
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    rng = np.random.default_rng(int(sys.argv[3]) if len(sys.argv) > 3 else 7)
    np.random.seed(int(sys.argv[3]) if len(sys.argv) > 3 else 7)
    for i in range(k):
        print(make_song(i, out, rng))
