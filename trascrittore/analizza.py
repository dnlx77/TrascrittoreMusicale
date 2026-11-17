"""Passo 1: analisi di un brano separato (voce + strumentale).

Uso: python analizza.py <nome_brano>
Legge data/input/<nome>.mp3 e data/stems/<nome>_{vocals,instrumental}.wav
Scrive in data/analisi/<nome>/: skeleton.json, analisi.mid, verifica.wav
"""
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
import scipy.signal as sps
import soundfile as sf

SR = 22050
HOP = 256
METODO_MELODIA = "pyin"  # "crepe" (più preciso) o "pyin" (più veloce, nessun modello)
NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

# Profili Krumhansl-Kessler per la tonalità
KK_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def load(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y


def beats_and_downbeats(mix, bass_audio):
    tempo, beats = librosa.beat.beat_track(y=mix, sr=SR, hop_length=HOP, trim=False)
    beat_t = librosa.frames_to_time(beats, sr=SR, hop_length=HOP)
    # Fase della battuta (4/4): la fase in cui il basso attacca più forte
    onset = librosa.onset.onset_strength(y=bass_audio, sr=SR, hop_length=HOP)
    strength = onset[np.clip(beats, 0, len(onset) - 1)]
    phase = int(np.argmax([strength[p::4].mean() for p in range(4)]))
    return float(np.atleast_1d(tempo)[0]), beat_t, phase


def estimate_key(chroma):
    prof = chroma.sum(axis=1)
    best = None
    for tonic in range(12):
        for mode, kk in (("maggiore", KK_MAJ), ("minore", KK_MIN)):
            r = np.corrcoef(prof, np.roll(kk, tonic))[0, 1]
            if best is None or r > best[0]:
                best = (r, tonic, mode)
    return {"tonica": NOTE_NAMES[best[1]], "modo": best[2], "pc": best[1], "confidenza": round(float(best[0]), 3)}


def f0_crepe(y, fmin, fmax, n_frames):
    """Intonazione con CREPE (rete neurale, più robusta di pYIN sulla voce cantata)."""
    import torch
    import torchcrepe

    y16 = librosa.resample(y, orig_sr=SR, target_sr=16000)
    audio = torch.tensor(y16, dtype=torch.float32)[None]
    pitch, per = torchcrepe.predict(audio, 16000, hop_length=160, fmin=fmin, fmax=fmax, model="full",
                                    batch_size=512, device="cpu", return_periodicity=True)
    per = torchcrepe.filter.median(per, 3)[0].numpy()
    pitch = pitch[0].numpy()
    t_crepe = np.arange(len(pitch)) * 0.01
    t = librosa.frames_to_time(np.arange(n_frames), sr=SR, hop_length=HOP)
    f0 = np.interp(t, t_crepe, pitch)
    p = np.interp(t, t_crepe, per)
    return f0, p


def pitch_track_notes(y, fmin, fmax, min_dur, metodo="pyin", split_onsets=False):
    rms_all = librosa.feature.rms(y=y, hop_length=HOP)[0]
    if metodo == "crepe":
        f0, per = f0_crepe(y, fmin, fmax, len(rms_all))
        loud = rms_all > 0.15 * np.percentile(rms_all[rms_all > 0], 90)
        voiced = (per > 0.4) & loud
    else:
        f0, voiced, prob = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=SR, hop_length=HOP, frame_length=2048)
        n = min(len(f0), len(rms_all))
        f0, voiced = f0[:n], voiced[:n]
    midi = np.where(voiced, librosa.hz_to_midi(np.nan_to_num(f0, nan=1.0)), np.nan)
    # Smussa le piccole oscillazioni e arrotonda al semitono
    m = np.round(sps.medfilt(np.nan_to_num(midi, nan=0.0), 5))
    m[~voiced] = 0
    split = set()
    if split_onsets:
        # note ripetute (stessa altezza, nuova sillaba): separa dove la voce ha un attacco
        ons = librosa.onset.onset_detect(y=y, sr=SR, hop_length=HOP, backtrack=False, delta=0.4)
        k = 8  # ~90 ms: entrambe le metà devono essere note vere, non rumore
        for o in ons:
            if k <= o < len(m) - k and m[o] > 0 and np.all(m[o - k:o + k] == m[o]):
                m[o - 1] = 0
                split.add(int(o))
    times = librosa.frames_to_time(np.arange(len(m)), sr=SR, hop_length=HOP)
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    rms = rms[: len(m)]
    notes, start, cur = [], None, 0
    for i, v in enumerate(np.append(m, 0)):
        if v != cur:
            if cur > 0 and start is not None:
                t0, t1 = times[start], times[min(i, len(times) - 1)]
                if t1 - t0 >= min_dur:
                    vel = float(np.mean(rms[start:i])) if i > start else 0.0
                    notes.append({"pitch": int(cur), "start": float(t0), "end": float(t1), "energia": vel, "_i": start})
            start, cur = i, v
    # Unisci note uguali separate da micro-pause
    merged = []
    for n in notes:
        if merged and merged[-1]["pitch"] == n["pitch"] and n["start"] - merged[-1]["end"] < 0.05 and n["_i"] not in split:
            merged[-1]["end"] = n["end"]
        else:
            merged.append(n)
    for n in merged:
        n.pop("_i", None)
    return merged


def chord_templates():
    names, temps = [], []
    for r in range(12):
        for q, ints in (("", (0, 4, 7)), ("m", (0, 3, 7))):
            t = np.zeros(12)
            t[[(r + i) % 12 for i in ints]] = 1
            names.append(NOTE_NAMES[r] + q)
            temps.append(t / np.linalg.norm(t))
    return names, np.array(temps)


def estimate_chords(instr, beat_t, duration):
    harm = librosa.effects.harmonic(instr)
    chroma = librosa.feature.chroma_cqt(y=harm, sr=SR, hop_length=HOP, bins_per_octave=36)
    bounds = np.concatenate([[0.0], beat_t, [duration]])
    frames = librosa.time_to_frames(bounds, sr=SR, hop_length=HOP)
    frames = np.clip(frames, 0, chroma.shape[1])
    sync = np.array([chroma[:, a:b].mean(axis=1) if b > a else np.zeros(12) for a, b in zip(frames[:-1], frames[1:])]).T
    names, temps = chord_templates()
    # energia per beat: i beat quasi silenziosi diventano "N" (nessun accordo)
    rms = librosa.feature.rms(y=harm, hop_length=HOP)[0]
    beat_rms = np.array([rms[a:b].mean() if b > a else 0.0 for a, b in zip(frames[:-1], frames[1:])])
    silent = beat_rms < 0.1 * np.median(beat_rms)
    sync = np.log1p(20 * sync)  # compressione: evita che poche note forti dominino
    sync = sync - sync.mean(axis=0, keepdims=True) * 0.5
    norm = sync / (np.linalg.norm(sync, axis=0, keepdims=True) + 1e-9)
    sim = temps @ norm  # (accordi, beat)
    prob = np.exp(sim * 12)
    prob /= prob.sum(axis=0, keepdims=True)
    # Viterbi: gli accordi tendono a restare per più beat
    n = len(names)
    trans = np.full((n, n), (1 - 0.9) / (n - 1))
    np.fill_diagonal(trans, 0.9)
    path = librosa.sequence.viterbi_discriminative(prob, trans)
    chords = []
    names = names + ["N"]
    path = np.where(silent, len(names) - 1, path)
    for i, c in enumerate(path):
        t0, t1 = float(bounds[i]), float(bounds[i + 1])
        if chords and chords[-1]["accordo"] == names[c]:
            chords[-1]["end"] = t1
        else:
            chords.append({"accordo": names[c], "start": t0, "end": t1})
    return chords, chroma


def quantize(notes, beat_t, subdiv=4):
    """Allinea le note alla griglia di sedicesimi derivata dai beat."""
    grid = []
    for a, b in zip(beat_t[:-1], beat_t[1:]):
        grid.extend(np.linspace(a, b, subdiv, endpoint=False))
    grid.append(beat_t[-1])
    grid = np.array(grid)
    out = []
    for n in notes:
        s = int(np.argmin(np.abs(grid - n["start"])))
        e = int(np.argmin(np.abs(grid - n["end"])))
        if e <= s:
            e = s + 1
        out.append({**n, "step_start": s, "step_end": e, "beat": s / subdiv, "durata_beat": (e - s) / subdiv})
    # elimina sovrapposizioni dopo la quantizzazione (linea monofonica)
    clean = []
    for n in out:
        if clean and n["step_start"] < clean[-1]["step_end"]:
            if n["step_start"] == clean[-1]["step_start"]:
                continue
            clean[-1]["step_end"] = n["step_start"]
            clean[-1]["durata_beat"] = (clean[-1]["step_end"] - clean[-1]["step_start"]) / subdiv
        clean.append(n)
    return clean


def main(name, root=None):
    root = Path(root) if root else Path(__file__).parent / "data"
    out = root / "analisi" / name
    out.mkdir(parents=True, exist_ok=True)
    mix = load(root / "input" / f"{name}.mp3")
    voc = load(root / "stems" / f"{name}_vocals.wav")
    instr = load(root / "stems" / f"{name}_instrumental.wav")
    duration = len(mix) / SR

    b, a = sps.butter(4, 250 / (SR / 2), btype="low")
    bass_audio = sps.filtfilt(b, a, instr).astype(np.float32)

    tempo, beat_t, phase = beats_and_downbeats(mix, bass_audio)
    chords, chroma = estimate_chords(instr, beat_t, duration)
    key = estimate_key(chroma)

    melody = pitch_track_notes(voc, librosa.note_to_hz("E2"), librosa.note_to_hz("C6"), 0.08,
                               metodo=METODO_MELODIA, split_onsets=True)
    # scarta note deboli (respiri, rientri dagli strumenti)
    if melody:
        thr = np.percentile([n["energia"] for n in melody], 15)
        melody = [n for n in melody if n["energia"] >= thr]
    bass = pitch_track_notes(bass_audio, librosa.note_to_hz("E1"), librosa.note_to_hz("G3"), 0.1)

    mel_q = quantize(melody, beat_t)
    bass_q = quantize(bass, beat_t)

    skeleton = {
        "brano": name,
        "durata_s": round(duration, 2),
        "tempo_bpm": round(tempo, 1),
        "metro": "4/4 (ipotesi)",
        "primo_downbeat_beat": phase,
        "tonalita": key,
        "beat_times": [round(float(t), 3) for t in beat_t],
        "accordi": chords,
        "melodia": mel_q,
        "basso": bass_q,
    }
    (out / "skeleton.json").write_text(json.dumps(skeleton, indent=1, ensure_ascii=False))

    # MIDI di verifica: melodia, basso, accordi (tempi reali, non quantizzati)
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    for label, notes, prog in (("Melodia", melody, 0), ("Basso", bass, 33)):
        inst = pretty_midi.Instrument(program=prog, name=label)
        for n in notes:
            inst.notes.append(pretty_midi.Note(90, n["pitch"], n["start"], n["end"]))
        pm.instruments.append(inst)
    ch = pretty_midi.Instrument(program=48, name="Accordi")
    for c in chords:
        if c["accordo"] == "N":
            continue
        root_pc = NOTE_NAMES.index(c["accordo"].rstrip("m"))
        third = 3 if c["accordo"].endswith("m") else 4
        for iv in (0, third, 7):
            ch.notes.append(pretty_midi.Note(55, 60 + (root_pc + iv) % 12, c["start"], c["end"]))
    pm.instruments.append(ch)
    pm.write(str(out / "analisi.mid"))

    # Audio di verifica: originale basso volume (sx) + sintesi (dx) per confronto
    synth = pm.synthesize(fs=SR)
    L = max(len(synth), len(mix))
    synth = np.pad(synth, (0, L - len(synth)))
    m = np.pad(mix, (0, L - len(mix)))
    synth /= np.abs(synth).max() + 1e-9
    m /= np.abs(m).max() + 1e-9
    stereo = np.stack([0.35 * m + 0.5 * synth, 0.35 * m + 0.5 * synth], axis=1)
    sf.write(out / "verifica.wav", stereo, SR)

    print(json.dumps({
        "brano": name, "tempo": skeleton["tempo_bpm"], "tonalita": f"{key['tonica']} {key['modo']} ({key['confidenza']})",
        "note_melodia": len(mel_q), "note_basso": len(bass_q), "cambi_accordo": len(chords),
        "accordi_piu_frequenti": sorted({c["accordo"] for c in chords}, key=lambda a: -sum(c["end"] - c["start"] for c in chords if c["accordo"] == a))[:6],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1])
