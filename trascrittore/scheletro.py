"""Passo 2: dallo scheletro grezzo (passo 1) a uno scheletro musicale pulito.

Uso: python scheletro.py <nome_brano>
Legge data/analisi/<nome>/skeleton.json + audio, scrive in data/scheletro/<nome>/:
  scheletro.json, lead_sheet.musicxml, lead_sheet.pdf, scheletro.mid, verifica.wav
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
import scipy.signal as sps
import soundfile as sf

from .analizza import HOP, NOTE_NAMES, SR, load

FLAT_KEYS = {"F", "Bb", "Eb", "Ab", "C#", "F#"}  # tonalità maggiori con bemolle (C#/F# -> Db/Gb)
MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]
SMUFL_TEXT = {"\uea64": "\u266d", "\uea66": "\u266f", "\uea65": "\u266e", "\ueca5": "\u2669"}


# ---------------------------------------------------------------- tempo e battute
def fix_beats(beat_t, tempo):
    """Porta il tempo nell'intervallo 90-180 BPM raddoppiando/dimezzando i beat."""
    beat_t = np.asarray(beat_t)
    if tempo < 90:
        mids = (beat_t[:-1] + beat_t[1:]) / 2
        beat_t = np.sort(np.concatenate([beat_t, mids]))
        tempo *= 2
    elif tempo > 180:
        beat_t = beat_t[::2]
        tempo /= 2
    return beat_t, tempo


def downbeat_phase(beat_t, mix, bass_audio):
    """Sceglie su quale beat cadono i primi tempi (4/4): basso + accenti del mix."""
    on_b = librosa.onset.onset_strength(y=bass_audio, sr=SR, hop_length=HOP)
    on_m = librosa.onset.onset_strength(y=mix, sr=SR, hop_length=HOP)
    fr = np.clip(librosa.time_to_frames(beat_t, sr=SR, hop_length=HOP), 0, len(on_b) - 1)
    s = on_b[fr] / (on_b.max() + 1e-9) + 0.5 * on_m[fr] / (on_m.max() + 1e-9)
    return int(np.argmax([s[p::4].mean() for p in range(4)]))


def time_to_beat(t, beat_t):
    """Tempo in secondi -> posizione in beat (frazionaria), interpolando i beat."""
    idx = np.arange(len(beat_t))
    period = np.median(np.diff(beat_t))
    if t < beat_t[0]:
        return (t - beat_t[0]) / period
    if t > beat_t[-1]:
        return idx[-1] + (t - beat_t[-1]) / period
    return float(np.interp(t, beat_t, idx))


# ---------------------------------------------------------------- accordi
def chord_vocab(key):
    tonic = NOTE_NAMES.index(key["tonica"])
    scale = MAJOR_SCALE if key["modo"] == "maggiore" else MINOR_SCALE
    scale_pcs = {(tonic + s) % 12 for s in scale}
    names, temps, prior, roots, quals = [], [], [], [], []
    for r in range(12):
        for q, ints in (("", (0, 4, 7)), ("m", (0, 3, 7))):
            pcs = [(r + i) % 12 for i in ints]
            t = np.zeros(12)
            t[pcs] = [1.0, 0.8, 0.9]
            names.append(NOTE_NAMES[r] + q)
            temps.append(t / np.linalg.norm(t))
            diatonic = all(p in scale_pcs for p in pcs)
            # in minore il V maggiore (sensibile) è comunissimo
            if key["modo"] == "minore" and r == (tonic + 7) % 12 and q == "":
                diatonic = True
            prior.append(1.0 if diatonic else 0.35)
            roots.append(r)
            quals.append(q)
    return names, np.array(temps), np.array(prior), roots, quals


def estimate_chords(instr, bass_audio, beat_t, duration, key, phase, beats_per_slot=2):
    harm = librosa.effects.harmonic(instr)
    chroma = librosa.feature.chroma_cqt(y=harm, sr=SR, hop_length=HOP, bins_per_octave=36)
    bchroma = librosa.feature.chroma_cqt(y=bass_audio, sr=SR, hop_length=HOP, fmin=librosa.note_to_hz("C1"), n_octaves=3)
    rms = librosa.feature.rms(y=harm, hop_length=HOP)[0]
    # slot di mezza battuta allineati ai primi tempi
    slot_bounds_beats = np.arange(phase - 4, len(beat_t) + 4, beats_per_slot)
    period = np.median(np.diff(beat_t))

    def beat_to_time(b):
        if b < 0:
            return beat_t[0] + b * period
        if b >= len(beat_t) - 1:
            return beat_t[-1] + (b - len(beat_t) + 1) * period
        return float(np.interp(b, np.arange(len(beat_t)), beat_t))

    slots = []
    for b0 in slot_bounds_beats:
        t0, t1 = beat_to_time(b0), beat_to_time(b0 + beats_per_slot)
        if t1 <= 0 or t0 >= duration:
            continue
        slots.append((b0, max(t0, 0.0), min(t1, duration)))
    names, temps, prior, roots, quals = chord_vocab(key)
    feats, bfeats, energy = [], [], []
    for _, t0, t1 in slots:
        a, b = librosa.time_to_frames([t0, t1], sr=SR, hop_length=HOP)
        b = max(b, a + 1)
        feats.append(chroma[:, a:b].mean(axis=1))
        bfeats.append(bchroma[:, a:b].mean(axis=1))
        energy.append(rms[a:b].mean() if b <= len(rms) else 0.0)
    F = np.log1p(20 * np.array(feats).T)
    F = F - 0.5 * F.mean(axis=0, keepdims=True)
    F /= np.linalg.norm(F, axis=0, keepdims=True) + 1e-9
    B = np.array(bfeats).T
    B /= B.max(axis=0, keepdims=True) + 1e-9
    sim = temps @ F + 0.25 * B[roots, :]  # bonus se il basso suona la fondamentale
    logp = 10 * sim + np.log(prior)[:, None]
    prob = np.exp(logp - logp.max(axis=0, keepdims=True))
    prob /= prob.sum(axis=0, keepdims=True)
    n = len(names)
    trans = np.full((n, n), 0.4 / (n - 1))
    np.fill_diagonal(trans, 0.6)
    path = librosa.sequence.viterbi_discriminative(prob, trans)
    energy = np.array(energy)
    silent = energy < 0.1 * np.median(energy)
    out = []
    for (b0, t0, t1), c, sil in zip(slots, path, silent):
        out.append({"beat": float(b0), "durata_beat": beats_per_slot, "accordo": "N" if sil else names[c],
                    "root_pc": None if sil else roots[c], "qualita": None if sil else quals[c],
                    "basso_chroma": B[:, len(out)].tolist()})
    return out


# ---------------------------------------------------------------- melodia
def simplify_melody(raw_notes, beat_t, key, grid=0.5):
    """Riduce la melodia a una linea cantabile su griglia di crome."""
    if not raw_notes:
        return []
    tonic = NOTE_NAMES.index(key["tonica"])
    scale = MAJOR_SCALE if key["modo"] == "maggiore" else MINOR_SCALE
    scale_pcs = {(tonic + s) % 12 for s in scale}
    notes = [{"pitch": n["pitch"], "b0": time_to_beat(n["start"], beat_t), "b1": time_to_beat(n["end"], beat_t),
              "e": n["energia"]} for n in raw_notes]
    # correzione errori d'ottava (salti isolati > 9 semitoni)
    for i in range(1, len(notes) - 1):
        p, a, b = notes[i]["pitch"], notes[i - 1]["pitch"], notes[i + 1]["pitch"]
        for shift in (-12, 12):
            if abs(p - a) > 9 and abs(p - b) > 9 and abs(p + shift - a) <= 7 and abs(p + shift - b) <= 7:
                notes[i]["pitch"] = p + shift
    last = max(n["b1"] for n in notes)
    nslots = int(np.ceil(last / grid)) + 1
    weight = [defaultdict(float) for _ in range(nslots)]
    onset = [set() for _ in range(nslots)]
    for n in notes:
        s0, s1 = int(np.floor(n["b0"] / grid)), int(np.floor(n["b1"] / grid))
        for s in range(max(s0, 0), min(s1, nslots - 1) + 1):
            ov = min(n["b1"], (s + 1) * grid) - max(n["b0"], s * grid)
            if ov > 0:
                weight[s][n["pitch"]] += ov * n["e"]
        # attacco: arrotondato alla croma più vicina
        so = int(round(n["b0"] / grid))
        if 0 <= so < nslots:
            onset[so].add(n["pitch"])
    emax = np.median([n["e"] for n in notes])
    seq = []
    for s in range(nslots):
        if not weight[s]:
            seq.append(None)
            continue
        p, w = max(weight[s].items(), key=lambda kv: kv[1])
        seq.append(p if w >= 0.3 * grid * emax else None)
    # costruisci note: nuova nota se cambia altezza o c'è un attacco
    mel = []
    for s, p in enumerate(seq):
        if p is None:
            continue
        if mel and mel[-1]["pitch"] == p and mel[-1]["end"] == s and p not in onset[s]:
            mel[-1]["end"] = s + 1
        else:
            mel.append({"pitch": p, "start": s, "end": s + 1})
    # note fuori tonalità lunghe una sola croma -> grado della scala più vicino
    for i, n in enumerate(mel):
        if n["pitch"] % 12 not in scale_pcs and n["end"] - n["start"] == 1:
            cands = [n["pitch"] + d for d in (-1, 1) if (n["pitch"] + d) % 12 in scale_pcs]
            ref = mel[i - 1]["pitch"] if i else n["pitch"]
            n["pitch"] = min(cands, key=lambda c: abs(c - ref))
    # unisci note ripetute brevissime nate dalla semplificazione e riempi i buchi di una croma
    out = []
    for n in mel:
        if out and n["start"] - out[-1]["end"] == 1:
            out[-1]["end"] = n["start"]
        out.append(n)
    return [{"pitch": n["pitch"], "beat": n["start"] * grid, "durata_beat": (n["end"] - n["start"]) * grid} for n in out]


# ---------------------------------------------------------------- basso
def bass_line(chords, raw_bass, beat_t):
    """Una nota di basso per accordo: fondamentale, o rivolto se il basso suona chiaramente un'altra nota dell'accordo."""
    bnotes = [(time_to_beat(n["start"], beat_t), time_to_beat(n["end"], beat_t), n["pitch"]) for n in raw_bass]
    out = []
    for c in chords:
        if c["root_pc"] is None:
            continue
        r, q = c["root_pc"], c["qualita"]
        tones = {r, (r + (3 if q == "m" else 4)) % 12, (r + 7) % 12}
        b0, b1 = c["beat"], c["beat"] + c["durata_beat"]
        cnt = Counter()
        for s, e, p in bnotes:
            ov = min(e, b1) - max(s, b0)
            if ov > 0:
                cnt[p % 12] += ov
        pc = r
        if cnt:
            top, w = cnt.most_common(1)[0]
            if top in tones and top != r and w > 0.6 * c["durata_beat"]:
                pc = top
        pitch = 36 + ((pc - 0) % 12)  # C2..B2
        if pitch > 43:
            pitch -= 12  # tiene il basso tra E1 e G2 circa
        out.append({"pitch": pitch, "beat": b0, "durata_beat": c["durata_beat"], "accordo": c["accordo"]})
    return out


def merge_chords(chords):
    out = []
    for c in chords:
        c = {k: v for k, v in c.items() if k != "basso_chroma"}
        if out and out[-1]["accordo"] == c["accordo"]:
            out[-1]["durata_beat"] += c["durata_beat"]
        else:
            out.append(c)
    return out


# ---------------------------------------------------------------- struttura
def structure(mix, beat_t, phase, n_bars):
    chroma = librosa.feature.chroma_cqt(y=mix, sr=SR, hop_length=HOP)
    mfcc = librosa.feature.mfcc(y=mix, sr=SR, hop_length=HOP, n_mfcc=13)
    bars_t = [beat_t[i] for i in range(phase, len(beat_t), 4)]
    fr = librosa.time_to_frames(bars_t, sr=SR, hop_length=HOP)
    feat = np.vstack([librosa.util.normalize(chroma, axis=0), 0.3 * librosa.util.normalize(mfcc, axis=1)])
    sync = librosa.util.sync(feat, fr, aggregate=np.mean)
    k = int(np.clip(round(len(bars_t) / 8), 4, 12))
    bounds = librosa.segment.agglomerative(sync, k)
    # frasi pop: confini allineati a gruppi di 4 battute, sezioni di almeno 4 battute
    bounds = sorted({int(round(b / 4) * 4) for b in bounds} | {0})
    bounds = [b for b in bounds if b < sync.shape[1] - 2] + [sync.shape[1]]
    segs = [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
    means = [sync[:, a:b].mean(axis=1) for a, b in segs]
    labels, protos = [], []
    for m in means:
        m = m / (np.linalg.norm(m) + 1e-9)
        sims = [float(m @ p) for p in protos]
        if sims and max(sims) > 0.93:
            labels.append(chr(65 + int(np.argmax(sims))))
        else:
            protos.append(m)
            labels.append(chr(64 + len(protos)))
    return [{"sezione": l, "battuta_inizio": int(a) + 1, "battuta_fine": int(b)} for l, (a, b) in zip(labels, segs)]


# ---------------------------------------------------------------- export
def to_music21(sk):
    from music21 import clef, harmony, key as m21key, meter, metadata, note, stream, tempo, layout

    pad = sk["pad_beat"]
    k = m21key.Key(sk["tonalita"]["tonica"].replace("b", "-"), "major" if sk["tonalita"]["modo"] == "maggiore" else "minor")
    use_flats = k.sharps < 0

    def mk_pitch(n):
        from music21 import pitch
        p = pitch.Pitch(midi=n)
        if use_flats and p.accidental is not None and p.accidental.name == "sharp":
            p = p.getEnharmonic()
        return p

    score = stream.Score()
    score.metadata = metadata.Metadata(title=sk["brano"], composer="Scheletro automatico")
    mel = stream.Part(id="melodia")
    mel.partName = "Melodia"
    mel.insert(0, clef.TrebleClef())
    mel.insert(0, k)
    mel.insert(0, meter.TimeSignature("4/4"))
    mel.insert(0, tempo.MetronomeMark(number=round(sk["tempo_bpm"])))
    for n in sk["melodia"]:
        x = note.Note(mk_pitch(n["pitch"]))
        x.quarterLength = n["durata_beat"]
        mel.insert(n["beat"] + pad, x)
    for c in sk["accordi"]:
        if c["accordo"] == "N":
            continue
        fig = c["accordo"].replace("b", "-")
        cs = harmony.ChordSymbol(fig)
        cs.writeAsChord = False
        mel.insert(c["beat"] + pad, cs)
    bas = stream.Part(id="basso")
    bas.partName = "Basso"
    bas.insert(0, clef.BassClef())
    bas.insert(0, k)
    bas.insert(0, meter.TimeSignature("4/4"))
    for n in sk["basso"]:
        x = note.Note(mk_pitch(n["pitch"]))
        x.quarterLength = n["durata_beat"]
        bas.insert(n["beat"] + pad, x)
    for p in (mel, bas):
        p.makeRests(fillGaps=True, inPlace=True, timeRangeFromBarDuration=True)
        p.makeMeasures(inPlace=True)
        p.makeTies(inPlace=True)
        from music21.stream import makeNotation as _mn
        _mn.makeAccidentalsInMeasureStream(p.getElementsByClass("Measure"), useKeySignature=k, overrideStatus=True)
        score.insert(0, p)
    # segna le sezioni come testo sopra la battuta
    from music21 import expressions
    measures = list(score.parts[0].getElementsByClass("Measure"))
    for s in sk["struttura"]:
        idx = s["battuta_inizio"] - 1 + (1 if pad else 0)
        if 0 <= idx < len(measures):
            te = expressions.TextExpression(f"[{s['sezione']}]")
            te.style.fontWeight = "bold"
            te.placement = "above"
            measures[idx].insert(0, te)
    return score


def render_pdf(xml_path, pdf_path):
    """MusicXML -> PDF con verovio + cairosvg. Se cairo non è installato salva le pagine in SVG."""
    import verovio
    try:
        import cairosvg
        from pypdf import PdfWriter
    except (ImportError, OSError):
        cairosvg = None

    # music21 scrive <root-alter>0</root-alter> su alcune sigle: verovio lo mostra come bequadro
    xml_txt = Path(xml_path).read_text()
    xml_txt = re.sub(r"\s*<root-alter>0</root-alter>", "", xml_txt)
    Path(xml_path).write_text(xml_txt)
    tk = verovio.toolkit()
    tk.setOptions({"pageWidth": 2100, "pageHeight": 2970, "scale": 40, "adjustPageHeight": False,
                   "footer": "none", "breaks": "auto"})
    tk.loadFile(str(xml_path))
    writer = PdfWriter() if cairosvg else None
    tmp = []
    for i in range(1, tk.getPageCount() + 1):
        p = pdf_path.with_suffix(f".p{i}.pdf")
        svg = tk.renderToSVG(i)
        if cairosvg is None:
            pdf_path.with_suffix(f".pagina{i}.svg").write_text(svg)
            continue
        # i glifi SMuFL nei testi (sigle, metronomo) non hanno un font in cairo: usa caratteri Unicode
        svg = re.sub(r'<tspan font-family="Leipzig" font-size="(\d+)px">(.)</tspan>',
                     lambda m: f'<tspan font-family="DejaVu Serif" font-size="{int(int(m.group(1)) * 0.55)}px">'
                               f'{SMUFL_TEXT.get(m.group(2), m.group(2))}</tspan>', svg)
        cairosvg.svg2pdf(bytestring=svg.encode(), write_to=str(p))
        writer.append(str(p))
        tmp.append(p)
    if writer is None:
        return tk.getPageCount()
    with open(pdf_path, "wb") as f:
        writer.write(f)
    for p in tmp:
        p.unlink()
    return tk.getPageCount()


def write_midi_and_check(sk, beat_t, mix, out):
    period = np.median(np.diff(beat_t))

    def bt(b):
        return float(np.interp(b, np.arange(len(beat_t)), beat_t)) if 0 <= b <= len(beat_t) - 1 else (
            beat_t[0] + b * period if b < 0 else beat_t[-1] + (b - len(beat_t) + 1) * period)

    pm = pretty_midi.PrettyMIDI(initial_tempo=sk["tempo_bpm"])
    for label, notes, prog, vel in (("Melodia", sk["melodia"], 0, 100), ("Basso", sk["basso"], 33, 90)):
        inst = pretty_midi.Instrument(program=prog, name=label)
        for n in notes:
            s, e = bt(n["beat"]), bt(n["beat"] + n["durata_beat"])
            if e > s >= 0:
                inst.notes.append(pretty_midi.Note(vel, n["pitch"], s, e))
        pm.instruments.append(inst)
    ch = pretty_midi.Instrument(program=48, name="Accordi")
    for c in sk["accordi"]:
        if c["accordo"] == "N":
            continue
        s, e = bt(c["beat"]), bt(c["beat"] + c["durata_beat"])
        third = 3 if c["qualita"] == "m" else 4
        for iv in (0, third, 7):
            if e > s >= 0:
                ch.notes.append(pretty_midi.Note(50, 55 + (c["root_pc"] + iv - 7) % 12, s, e))
    pm.instruments.append(ch)
    pm.write(str(out / "scheletro.mid"))
    synth = pm.synthesize(fs=SR)
    L = max(len(synth), len(mix))
    synth = np.pad(synth, (0, L - len(synth)))
    m = np.pad(mix, (0, L - len(mix)))
    synth /= np.abs(synth).max() + 1e-9
    m /= np.abs(m).max() + 1e-9
    y = 0.3 * m + 0.6 * synth
    sf.write(out / "verifica.wav", np.stack([y, y], axis=1), SR)


def main(name, root=None):
    ROOT = Path(root) if root else Path(__file__).parent / "data"
    raw = json.loads((ROOT / "analisi" / name / "skeleton.json").read_text())
    out = ROOT / "scheletro" / name
    out.mkdir(parents=True, exist_ok=True)
    mix = load(ROOT / "input" / f"{name}.mp3")
    instr = load(ROOT / "stems" / f"{name}_instrumental.wav")
    b, a = sps.butter(4, 250 / (SR / 2), btype="low")
    bass_audio = sps.filtfilt(b, a, instr).astype(np.float32)
    duration = len(mix) / SR

    beat_t, tempo = fix_beats(raw["beat_times"], raw["tempo_bpm"])
    phase = downbeat_phase(beat_t, mix, bass_audio)
    key = raw["tonalita"]
    chords_slots = estimate_chords(instr, bass_audio, beat_t, duration, key, phase)
    melody = simplify_melody(raw["melodia"], beat_t, key)
    # voce maschile: porta la melodia nel registro della mano destra (centro intorno a Do4-Sol4)
    shift = 0
    if melody:
        med = float(np.median([n["pitch"] for n in melody]))
        shift = 12 * int(round((67 - med) / 12))
        for n in melody:
            n["pitch"] += shift
    bass = bass_line(chords_slots, raw["basso"], beat_t)
    chords = merge_chords(chords_slots)
    n_bars = int((len(beat_t) - phase) // 4)
    struct = structure(mix, beat_t, phase, n_bars)

    # riallinea: i primi tempi cadono sui multipli di 4 dopo il "pad"
    pad = (4 - phase) % 4
    sk = {
        "brano": name, "tempo_bpm": round(tempo, 1), "metro": "4/4", "tonalita": key,
        "primo_downbeat_beat": phase, "pad_beat": pad, "griglia_melodia": "croma", "trasposizione_ottava_melodia": shift,
        "struttura": struct, "accordi": chords, "melodia": melody, "basso": bass,
        "beat_times": [round(float(t), 3) for t in beat_t],
    }
    # elimina eventi prima dell'inizio del pezzo
    for kk in ("accordi", "melodia", "basso"):
        sk[kk] = [e for e in sk[kk] if e["beat"] + pad >= 0]
    (out / "scheletro.json").write_text(json.dumps(sk, indent=1, ensure_ascii=False))

    score = to_music21(sk)
    xml = out / "lead_sheet.musicxml"
    score.write("musicxml", fp=str(xml))
    pages = render_pdf(xml, out / "lead_sheet.pdf")
    write_midi_and_check(sk, beat_t, mix, out)

    durs = Counter(n["durata_beat"] for n in melody)
    print(json.dumps({
        "brano": name, "tempo": sk["tempo_bpm"], "battute": n_bars, "note_melodia": len(melody),
        "accordi": len(chords), "sezioni": "".join(s["sezione"] for s in struct), "pagine": pages,
        "durate_piu_comuni": durs.most_common(4),
        "giro_iniziale": [c["accordo"] for c in chords if c["accordo"] != "N"][:12],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1])
