"""Misura la precisione dell'analisi sui brani sintetici prodotti da genera.py.

Uso: python valutazione/valuta.py <cartella> [--salta-analisi]
Esegue analizza + scheletro su ogni brano e confronta con la verità:
  tonalità, tempo, accordi (% di tempo con fondamentale+qualità giuste), melodia (F1 sulle note).
"""
import contextlib
import io
import json
import sys
import warnings
from pathlib import Path

import mir_eval
import numpy as np

from trascrittore import analizza, scheletro

NOTE_NAMES = analizza.NOTE_NAMES


def chords_at(segs, times, key_start="start", key_end="end"):
    out = []
    j = 0
    segs = sorted(segs, key=lambda c: c[key_start])
    for t in times:
        while j < len(segs) and segs[j][key_end] <= t:
            j += 1
        if j < len(segs) and segs[j][key_start] <= t:
            c = segs[j]
            out.append(None if c.get("root_pc") is None else (c["root_pc"], c.get("qualita") or ""))
        else:
            out.append(None)
    return out


def beat_to_time_fn(beat_t):
    beat_t = np.asarray(beat_t)
    period = np.median(np.diff(beat_t))

    def f(b):
        if b < 0:
            return beat_t[0] + b * period
        if b > len(beat_t) - 1:
            return beat_t[-1] + (b - len(beat_t) + 1) * period
        return float(np.interp(b, np.arange(len(beat_t)), beat_t))
    return f


def note_f1(ref, est, onset_tol=0.1):
    if not est:
        return 0.0, 0.0, 0.0
    ri = np.array([[n["start"], max(n["end"], n["start"] + 0.01)] for n in ref])
    rp = np.array([440 * 2 ** ((n["pitch"] - 69) / 12) for n in ref])
    ei = np.array([[n["start"], max(n["end"], n["start"] + 0.01)] for n in est])
    ep = np.array([440 * 2 ** ((n["pitch"] - 69) / 12) for n in est])
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(ri, rp, ei, ep, onset_tolerance=onset_tol, offset_ratio=None)
    return p, r, f


def evaluate(root, name):
    truth = json.loads((root / "verita" / f"{name}.json").read_text())
    raw = json.loads((root / "analisi" / name / "skeleton.json").read_text())
    sk = json.loads((root / "scheletro" / name / "scheletro.json").read_text())
    res = {"brano": name}
    # tonalità
    est = (NOTE_NAMES.index(sk["tonalita"]["tonica"]), sk["tonalita"]["modo"])
    ref = (NOTE_NAMES.index(truth["tonica"]), truth["modo"])
    rel = ((ref[0] + (3 if ref[1] == "minore" else -3)) % 12, "maggiore" if ref[1] == "minore" else "minore")
    res["tonalita"] = 1.0 if est == ref else (0.5 if est == rel else 0.0)
    # tempo
    res["tempo"] = float(abs(sk["tempo_bpm"] - truth["tempo_bpm"]) / truth["tempo_bpm"] < 0.04)
    # accordi (fondamentale + maggiore/minore), campionati ogni 50 ms dove la verità ha un accordo
    t_end = max(c["end"] for c in truth["accordi"])
    times = np.arange(truth["lead_in"], t_end, 0.05)
    b2t = beat_to_time_fn(sk["beat_times"])
    est_segs = [dict(c, start=b2t(c["beat"]), end=b2t(c["beat"] + c["durata_beat"])) for c in sk["accordi"]]
    ref_c = chords_at(truth["accordi"], times)
    est_c = chords_at(est_segs, times)
    res["accordi"] = float(np.mean([a == b for a, b in zip(ref_c, est_c)]))
    res["accordi_fondamentale"] = float(np.mean([b is not None and a[0] == b[0] for a, b in zip(ref_c, est_c)]))
    # melodia grezza (passo 1, tempi reali) e semplificata (passo 2, griglia di crome)
    p, r, f = note_f1(truth["melodia"], raw["melodia"])
    res["melodia_grezza_f1"] = f
    shift = sk.get("trasposizione_ottava_melodia", 0)
    mel = [{"pitch": n["pitch"] - shift, "start": b2t(n["beat"]), "end": b2t(n["beat"] + n["durata_beat"])} for n in sk["melodia"]]
    p, r, f = note_f1(truth["melodia"], mel, onset_tol=0.15)
    res["melodia_f1"] = f
    res["melodia_precisione"], res["melodia_richiamo"] = p, r
    return res


def main(root, skip=False):
    root = Path(root)
    names = sorted(p.stem for p in (root / "verita").glob("*.json"))
    rows = []
    for n in names:
        if not skip:
            with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
                warnings.simplefilter("ignore")
                analizza.main(n, root=root)
                scheletro.main(n, root=root)
        rows.append(evaluate(root, n))
        r = rows[-1]
        print(f"{n}: tonalità {r['tonalita']:.1f}  tempo {r['tempo']:.0f}  accordi {r['accordi']:.2f} "
              f"(fond. {r['accordi_fondamentale']:.2f})  melodia F1 {r['melodia_grezza_f1']:.2f} -> {r['melodia_f1']:.2f}", flush=True)
    keys = [k for k in rows[0] if k != "brano"]
    media = {k: round(float(np.mean([r[k] for r in rows])), 3) for k in keys}
    print("MEDIA", json.dumps(media))
    (root / "risultati.json").write_text(json.dumps({"media": media, "brani": rows}, indent=1))
    return media


if __name__ == "__main__":
    main(sys.argv[1], "--salta-analisi" in sys.argv)
