"""Passo 3: dallo scheletro a uno spartito per pianoforte, in tre livelli.

Uso: python arrangia.py <nome_brano> [principiante|intermedio|avanzato ...]
Legge data/scheletro/<nome>/scheletro.json, scrive in data/arrangiamenti/<nome>/:
  <livello>.pdf, <livello>.musicxml, <livello>.mid, controllo.json
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pretty_midi

from .analizza import NOTE_NAMES
from .scheletro import render_pdf

SHARP_NAMES = NOTE_NAMES
FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
# alterazioni in chiave delle tonalità maggiori (per pitch class della tonica)
MAJOR_ACC = {0: 0, 7: 1, 2: 2, 9: 3, 4: 4, 11: 5, 6: 6, 1: 5, 8: 4, 3: 3, 10: 2, 5: 1}
FLAT_MAJORS = {5, 10, 3, 8, 1}

LIVELLI = {
    # rh: griglia melodia, estensione; lh: stile; max_acc: alterazioni max in chiave
    "principiante": {"griglia": 1.0, "rh": (60, 79), "centro": 67, "max_acc": 1, "lh": "lunghe", "armonia": 0},
    "intermedio": {"griglia": 0.5, "rh": (60, 84), "centro": 69, "max_acc": 3, "lh": "pattern", "armonia": 0},
    "avanzato": {"griglia": 0.5, "rh": (60, 86), "centro": 70, "max_acc": 7, "lh": "arpeggio", "armonia": 2},
}


# ---------------------------------------------------------------- tonalità
def key_signature_acc(tonic_pc, mode):
    major_pc = tonic_pc if mode == "maggiore" else (tonic_pc + 3) % 12
    return MAJOR_ACC[major_pc], major_pc


def choose_transposition(key, max_acc):
    tonic = NOTE_NAMES.index(key["tonica"])
    acc0, _ = key_signature_acc(tonic, key["modo"])
    if acc0 <= max_acc:
        return 0
    best = None
    for t in range(-6, 6):
        acc, _ = key_signature_acc((tonic + t) % 12, key["modo"])
        if acc <= max_acc:
            cand = (abs(t), acc, t)  # il trasporto più piccolo che rientra nel limite
            if best is None or cand < best:
                best = cand
    return best[2]


# ---------------------------------------------------------------- utilità
def chord_at(chords, beat):
    for c in chords:
        if c["beat"] <= beat < c["beat"] + c["durata_beat"]:
            return c
    return None


def chord_pcs(c):
    if c is None or c["root_pc"] is None:
        return []
    r = c["root_pc"]
    return [r, (r + (3 if c["qualita"] == "m" else 4)) % 12, (r + 7) % 12]


def fold(p, lo, hi):
    while p < lo:
        p += 12
    while p > hi:
        p -= 12
    return p


def place(pc, lo):
    """La nota con pitch class pc più bassa >= lo."""
    return lo + (pc - lo) % 12


# ---------------------------------------------------------------- mano destra
def melody_on_grid(melody, grid):
    """Rimappa la melodia (già a crome) su una griglia più larga: una nota per cella, senza sincopi."""
    if grid <= 0.5:
        return [dict(n) for n in melody]
    cells = defaultdict(lambda: defaultdict(float))
    onsets = defaultdict(set)
    for n in melody:
        b0, b1 = n["beat"], n["beat"] + n["durata_beat"]
        c = int(np.floor(b0 / grid))
        while c * grid < b1:
            ov = min(b1, (c + 1) * grid) - max(b0, c * grid)
            if ov > 0:
                cells[c][n["pitch"]] += ov
            c += 1
        onsets[int(np.floor(b0 / grid))].add(n["pitch"])
    out = []
    for c in sorted(cells):
        p, w = max(cells[c].items(), key=lambda kv: kv[1])
        if w < 0.5 * grid:
            continue
        if out and out[-1]["pitch"] == p and out[-1]["beat"] + out[-1]["durata_beat"] == c * grid and p not in onsets[c]:
            out[-1]["durata_beat"] += grid
        else:
            out.append({"pitch": p, "beat": c * grid, "durata_beat": grid})
    return out


def right_hand(sk, cfg, t):
    lo, hi = cfg["rh"]
    mel = [dict(n, pitch=n["pitch"] + t) for n in sk["melodia"]]
    mel = melody_on_grid(mel, cfg["griglia"])
    if mel:
        med = np.median([n["pitch"] for n in mel])
        shift = 12 * int(round((cfg["centro"] - med) / 12))
        for n in mel:
            n["pitch"] = fold(n["pitch"] + shift, lo, hi)
    # riempi i buchi piccoli (legato) e accorcia le note lunghissime
    for a, b in zip(mel, mel[1:]):
        gap = b["beat"] - (a["beat"] + a["durata_beat"])
        if 0 < gap <= cfg["griglia"]:
            a["durata_beat"] += gap
    for n in mel:
        n["durata_beat"] = min(n["durata_beat"], 4.0)
        n["pitches"] = [n["pitch"]]
    # armonizzazione (avanzato): note dell'accordo sotto la melodia sulle note lunghe in battere
    if cfg["armonia"]:
        for n in mel:
            if n["durata_beat"] < 1 or n["beat"] % 1:
                continue
            c = chord_at(sk["_chords_t"], n["beat"])
            pcs = chord_pcs(c)
            added = []
            for d in range(3, 10):  # da una terza a una sesta/settima sotto
                q = n["pitch"] - d
                if q % 12 in pcs and q >= lo - 5:
                    added.append(q)
            if added:
                n["pitches"] = [n["pitch"], added[0]]
                if cfg["armonia"] > 1 and n["durata_beat"] >= 2 and len(added) > 1:
                    low = [q for q in added[1:] if n["pitch"] - q <= 12]
                    if low:
                        n["pitches"].append(low[-1])
    return mel


# ---------------------------------------------------------------- mano sinistra
def voicings(pcs, lo=43, top_max=59):
    """Tutti i rivolti in posizione stretta (entro un'ottava) con la nota più bassa >= lo."""
    out = []
    for inv in range(len(pcs)):
        order = pcs[inv:] + pcs[:inv]
        for base in range(lo, lo + 12):
            if base % 12 != order[0]:
                continue
            notes = [base]
            for pc in order[1:]:
                notes.append(notes[-1] + (pc - notes[-1]) % 12)
            if notes[-1] <= top_max and notes[-1] - notes[0] <= 12:
                out.append(notes)
    return out


def voice_lead(prev, pcs):
    """Sceglie il rivolto più vicino all'accordo precedente: la mano si sposta il meno possibile."""
    cands = voicings(pcs)
    if not cands:
        return sorted(place(pc, 48) for pc in pcs)
    if prev is None:
        # primo accordo: centrato intorno a Re3-Fa3
        return min(cands, key=lambda v: abs(np.mean(v) - 53))
    return min(cands, key=lambda v: sum(abs(a - b) for a, b in zip(v, prev)) + 0.3 * abs(np.mean(v) - 53))


def bar_pieces(sk, b0, dur):
    """Spezza [b0, b0+dur) alle stanghette di battuta."""
    pos = b0
    while pos < b0 + dur - 1e-6:
        end_bar = (np.floor((pos + sk["pad_beat"]) / 4) + 1) * 4 - sk["pad_beat"]
        d = min(b0 + dur, end_bar) - pos
        yield pos, d
        pos += d


def left_hand_chords(sk, cfg):
    """Mano sinistra ad accordi (triadi nei rivolti più comodi), ritmo secondo il livello."""
    out, prev = [], None
    for c in sk["_chords_t"]:
        if c["root_pc"] is None:
            continue
        v = voice_lead(prev, chord_pcs(c))
        prev = v
        if cfg["lh"] == "lunghe":
            # accordo tenuto per tutta la battuta (o finché dura)
            for pos, d in bar_pieces(sk, c["beat"], c["durata_beat"]):
                out.append({"pitches": list(v), "beat": pos, "durata_beat": d})
        elif cfg["lh"] == "pattern":
            # accordo sul primo e sul terzo tempo (minime)
            for pos, d in bar_pieces(sk, c["beat"], c["durata_beat"]):
                k = pos
                while k < pos + d - 1e-6:
                    step = min(2.0, pos + d - k)
                    out.append({"pitches": list(v), "beat": k, "durata_beat": step})
                    k += step
        else:
            # basso (fondamentale grave) sul tempo forte, poi l'accordo sugli altri tempi
            root = place(c["root_pc"], 36)  # Do2..Si2
            for pos, d in bar_pieces(sk, c["beat"], c["durata_beat"]):
                out.append({"pitches": [root], "beat": pos, "durata_beat": 1.0 if d > 1 else d})
                k = pos + 1
                while k < pos + d - 1e-6:
                    out.append({"pitches": list(v), "beat": k, "durata_beat": min(1.0, pos + d - k)})
                    k += 1
    return out


def left_hand(sk, cfg, t, stile="accordi"):
    if stile == "accordi":
        return left_hand_chords(sk, cfg)
    out = []
    for c in sk["_chords_t"]:
        if c["root_pc"] is None:
            continue
        pcs = chord_pcs(c)
        r = c["root_pc"]
        b0, dur = c["beat"], c["durata_beat"]
        if cfg["lh"] == "lunghe":
            # una nota (fondamentale) per accordo, al massimo una per battuta: niente salti
            root = place(r, 48)  # Do3..Si3
            pos = b0
            while pos < b0 + dur - 1e-6:
                end_bar = (np.floor((pos + sk["pad_beat"]) / 4) + 1) * 4 - sk["pad_beat"]
                d = min(b0 + dur, end_bar) - pos
                out.append({"pitches": [root], "beat": pos, "durata_beat": d})
                pos += d
        elif cfg["lh"] == "pattern":
            # fondamentale + quinta + ottava + quinta (semiminime), posizione fissa della mano
            root = place(r, 41)  # Fa2..Mi3
            patt = [root, root + 7, root + 12, root + 7]
            for i in range(int(round(dur))):
                out.append({"pitches": [patt[i % 4]], "beat": b0 + i, "durata_beat": 1.0})
        else:
            # arpeggio in crome: R 5 8 10 8 5 ... (accordo spezzato, con pedale)
            root = place(r, 36)  # Do2..Si2
            third = root + 12 + (3 if c["qualita"] == "m" else 4)
            patt = [root, root + 7, root + 12, third, root + 12, root + 7, root + 12, third]
            for i in range(int(round(dur * 2))):
                out.append({"pitches": [patt[i % 8]], "beat": b0 + i * 0.5, "durata_beat": 0.5})
    return out


def separate_hands(rh, lh):
    """Le mani non devono incrociarsi: la sinistra resta sotto la nota più bassa della destra."""
    def rh_low_at(b0, b1):
        lows = [min(n["pitches"]) for n in rh if n["beat"] < b1 and n["beat"] + n["durata_beat"] > b0]
        return min(lows) if lows else 127
    for n in lh:
        lim = rh_low_at(n["beat"], n["beat"] + n["durata_beat"]) - 1
        if len(n["pitches"]) > 1 and max(n["pitches"]) > lim:
            low = [p - 12 for p in n["pitches"]]
            # accordo: si sposta tutto un'ottava sotto se non diventa troppo grave, altrimenti si toglie la nota in conflitto
            n["pitches"] = low if min(low) >= 43 else ([p for p in n["pitches"] if p <= lim] or [min(n["pitches"]) - 12])
        else:
            n["pitches"] = [p - 12 if p > lim else p for p in n["pitches"]]
    return lh


def check_playability(rh, lh, cfg):
    def stats(hand, window=4.0):
        spans = [max(n["pitches"]) - min(n["pitches"]) for n in hand]
        poly = [len(n["pitches"]) for n in hand]
        # estensione "di posizione": note entro ogni battuta
        pos_spans = []
        by_beat = defaultdict(list)
        for n in hand:
            by_beat[int(n["beat"] // window)].extend(n["pitches"])
        for v in by_beat.values():
            pos_spans.append(max(v) - min(v))
        dur = [n["durata_beat"] for n in hand]
        return {"note": len(hand), "max_note_insieme": max(poly, default=0), "max_estensione_accordo": max(spans, default=0),
                "estensione_per_battuta_p95": int(np.percentile(pos_spans, 95)) if pos_spans else 0,
                "durata_minima": min(dur, default=0), "registro": [min((min(n["pitches"]) for n in hand), default=0),
                                                                   max((max(n["pitches"]) for n in hand), default=0)]}
    r, l = stats(rh), stats(lh)
    incroci = sum(1 for a in lh for b in rh if a["beat"] < b["beat"] + b["durata_beat"] and b["beat"] < a["beat"] + a["durata_beat"]
                  and max(a["pitches"]) >= min(b["pitches"]))
    ok = r["max_estensione_accordo"] <= 12 and l["max_estensione_accordo"] <= 12 and incroci == 0
    return {"mano_destra": r, "mano_sinistra": l, "incroci_mani": incroci, "suonabile": ok}


# ---------------------------------------------------------------- export
def to_score(sk, rh, lh, livello, t, pedale=False):
    from music21 import chord, clef, expressions, harmony, key as m21key, layout, meter, metadata, note, pitch, stream, tempo

    tonic = (NOTE_NAMES.index(sk["tonalita"]["tonica"]) + t) % 12
    mode = sk["tonalita"]["modo"]
    _, major_pc = key_signature_acc(tonic, mode)
    flats = major_pc in FLAT_MAJORS or (major_pc == 6 and False)
    names = FLAT_NAMES if flats else SHARP_NAMES
    k = m21key.Key(names[tonic].replace("b", "-"), "major" if mode == "maggiore" else "minor")
    pad = sk["pad_beat"]

    def mk(n):
        p = pitch.Pitch(midi=n)
        if k.sharps < 0 and p.accidental is not None and p.accidental.name == "sharp":
            p = p.getEnharmonic()
        return p

    def obj(n):
        if len(n["pitches"]) == 1:
            x = note.Note(mk(n["pitches"][0]))
        else:
            x = chord.Chord([mk(p) for p in sorted(n["pitches"])])
        x.quarterLength = n["durata_beat"]
        return x

    score = stream.Score()
    titolo = sk["brano"].replace("_", " ").title()
    score.metadata = metadata.Metadata(title=f"{titolo}", composer=f"Pianoforte - livello {livello}")
    up, down = stream.PartStaff(id="dx"), stream.PartStaff(id="sx")
    for st, cl in ((up, clef.TrebleClef()), (down, clef.BassClef())):
        st.insert(0, cl)
        st.insert(0, k)
        st.insert(0, meter.TimeSignature("4/4"))
    up.insert(0, tempo.MetronomeMark(number=round(sk["tempo_bpm"])))
    for n in rh:
        if n["beat"] + pad >= 0:
            up.insert(n["beat"] + pad, obj(n))
    for n in lh:
        if n["beat"] + pad >= 0:
            down.insert(n["beat"] + pad, obj(n))
    for c in sk["_chords_t"]:
        if c["root_pc"] is None or c["beat"] + pad < 0:
            continue
        cs = harmony.ChordSymbol(names[c["root_pc"]].replace("b", "-") + ("m" if c["qualita"] == "m" else ""))
        cs.writeAsChord = False
        up.insert(c["beat"] + pad, cs)
    from music21.stream import makeNotation
    for st in (up, down):
        st.makeRests(fillGaps=True, inPlace=True, timeRangeFromBarDuration=True)
        st.makeMeasures(inPlace=True)
        st.makeTies(inPlace=True)
        makeNotation.makeAccidentalsInMeasureStream(st.getElementsByClass("Measure"), useKeySignature=k, overrideStatus=True)
    # pareggia il numero di battute tra le due mani
    mu, md = list(up.getElementsByClass("Measure")), list(down.getElementsByClass("Measure"))
    measures = mu
    for s in sk["struttura"]:
        idx = s["battuta_inizio"] - 1 + (1 if pad else 0)
        if 0 <= idx < len(measures):
            te = expressions.TextExpression(f"[{s['sezione']}]")
            te.style.fontWeight = "bold"
            measures[idx].insert(0, te)
    if pedale:
        te = expressions.TextExpression("con pedale")
        te.style.fontStyle = "italic"
        md[0].insert(0, te)
    score.insert(0, up)
    score.insert(0, down)
    score.insert(0, layout.StaffGroup([up, down], symbol="brace", barTogether=True))
    return score, k


def write_midi(sk, rh, lh, path):
    pm = pretty_midi.PrettyMIDI(initial_tempo=sk["tempo_bpm"])
    spb = 60.0 / sk["tempo_bpm"]
    pad = sk["pad_beat"]
    piano = pretty_midi.Instrument(program=0, name="Pianoforte")
    for hand, vel in ((rh, 92), (lh, 70)):
        for n in hand:
            s = (n["beat"] + pad) * spb
            if s < 0:
                continue
            for p in n["pitches"]:
                piano.notes.append(pretty_midi.Note(vel, int(p), s, s + n["durata_beat"] * spb * 0.98))
    pm.instruments.append(piano)
    pm.write(str(path))


def arrange(name, livello, ROOT, stile_sx="accordi"):
    sk = json.loads((ROOT / "scheletro" / name / "scheletro.json").read_text())
    cfg = LIVELLI[livello]
    t = choose_transposition(sk["tonalita"], cfg["max_acc"])
    sk["_chords_t"] = [dict(c, root_pc=None if c["root_pc"] is None else (c["root_pc"] + t) % 12) for c in sk["accordi"]]
    if livello == "principiante":
        # accordi al massimo uno per battuta (il più lungo), per non cambiare posizione troppo spesso
        merged = []
        bars = defaultdict(list)
        for c in sk["_chords_t"]:
            if c["root_pc"] is None:
                continue
            b = c["beat"]
            while b < c["beat"] + c["durata_beat"] - 1e-6:
                bar = int(np.floor((b + sk["pad_beat"]) / 4))
                end = min(c["beat"] + c["durata_beat"], (bar + 1) * 4 - sk["pad_beat"])
                bars[bar].append((end - b, c))
                b = end
        for bar in sorted(bars):
            _, c = max(bars[bar], key=lambda x: x[0])
            nc = dict(c, beat=bar * 4 - sk["pad_beat"], durata_beat=4.0)
            if merged and merged[-1]["root_pc"] == nc["root_pc"] and merged[-1]["qualita"] == nc["qualita"]:
                merged[-1]["durata_beat"] += 4.0
            else:
                merged.append(nc)
        sk["_chords_t"] = merged
    rh = right_hand(sk, cfg, t)
    lh = separate_hands(rh, left_hand(sk, cfg, t, stile_sx))
    report = check_playability(rh, lh, cfg)
    out = ROOT / "arrangiamenti" / name
    out.mkdir(parents=True, exist_ok=True)
    score, k = to_score(sk, rh, lh, livello, t, pedale=(stile_sx == "classica" and cfg["lh"] == "arpeggio"))
    xml = out / f"{livello}.musicxml"
    score.write("musicxml", fp=str(xml))
    pages = render_pdf(xml, out / f"{livello}.pdf")
    write_midi(sk, rh, lh, out / f"{livello}.mid")
    report.update({"livello": livello, "tonalita": f"{k.tonic.name.replace('-', 'b')} {k.mode}", "trasposizione_semitoni": t,
                   "alterazioni_in_chiave": abs(k.sharps), "pagine": pages})
    return report


def main(name, livelli, root=None, stile_sx="accordi"):
    ROOT = Path(root) if root else Path(__file__).parent / "data"
    out = ROOT / "arrangiamenti" / name
    reports = {lv: arrange(name, lv, ROOT, stile_sx) for lv in livelli}
    (out / "controllo.json").write_text(json.dumps(reports, indent=1, ensure_ascii=False))
    for lv, r in reports.items():
        print(name, lv, r["tonalita"], f"T{r['trasposizione_semitoni']:+d}", "RH", r["mano_destra"]["note"], "LH", r["mano_sinistra"]["note"],
              "suonabile" if r["suonabile"] else "NON suonabile", "incroci", r["incroci_mani"],
              "estensioni", r["mano_destra"]["max_estensione_accordo"], r["mano_sinistra"]["estensione_per_battuta_p95"])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:] or list(LIVELLI))
