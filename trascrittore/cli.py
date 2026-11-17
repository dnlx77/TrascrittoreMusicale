"""trascrivi: da un brano audio a spartiti per pianoforte in tre livelli.

Esempi:
  trascrivi brano.mp3
  trascrivi brano.mp3 --livello principiante
  trascrivi canzone1.mp3 canzone2.wav --uscita spartiti --audio
"""
import argparse
import contextlib
import io
import json
import re
import shutil
import subprocess
import sys
import time
import unicodedata
import warnings
from pathlib import Path

LIVELLI = ["principiante", "intermedio", "avanzato"]


def nome_sicuro(path):
    s = unicodedata.normalize("NFKD", Path(path).stem).encode("ascii", "ignore").decode()
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()
    return s or "brano"


def passo(msg):
    print(f"  • {msg}", flush=True)


@contextlib.contextmanager
def silenzioso(attivo=True):
    """Nasconde i messaggi tecnici delle librerie (rimangono visibili con --dettagli)."""
    if not attivo:
        yield
        return
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        yield


def trascrivi(audio, uscita, livelli, cache, audio_anteprima, dettagli):
    from . import analizza, arrangia, scheletro
    from .separa import separa

    audio = Path(audio).expanduser().resolve()
    if not audio.exists():
        print(f"File non trovato: {audio}", file=sys.stderr)
        return False
    nome = nome_sicuro(audio)
    lavoro = Path(cache)
    (lavoro / "input").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"\n♪ {audio.name}  →  {Path(uscita) / nome}")

    # i moduli si aspettano <lavoro>/input/<nome>.mp3
    mp3 = lavoro / "input" / f"{nome}.mp3"
    if not mp3.exists():
        if audio.suffix.lower() == ".mp3":
            shutil.copy(audio, mp3)
        else:
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(audio), "-b:a", "192k", str(mp3)], check=True)

    passo("separo voce e base (la prima volta scarica il modello, ~70 MB)")
    with silenzioso(not dettagli):
        separa(mp3, nome, lavoro / "stems", lavoro / "modelli")
    passo("analizzo tempo, tonalità, melodia, basso e accordi")
    with silenzioso(not dettagli):
        analizza.main(nome, root=lavoro)
    passo("costruisco lo scheletro e il lead sheet")
    with silenzioso(not dettagli):
        scheletro.main(nome, root=lavoro)
    passo(f"arrangio per pianoforte: {', '.join(livelli)}")
    with silenzioso(not dettagli):
        arrangia.main(nome, livelli, root=lavoro)

    dest = Path(uscita) / nome
    dest.mkdir(parents=True, exist_ok=True)
    arr = lavoro / "arrangiamenti" / nome
    sch = lavoro / "scheletro" / nome
    for lv in livelli:
        for ext in ("pdf", "musicxml", "mid"):
            if (arr / f"{lv}.{ext}").exists():
                shutil.copy(arr / f"{lv}.{ext}", dest / f"{nome}_{lv}.{ext}")
        for svg in arr.glob(f"{lv}.pagina*.svg"):
            shutil.copy(svg, dest / f"{nome}_{svg.name}")
    for ext in ("pdf", "musicxml"):
        if (sch / f"lead_sheet.{ext}").exists():
            shutil.copy(sch / f"lead_sheet.{ext}", dest / f"{nome}_lead_sheet.{ext}")
    shutil.copy(arr / "controllo.json", dest / f"{nome}_controllo.json")

    if audio_anteprima:
        from .ascolta import render
        passo("creo le anteprime audio")
        for lv in livelli:
            wav = dest / f"{nome}_{lv}.wav"
            render(str(arr / f"{lv}.mid"), str(wav))
            if shutil.which("ffmpeg"):
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav), "-b:a", "128k", str(wav.with_suffix(".mp3"))], check=True)
                wav.unlink()

    rep = json.loads((arr / "controllo.json").read_text())
    sk = json.loads((sch / "scheletro.json").read_text())
    print(f"  ✓ fatto in {time.time() - t0:.0f} s — tempo {sk['tempo_bpm']:.0f} BPM, tonalità originale "
          f"{sk['tonalita']['tonica']} {sk['tonalita']['modo']}")
    for lv, r in rep.items():
        stato = "suonabile" if r["suonabile"] else "DA CONTROLLARE"
        tr = f", trasportato di {r['trasposizione_semitoni']:+d} semitoni" if r["trasposizione_semitoni"] else ""
        ton = r["tonalita"].replace("major", "maggiore").replace("minor", "minore")
        print(f"    {lv:<13} {ton}{tr} — {r['pagine']} pagine — {stato}")
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(prog="trascrivi", description="Da un brano audio a spartiti per pianoforte suonabili.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split("\n", 2)[2])
    ap.add_argument("file", nargs="+", metavar="audio", help="uno o più file audio (mp3, wav, flac, m4a…)")
    ap.add_argument("-l", "--livello", choices=LIVELLI + ["tutti"], default="tutti", help="livello dello spartito (default: tutti)")
    ap.add_argument("-o", "--uscita", default="spartiti", help="cartella dove salvare gli spartiti (default: ./spartiti)")
    ap.add_argument("--audio", dest="anteprima", action="store_true", help="crea anche un'anteprima audio mp3 di ogni spartito")
    ap.add_argument("--cache", default=str(Path.home() / ".cache" / "trascrittore"),
                    help="cartella di lavoro: tracce separate e modelli, riusate tra un'esecuzione e l'altra")
    ap.add_argument("--dettagli", action="store_true", help="mostra i messaggi tecnici delle librerie")
    a = ap.parse_args(argv)
    livelli = LIVELLI if a.livello == "tutti" else [a.livello]
    ok = all([trascrivi(f, a.uscita, livelli, a.cache, a.anteprima, a.dettagli) for f in a.file])
    print(f"\nSpartiti in: {Path(a.uscita).resolve()}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
