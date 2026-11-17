# Trascrittore

Da un brano audio (anche una canzone completa, con voce e band) a **spartiti per pianoforte suonabili**, in tre livelli:

| Livello | Mano destra | Mano sinistra | Tonalità |
|---|---|---|---|
| principiante | melodia a semiminime/minime, senza sincopi | una nota lunga per battuta (fondamentale) | trasportata, max 1 alterazione |
| intermedio | melodia completa a crome | fondamentale–quinta–ottava–quinta | max 3 alterazioni |
| avanzato | melodia armonizzata (bicordi/accordi) | arpeggi a crome con pedale | originale |

Per ogni brano produce: PDF, MusicXML (si apre e si corregge in MuseScore), MIDI, un lead sheet
(melodia + sigle) e `controllo.json` con le verifiche di suonabilità.

## Installazione

Serve **Python 3.10 o più recente** e **ffmpeg**.

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate

pip install git+https://github.com/dnlx77/TrascrittoreMusicale
# oppure, da una copia locale:  git clone https://github.com/dnlx77/TrascrittoreMusicale && pip install ./TrascrittoreMusicale
```

Per il PDF serve la libreria grafica Cairo (`brew install cairo` su macOS, `apt install libcairo2` su Linux;
su Windows si installa con GTK). Senza Cairo il programma salva le pagine in SVG e funziona tutto il resto.

## Uso

```bash
trascrivi brano.mp3                          # tutti e tre i livelli, in ./spartiti/<brano>/
trascrivi brano.mp3 --livello principiante   # un solo livello
trascrivi a.mp3 b.wav --uscita miei_spartiti --audio   # più brani + anteprima audio mp3
```

La prima esecuzione scarica il modello di separazione della voce (~70 MB). Un brano di 4 minuti richiede
circa 4 minuti su un normale computer; le tracce separate restano in cache (`~/.cache/trascrittore`),
quindi rigenerare gli spartiti dello stesso brano è molto più veloce.

## Come funziona

1. **Separazione** (`separa.py`): voce e base strumentale con il modello MDX-Net *Kim_Vocal_2*.
2. **Analisi** (`analizza.py`): tempo e battiti, tonalità, melodia dalla voce (pYIN), basso dalla base filtrata, accordi.
3. **Scheletro** (`scheletro.py`): tempo corretto, accordi per mezza battuta guidati dalla tonalità, melodia
   ridotta a crome, basso, sezioni del brano, lead sheet.
4. **Arrangiamento** (`arrangia.py`): mano destra e sinistra per livello, trasporto, controllo che le mani non
   si incrocino e che nessun accordo superi l'ottava.

## Limiti noti

- Gli accordi sono solo maggiori/minori e qualche volta sbagliati; la melodia può perdere o aggiungere note.
- Metro sempre 4/4; le intro a tempo libero escono confuse.
- Le anteprime audio usano un sintetizzatore semplice, non un vero pianoforte.
