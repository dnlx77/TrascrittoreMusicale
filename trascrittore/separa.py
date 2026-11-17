"""Separazione voce / base strumentale con un modello MDX-Net (Kim_Vocal_2, progetto UVR)."""
import logging
from pathlib import Path

MODELLO = "Kim_Vocal_2.onnx"


def separa(audio, nome, cartella_stems, cartella_modelli):
    cartella_stems = Path(cartella_stems)
    voc = cartella_stems / f"{nome}_vocals.wav"
    ins = cartella_stems / f"{nome}_instrumental.wav"
    if voc.exists() and ins.exists():
        return voc, ins  # già fatto in precedenza
    from audio_separator.separator import Separator

    cartella_stems.mkdir(parents=True, exist_ok=True)
    sep = Separator(log_level=logging.WARNING, model_file_dir=str(cartella_modelli),
                    output_dir=str(cartella_stems), output_format="WAV")
    sep.load_model(model_filename=MODELLO)
    sep.separate(str(audio), {"Vocals": f"{nome}_vocals", "Instrumental": f"{nome}_instrumental"})
    return voc, ins
