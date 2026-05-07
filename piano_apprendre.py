"""
Piano Learner — Apprends une melodie note par note ou accord par accord
=======================================================================
Dependances : pip install sounddevice numpy scipy
"""

import os
import json
import time
import subprocess
from datetime import datetime
import scipy.io.wavfile as wavfile

import tempfile
import sounddevice as sd
import numpy as np
import librosa
import music21
from scipy import stats
from scipy.ndimage import median_filter
import yt_dlp
from basic_pitch.inference import predict, Model
from basic_pitch import ICASSP_2022_MODEL_PATH


SAMPLE_RATE    = 44100
DOSSIER        = "melodies"
os.makedirs(DOSSIER, exist_ok=True)


# ── Accords connus (noms sans octave, en ordre alphabetique) ─────────────
NOMS_ACCORDS = {
    ("Do",  "Mi",  "Sol"):        "Do majeur",
    ("Do",  "Mi",  "Sol", "Si"):  "Do maj7",
    ("Do",  "Re#", "Sol"):        "Do mineur",
    ("Re",  "Fa#", "La"):         "Re majeur",
    ("Re",  "Fa",  "La"):         "Re mineur",
    ("Mi",  "Sol#","Si"):         "Mi majeur",
    ("Mi",  "Sol", "Si"):         "Mi mineur",
    ("Fa",  "La",  "Do"):         "Fa majeur",
    ("Fa",  "La#", "Do"):         "Fa mineur",
    ("Sol", "Si",  "Re"):         "Sol majeur",
    ("Sol", "La#", "Re"):         "Sol mineur",
    ("La",  "Do#", "Mi"):         "La majeur",
    ("La",  "Do",  "Mi"):         "La mineur",
    ("Si",  "Re#", "Fa#"):        "Si majeur",
    ("Si",  "Re",  "Fa#"):        "Si mineur",
}

NOMS_NOTES = ["Do", "Do#", "Re", "Re#", "Mi", "Fa", "Fa#", "Sol", "Sol#", "La", "La#", "Si"]

# Conversion notation internationale (C, D, E...) → française (Do, Re, Mi...)
_INTL_VERS_FR = {
    "C": "Do", "C#": "Do#", "D": "Re", "D#": "Re#", "E": "Mi",
    "F": "Fa", "F#": "Fa#", "G": "Sol", "G#": "Sol#", "A": "La",
    "A#": "La#", "B": "Si",
    "Db": "Do#", "Eb": "Re#", "Gb": "Fa#", "Ab": "Sol#", "Bb": "La#",
}

def _note_intl_vers_fr(nom):
    """'C4' -> 'Do4', 'F#3' -> 'Fa#3'. Retourne None si inconnu."""
    if not nom:
        return None
    octave = nom[-1] if nom[-1].isdigit() else ""
    base   = nom[:-1] if octave else nom
    fr     = _INTL_VERS_FR.get(base)
    if fr:
        return f"{fr}{octave}"
    return nom if base in NOMS_NOTES else None

CONSEILS = {
    "Do":  "1ere blanche apres le groupe de 2 noires",
    "Re":  "2e blanche dans le groupe de 2 noires",
    "Mi":  "3e blanche, juste avant les 3 noires",
    "Fa":  "1ere blanche du groupe de 3 noires",
    "Sol": "2e blanche du groupe de 3 noires",
    "La":  "3e blanche du groupe de 3 noires",
    "Si":  "Derniere blanche avant le Do suivant",
    "Do#": "1ere noire du groupe de 2",
    "Re#": "2e noire du groupe de 2",
    "Fa#": "1ere noire du groupe de 3",
    "Sol#":"2e noire du groupe de 3",
    "La#": "3e noire du groupe de 3",
}


# ═══════════════════════════════════════════════════════════════
# AFFICHAGE CLAVIER ASCII
# ═══════════════════════════════════════════════════════════════

def afficher_clavier(notes_cibles):
    """
    Affiche une octave de piano ASCII.
    notes_cibles : liste de notes a surligner (ex: ["Do4", "Mi4", "Sol4"])
    """
    if isinstance(notes_cibles, str):
        notes_cibles = [notes_cibles]

    # Normaliser les noms pour la comparaison
    noms_cibles  = {n[:-1].replace("e", "e") for n in notes_cibles}
    est_noire    = {n for n in noms_cibles if "#" in n}
    est_blanche  = {n for n in noms_cibles if "#" not in n}

    blanches = ["Do", "Re", "Mi", "Fa", "Sol", "La", "Si"]
    noires   = ["Do#", "Re#", None, "Fa#", "Sol#", "La#", None]
    W = 6

    # Nom de l'accord si plusieurs notes
    label = " + ".join(notes_cibles)
    if len(notes_cibles) > 1:
        cle = tuple(sorted(n[:-1] for n in notes_cibles))
        label += f"  →  {NOMS_ACCORDS.get(cle, 'accord')}"

    print(f"\n  {label}")
    print()

    # ── Touches noires ──
    l1 = l2 = l3 = "  "
    for i, tb in enumerate(blanches):
        tn = noires[i]
        actif = tn is not None and tn in est_noire
        if tn is None:
            l1 += " " * W; l2 += " " * W; l3 += " " * W
        elif actif:
            l1 += "  [***]"[0:W]; l2 += f"  {tn:^4}"[0:W]; l3 += "  [***]"[0:W]
        else:
            l1 += "  |---|"[0:W]; l2 += "  |   |"[0:W]; l3 += "  |___|"[0:W]

    print(l1); print(l2); print(l3)
    print("  " + "+-----" * len(blanches) + "+")

    # ── Corps touches blanches ──
    corps = "  "
    for tb in blanches:
        corps += "| ### " if tb in est_blanche else "|     "
    print(corps + "|")

    # ── Noms touches blanches ──
    noms_ligne = "  "
    for tb in blanches:
        noms_ligne += f"|>{tb:<4}" if tb in est_blanche else f"| {tb:<4}"
    print(noms_ligne + "|")

    print("  " + "+-----" * len(blanches) + "+")

    # ── Fleche sous la touche blanche (premiere si plusieurs) ──
    if est_blanche:
        premiere = next(tb for tb in blanches if tb in est_blanche)
        idx = blanches.index(premiere)
        print("  " + " " * (idx * W + 2) + " ^^^")

    # ── Conseil ──
    for nom in sorted(noms_cibles):
        conseil = CONSEILS.get(nom, "")
        if conseil:
            print(f"  {nom} : {conseil}")

    print()


# ═══════════════════════════════════════════════════════════════
# AUDIO — SYNTHESE (lecture) ET DETECTION (micro)
# ═══════════════════════════════════════════════════════════════

def note_vers_freq(note):
    """Ex: 'Do4' -> 261.63 Hz"""
    nom = note[:-1]
    octave = int(note[-1])
    if nom not in NOMS_NOTES:
        return None
    idx = NOMS_NOTES.index(nom)
    demi_tons = (octave - 4) * 12 + (idx - 9)
    return 440.0 * (2 ** (demi_tons / 12))


def freq_vers_note(freq):
    """Ex: 261.63 Hz -> 'Do4'"""
    if freq < 60 or freq > 5000:
        return None
    demi_tons = round(12 * np.log2(freq / 440.0))
    idx = (demi_tons + 9) % 12
    octave = 4 + (demi_tons + 9) // 12
    return f"{NOMS_NOTES[idx]}{octave}"


def midi_vers_note(midi):
    """Ex: 60 -> 'Do4'"""
    octave = (midi // 12) - 1
    idx = midi % 12
    if 0 <= idx < len(NOMS_NOTES) and 2 <= octave <= 8:
        return f"{NOMS_NOTES[idx]}{octave}"
    return None


# ── FluidSynth : vrai son de piano via SoundFont ─────────────
import fluidsynth as fs

# Emplacements possibles du SoundFont selon Mac Intel ou Apple Silicon
_SOUNDFONT_PATHS = [
    os.path.join(os.path.dirname(__file__), "piano.sf3"),   # FluidR3 GM dans le projet
    "/opt/homebrew/Cellar/fluid-synth/2.5.4/share/fluid-synth/sf2/VintageDreamsWaves-v2.sf2",
]

def _trouver_soundfont():
    for p in _SOUNDFONT_PATHS:
        if os.path.exists(p):
            return p
    return None

_sf_path = _trouver_soundfont()
if not _sf_path:
    print("SoundFont introuvable.")
    print("Installe-le avec : brew install fluid-soundfont-gm")
    raise SystemExit(1)

_synth = fs.Synth(gain=0.8)
_synth.start(driver="coreaudio")   # driver audio natif Mac
_sfid  = _synth.sfload(_sf_path)
_synth.program_select(0, _sfid, 0, 0)   # canal 0, Grand Piano

# Pré-charger le modèle Basic Pitch une seule fois au démarrage
print("Chargement du modèle Basic Pitch...")
_bp_model = Model(ICASSP_2022_MODEL_PATH)
print("Modèle prêt.")


def _note_vers_midi(note):
    """Ex: 'Do4' -> 60"""
    nom    = note[:-1]
    octave = int(note[-1])
    if nom not in NOMS_NOTES:
        return None
    return 12 * (octave + 1) + NOMS_NOTES.index(nom)


def jouer_accord(notes, duree=0.2, volume=100):
    """Joue un accord via FluidSynth — vrai son de piano."""
    if isinstance(notes, str):
        notes = [notes]

    midis = [m for n in notes if (m := _note_vers_midi(n)) is not None]
    for m in midis:
        _synth.noteon(0, m, volume)

    time.sleep(duree)

    for m in midis:
        _synth.noteoff(0, m)


def jouer_sequence(sequence, pauses=None, durees=None, pause=0.15, verifier=False):
    """Joue une sequence fluide. Si verifier=True, écoute le micro en parallèle."""
    import threading

    suspects = []  # indices des notes qui semblent fausses

    for i, element in enumerate(sequence):
        d = durees[i] if durees and i < len(durees) else 0.25
        d = max(0.1, min(1.5, d))
        p = pauses[i] if pauses and i < len(pauses) else pause
        p = max(0.05, min(1.0, p))
        intervalle = max(d, p)

        midis = [m for n in (element if isinstance(element, list) else [element])
                 if (m := _note_vers_midi(n)) is not None]
        for m in midis:
            _synth.noteon(0, m, 100)

        if verifier:
            # Enregistrer ce qui sort pendant la durée de la note
            duree_enreg = max(0.15, min(d, 0.4))
            buf = []
            done = threading.Event()

            def _cb(indata, *_):
                buf.append(indata[:, 0].copy())
                if sum(len(b) for b in buf) >= int(duree_enreg * SAMPLE_RATE):
                    done.set()

            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                                blocksize=512, callback=_cb, dtype="float32"):
                done.wait(timeout=duree_enreg + 0.2)

            audio_enreg = np.concatenate(buf) if buf else np.array([], dtype=np.float32)
            if len(audio_enreg) > 0:
                entendues = detecter_accord_audio(audio_enreg)
                attendues = {n[:-1] for n in element}
                entendues_base = {n[:-1] for n in entendues}
                # Fausse note si aucune note attendue n'est entendue
                if entendues and not attendues & entendues_base:
                    suspects.append((i, element, entendues))

        time.sleep(max(0.05, intervalle - 0.02))
        for m in midis:
            _synth.noteoff(0, m)
        time.sleep(0.02)

    return suspects


def detecter_accord_audio(audio, sr=SAMPLE_RATE):
    """
    Détecte les notes/accords dans un chunk audio via Basic Pitch (Spotify).
    Beaucoup plus précis que librosa pour le piano.
    """
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 0.004:
        return []

    try:
        # Sauvegarder en WAV temporaire (Basic Pitch travaille sur des fichiers)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            tmp_path = f.name
        audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        wavfile.write(tmp_path, sr, audio_int16)

        _, _, note_events = predict(tmp_path, _bp_model)
        os.unlink(tmp_path)

        if not note_events:
            return []

        # Convertir les pitches MIDI en noms de notes français (sans doublons)
        notes = []
        noms_vus = set()
        for event in note_events:
            note_name = midi_vers_note(int(event[2]))
            nom_base  = note_name[:-1] if note_name else None
            if note_name and nom_base not in noms_vus:
                notes.append(note_name)
                noms_vus.add(nom_base)

        return notes

    except Exception:
        return []


def ecouter_accord(duree_attendue=None):
    """
    Attend qu'une note soit jouee, enregistre et retourne (notes, duree_tenue).
    Si duree_attendue > 0.6s, attend que la note soit relâchée pour mesurer la durée.
    """
    SEUIL_ONSET   = 0.015
    SEUIL_SILENCE = 0.005   # en dessous = note relâchée
    DUREE_ANALYSE = 0.3     # durée minimale d'enregistrement pour la détection
    DUREE_MAX     = 4.0     # durée max d'attente pour une note longue
    BLOCK         = 1024

    buffer        = [np.array([], dtype=np.float32)]
    onset         = [False]
    enreg         = [np.array([], dtype=np.float32)]
    fini          = [False]
    silence_count = [0]

    attendre_relachement = duree_attendue is not None and duree_attendue > 0.6

    def callback(indata, _f, _t, _s):
        audio = indata[:, 0]
        buffer[0] = np.concatenate([buffer[0], audio])

        if not onset[0]:
            rms = np.sqrt(np.mean(audio ** 2))
            if rms > SEUIL_ONSET:
                onset[0] = True
                pre = int(0.02 * SAMPLE_RATE)
                enreg[0] = buffer[0][-pre:] if len(buffer[0]) > pre else buffer[0].copy()
        else:
            enreg[0] = np.concatenate([enreg[0], audio])
            duree_actuelle = len(enreg[0]) / SAMPLE_RATE

            if attendre_relachement:
                # Attendre que la note soit relâchée (silence) ou timeout
                rms = np.sqrt(np.mean(audio ** 2))
                if rms < SEUIL_SILENCE:
                    silence_count[0] += 1
                else:
                    silence_count[0] = 0
                # 5 blocs consécutifs silencieux (~120ms) = note relâchée
                if silence_count[0] >= 5 or duree_actuelle >= DUREE_MAX:
                    fini[0] = True
            else:
                if duree_actuelle >= DUREE_ANALYSE:
                    fini[0] = True

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        blocksize=BLOCK, callback=callback,
                        dtype="float32"):
        while not fini[0]:
            time.sleep(0.01)

    duree_tenue = len(enreg[0]) / SAMPLE_RATE
    notes = detecter_accord_audio(enreg[0][:int(DUREE_ANALYSE * SAMPLE_RATE)])
    return notes, duree_tenue


def charger_et_analyser_audio(chemin_fichier):
    """
    Charge un fichier audio et détecte la séquence de notes via Basic Pitch.
    Retourne (sequence, confiances, chunks_audio, pauses, durees).
    pauses = intervalles entre notes (rythme).
    durees = durée de tenue de chaque note/accord (en secondes).
    """
    try:
        print("  Analyse Basic Pitch en cours...")
        _, _, note_events = predict(chemin_fichier, _bp_model)

        if not note_events:
            return [], [], [], [], []

        # Regrouper les notes qui commencent dans une fenêtre de 60ms → un accord
        FENETRE = 0.06
        events  = sorted(note_events, key=lambda x: x[0])

        sequence   = []
        confiances = []
        t_debuts   = []
        t_fins     = []
        i = 0
        while i < len(events):
            t_debut   = events[i][0]
            accord    = []
            noms_vus  = set()
            t_fin_max = events[i][1]
            j = i
            while j < len(events) and events[j][0] - t_debut < FENETRE:
                note_name = midi_vers_note(int(events[j][2]))
                nom_base  = note_name[:-1] if note_name else None
                if note_name and nom_base not in noms_vus:
                    accord.append(note_name)
                    noms_vus.add(nom_base)
                t_fin_max = max(t_fin_max, events[j][1])
                j += 1
            if accord:
                sequence.append(accord)
                confiances.append(1.0)
                t_debuts.append(t_debut)
                t_fins.append(t_fin_max)
            i = j

        # Durée de tenue de chaque note
        durees = [round(max(0.1, t_fins[k] - t_debuts[k]), 3) for k in range(len(t_debuts))]

        # Pauses entre les débuts de notes consécutives (rythme)
        pauses = []
        for k in range(len(t_debuts) - 1):
            intervalle = t_debuts[k + 1] - t_debuts[k]
            pauses.append(round(max(0.1, min(3.0, intervalle)), 3))
        pauses.append(0.5)

        # Correction automatique des fausses notes via Ollama
        print("  Correction des fausses notes...")
        sequence = corriger_sequence_ollama(sequence)

        return sequence, confiances, [], pauses, durees

    except Exception as e:
        print(f"  Erreur lors de l'analyse : {e}")
        return [], [], [], [], []


def telecharger_youtube(url):
    """
    Télécharge un audio YouTube et le convertit en WAV.
    Retourne le chemin du fichier WAV.
    """
    print("\n  Téléchargement de l'audio YouTube...")
    
    try:
        # Configuration yt-dlp
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'outtmpl': '%(title)s.%(ext)s',
            'quiet': False,
            'no_warnings': False,
            'progress_hooks': [_afficher_progres_telecharge],
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            fichier_wav = f"{info['title']}.wav"
        
        if os.path.exists(fichier_wav):
            print(f"\n  Téléchargé : {fichier_wav}")
            return fichier_wav
        else:
            # Chercher le fichier avec une extension possible
            titre = info['title']
            for ext in ['.wav', '.m4a', '.mp3', '.flac']:
                chemin_test = titre + ext
                if os.path.exists(chemin_test):
                    print(f"  Trouvé : {chemin_test}")
                    return chemin_test
            
            print(f"  Fichier non trouvé : {fichier_wav}")
            return None
    
    except Exception as e:
        print(f"  Erreur YouTube : {e}")
        return None


def _afficher_progres_telecharge(d):
    """Callback pour afficher la progression du téléchargement."""
    if d['status'] == 'downloading':
        percent = d.get('_percent_str', 'N/A')
        vitesse = d.get('_speed_str', 'N/A')
        print(f"  {percent} à {vitesse}      ", end="\r", flush=True)
    elif d['status'] == 'finished':
        print("  Téléchargement terminé, conversion...    ")



def parse_partition_midi(chemin_fichier):
    """
    Charge un fichier MIDI et retourne une sequence de notes/accords.
    """
    try:
        score = music21.converter.parse(chemin_fichier)
        chordified = score.chordify()
        sequence = []
        for element in chordified.flat.getElementsByClass(['Chord', 'Note']):
            if isinstance(element, music21.chord.Chord):
                notes = []
                for p in element.pitches:
                    nom = p.name.replace('-', '#')
                    notes.append(f"{nom}{p.octave}")
                sequence.append(notes)
            elif isinstance(element, music21.note.Note):
                nom = element.pitch.name.replace('-', '#')
                sequence.append([f"{nom}{element.pitch.octave}"])
        return sequence
    except Exception as e:
        print(f"  Erreur MIDI : {e}")
        return []


def parse_partition_txt(chemin_fichier):
    """
    Charge un fichier texte contenant des notes ou accords.
    Chaque ligne peut être une note (Do4) ou un accord (Do4+Mi4+Sol4).
    """
    sequence = []
    try:
        with open(chemin_fichier, encoding="utf-8") as f:
            for ligne in f:
                ligne = ligne.strip()
                if not ligne or ligne.startswith("#"):
                    continue
                notes = [part.strip().replace('b', '#') for part in ligne.replace('♭', 'b').split('+') if part.strip()]
                if notes:
                    sequence.append(notes)
        return sequence
    except Exception as e:
        print(f"  Erreur texte : {e}")
        return []



OLLAMA_URL          = "http://localhost:11434/api/generate"
OLLAMA_VISION_MODEL = "llava:13b"
OLLAMA_TEXT_MODEL   = "qwen3:30b-a3b"


def corriger_sequence_ollama(sequence):
    """
    Envoie la séquence à Ollama pour corriger les fausses notes musicalement.
    Retourne la séquence corrigée.
    """
    import requests

    seq_str = json.dumps(sequence, ensure_ascii=False)
    prompt = f"""Tu es un expert en musique et en solfège.
Voici une séquence de notes de piano détectée automatiquement. Elle contient peut-être quelques erreurs : mauvaise octave, note hors gamme, note aberrante isolée.

Séquence (format JSON) :
{seq_str}

Corrige UNIQUEMENT les notes clairement fausses (ex: une note dans une octave impossible, une note qui ne s'intègre pas du tout dans la mélodie environnante).
Ne change pas les notes qui semblent correctes.
Conserve exactement le même nombre d'éléments.
Réponds UNIQUEMENT avec le JSON corrigé, sans texte autour. Même format que l'entrée."""

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_TEXT_MODEL, "prompt": prompt, "stream": False},
            timeout=60,
        )
        resp.raise_for_status()
        texte = resp.json().get("response", "").strip()
        debut = texte.find("[")
        fin   = texte.rfind("]") + 1
        if debut >= 0 and fin > debut:
            corrigee = json.loads(texte[debut:fin])
            if len(corrigee) == len(sequence):
                return corrigee
    except Exception as e:
        print(f"  Ollama indisponible : {e}")
    return sequence

def parse_partition_pdf(chemin_fichier):
    """
    Lit une partition PDF page par page via Ollama (modèle vision local).
    Retourne la séquence de notes en notation française.
    Nécessite : ollama pull llava
    """
    import fitz
    import base64
    import requests

    doc = fitz.open(chemin_fichier)
    pages_b64 = []
    for page in doc:
        pix = page.get_pixmap(dpi=150)
        pages_b64.append(base64.standard_b64encode(pix.tobytes("png")).decode())
    doc.close()

    if not pages_b64:
        return []

    # Vérifier que le modèle est disponible
    try:
        modeles = requests.get("http://localhost:11434/api/tags", timeout=5).json()
        noms = [m["name"].split(":")[0] for m in modeles.get("models", [])]
        if OLLAMA_VISION_MODEL not in noms:
            print(f"  Modèle '{OLLAMA_VISION_MODEL}' non installé.")
            print(f"  Lance : ollama pull {OLLAMA_VISION_MODEL}  (~4.7 Go)")
            return []
    except requests.exceptions.ConnectionError:
        print("  Ollama n'est pas lancé. Lance : ollama serve")
        return []

    print(f"  {len(pages_b64)} page(s) — lecture par {OLLAMA_VISION_MODEL}...")

    PROMPT = """Tu es un expert en solfège. Regarde cette partition de piano.
Extrais TOUTES les notes dans l'ordre de lecture (gauche à droite, ligne par ligne).

Réponds UNIQUEMENT avec une liste JSON, sans texte autour. Format :
[["Do4"], ["Mi4", "Sol4"], ["La4"], ...]

Règles :
- Notation française : Do Re Mi Fa Sol La Si (avec octave : Do4, Mi5…)
- Un accord = plusieurs notes dans le même sous-tableau
- Dièse = # (ex: Fa#4), bémol converti en dièse équivalent
- Ne saute aucune note, même les répétitions
- Plusieurs voix simultanées = fusionne en accord"""

    sequence = []
    for i, img_b64 in enumerate(pages_b64):
        if len(pages_b64) > 1:
            print(f"  Page {i+1}/{len(pages_b64)}...")
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={"model": OLLAMA_VISION_MODEL, "prompt": PROMPT,
                      "images": [img_b64], "stream": False},
                timeout=180,
            )
            resp.raise_for_status()
            texte = resp.json().get("response", "").strip()
            debut = texte.find("[")
            fin   = texte.rfind("]") + 1
            if debut >= 0 and fin > debut:
                sequence.extend(json.loads(texte[debut:fin]))
            else:
                print(f"  Page {i+1} : réponse inattendue — {texte[:80]}")
        except Exception as e:
            print(f"  Erreur page {i+1} : {e}")

    return sequence


def mode_pdf_partition():
    """
    Mode : Anton ouvre un PDF de partition et le reproduit au piano.
    Extrait les notes automatiquement (MuseScore), sinon saisie manuelle.
    """
    print("\033[2J\033[H", end="")
    print("━" * 50)
    print("  PARTITION PDF")
    print("━" * 50)

    pdfs = sorted(f for f in os.listdir('.') if f.lower().endswith('.pdf'))
    if not pdfs:
        print("\n  Aucun fichier PDF trouvé dans ce dossier.")
        input("  Entree pour continuer...")
        return

    print("\nPartitions PDF disponibles :")
    print("─" * 40)
    for i, f in enumerate(pdfs, 1):
        print(f"  {i}. {f}")
    print("─" * 40)

    choix = input("Choix (numero) : ").strip()
    if not choix.isdigit() or not (1 <= int(choix) <= len(pdfs)):
        print("Choix invalide.")
        return

    fichier = pdfs[int(choix) - 1]
    chemin  = os.path.abspath(fichier)

    # Ouvrir le PDF pour qu'Anton puisse le voir pendant qu'il joue
    print(f"\n  Ouverture de {fichier}...")
    try:
        subprocess.Popen(["open", chemin])
    except Exception:
        print("  (Impossible d'ouvrir automatiquement — ouvre le PDF toi-même.)")

    # Essayer d'extraire les notes
    print("  Extraction des notes de la partition...")
    sequence = parse_partition_pdf(chemin)

    if sequence:
        print(f"  {len(sequence)} notes extraites automatiquement.")
        print("  Écoute la mélodie extraite...")
        jouer_sequence(sequence)
        ok = input("\n  C'est correct ? (Entree=oui / n=non) : ").strip().lower()
        if ok == "n":
            sequence = []  # On passera à la saisie manuelle ci-dessous

    if not sequence:
        print(f"\n  Extraction automatique impossible.")
        print(f"  Lance d'abord : ollama pull {OLLAMA_VISION_MODEL}")
        print("  Regarde le PDF et entre les notes une à une.")
        print("  Format : Do4        (note seule)")
        print("           Do4+Mi4+Sol4  (accord)")
        print("  Entree vide = terminer la saisie.")
        print("─" * 40)
        sequence = []
        while True:
            ligne = input(f"  Note {len(sequence)+1} : ").strip()
            if not ligne:
                break
            notes = [p.strip() for p in ligne.split('+') if p.strip()]
            if notes:
                sequence.append(notes)

        if not sequence:
            print("  Aucune note entrée.")
            input("  Entree pour continuer...")
            return

        print("\n  Écoute la mélodie saisie...")
        jouer_sequence(sequence)
        ok = input("\n  C'est correct ? (Entree=oui / n=non) : ").strip().lower()
        if ok == "n":
            input("  Annulé. Entree pour continuer...")
            return

    # Sauvegarder la mélodie
    nom = os.path.splitext(fichier)[0]
    melodie = {
        "nom": nom,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "sequence": sequence,
        "source": fichier,
    }
    chemin_json = os.path.join(DOSSIER, f"{nom.replace(' ', '_')}.json")
    with open(chemin_json, "w", encoding="utf-8") as f:
        json.dump(melodie, f, ensure_ascii=False, indent=2)

    print(f"\n  Sauvegardé : {chemin_json}  ({len(sequence)} notes)")
    print("  Va en Mode 3 — Lire la partition — pour commencer !")
    input("  Entree pour continuer...")


def mode_partition():
    """
    Mode : importer une partition MIDI ou un fichier texte de notes.
    """
    print("\033[2J\033[H", end="")
    print("━" * 50)
    print("  ANTON CHARGE UNE PARTITION")
    print("━" * 50)

    fichiers = []
    for f in os.listdir('.'):
        if f.lower().endswith(('.mid', '.midi', '.txt')):
            fichiers.append(f)

    if not fichiers:
        print("\n  Aucun fichier de partition (.mid, .midi, .txt) trouvé dans ce dossier.")
        input("  Entree pour continuer...")
        return

    print("\nFichiers de partition disponibles :")
    print("─" * 40)
    for i, f in enumerate(fichiers, 1):
        print(f"  {i}. {f}")
    print("─" * 40)

    choix = input("Choix (numero) : ").strip()
    if not choix.isdigit() or not (1 <= int(choix) <= len(fichiers)):
        print("Choix invalide.")
        return

    fichier = fichiers[int(choix) - 1]
    extension = os.path.splitext(fichier)[1].lower()

    if extension in ('.mid', '.midi'):
        sequence = parse_partition_midi(fichier)
    else:
        sequence = parse_partition_txt(fichier)

    if not sequence:
        print("  Impossible de charger la partition ou aucune note detectee.")
        input("  Entree pour continuer...")
        return

    print(f"\n  Partition chargee : {len(sequence)} elements")
    if sequence:
        print("  " + " + ".join([" + ".join(n) for n in sequence[:10]]))
    print("\n  Anton peut maintenant pratiquer cette partition.")

    nom = input("\n  Nom de la melodie a sauvegarder : ").strip() or f"partition_{datetime.now().strftime('%H-%M-%S')}"
    melodie = {
        "nom": nom,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "sequence": sequence,
        "source": fichier
    }
    chemin = os.path.join(DOSSIER, f"{nom.replace(' ', '_')}.json")
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(melodie, f, ensure_ascii=False, indent=2)

    print(f"\n  Sauvegarde : {chemin}  ({len(sequence)} notes)")
    print("  Tu peux maintenant aller en Mode 2 pour apprendre cette melodie.")
    input("  Entree pour continuer...")


def mode_analyse_audio():
    """
    Mode 3 : Anton ecoute un fichier audio existant et se corrige lui-meme.
    """
    print("\033[2J\033[H", end="")
    print("━" * 50)
    print("  ANTON ÉCOUTE UN ENREGISTREMENT")
    print("━" * 50)
    
    # Choix : fichier local ou YouTube
    print("\n  Source :")
    print("  1. Fichier WAV local")
    print("  2. Audio YouTube")
    print("━" * 50)
    choix_source = input("  Choix (1 ou 2) : ").strip()
    
    fichier = None
    
    if choix_source == "2":
        # YouTube
        url = input("\n  Colle l'URL YouTube : ").strip()
        if not url:
            print("  URL vide.")
            return
        
        fichier = telecharger_youtube(url)
        if not fichier:
            input("  Entree pour continuer...")
            return
    else:
        # Fichier local
        fichiers_audio = []
        for f in os.listdir("."):
            if f.endswith(".wav"):
                fichiers_audio.append(f)
        
        if not fichiers_audio:
            print("\n  Aucun fichier .wav trouvé dans ce dossier.")
            input("  Entree pour continuer...")
            return
        
        print("\nFichiers audio disponibles :")
        print("─" * 40)
        for i, f in enumerate(fichiers_audio, 1):
            try:
                taux, data = wavfile.read(f)
                duree = len(data) / taux
                print(f"  {i}. {f}  ({duree:.1f}s)")
            except:
                print(f"  {i}. {f}")
        print("─" * 40)
        
        choix = input("Choix (numero) : ").strip()
        if not choix.isdigit() or not (1 <= int(choix) <= len(fichiers_audio)):
            print("Choix invalide.")
            return
        
        fichier = fichiers_audio[int(choix) - 1]
    
    print(f"\n  Chargement et analyse de {fichier}...")
    sequence, confiances, chunks_audio, pauses, durees = charger_et_analyser_audio(fichier)

    if not sequence:
        print("  Aucune note detectee. Le fichier est peut-etre trop silencieux.")
        input("  Entree pour continuer...")
        return

    # Boucle de correction iterative
    iteration = 0
    continuer_correction = True

    while continuer_correction:
        iteration += 1
        print("\033[2J\033[H", end="")
        print("━" * 50)
        if iteration == 1:
            print(f"  Anton analyse l'enregistrement...")
        else:
            print(f"  Anton ré-analyse ({iteration}e tentative)...")
        print("━" * 50)

        print(f"\n  Sequence detectee ({len(sequence)} notes) :")
        print("  " + " + ".join([" + ".join(n) for n in sequence]))

        print("\n  Anton joue ce qu'il a entendu...")
        time.sleep(0.5)
        jouer_sequence(sequence, pauses)
        time.sleep(0.3)
        
        # Correction automatique pour les notes incertaines
        SEUIL_CONFIANCE = 0.15
        incertaines = [i for i, c in enumerate(confiances) if c < SEUIL_CONFIANCE]
        
        if incertaines:
            print(f"\n  Anton n'est pas sur de {len(incertaines)} note(s). Il ré-écoute...")
            time.sleep(0.8)
            corrections = 0
            for i in incertaines:
                print(f"\n  Note {i+1} : {' + '.join(sequence[i])}  (confiance : {confiances[i]:.0%})")
                print("  Anton ré-analyse l'audio...")
                # Re-analyser le chunk audio brut
                if i < len(chunks_audio):
                    nouvelle = detecter_accord_audio(chunks_audio[i])
                    if nouvelle and nouvelle != sequence[i]:
                        print(f"  Correction : {' + '.join(sequence[i])}  →  {' + '.join(nouvelle)}")
                        jouer_accord(nouvelle, duree=0.5)
                        sequence[i] = nouvelle
                        corrections += 1
                    else:
                        print(f"  Confirmation : {' + '.join(sequence[i])}")
                        jouer_accord(sequence[i], duree=0.5)
                time.sleep(0.3)
            
            print(f"\n  Melodie après correction d'Anton ({corrections} changements) :")
            jouer_sequence(sequence, pauses)
            time.sleep(0.3)

            print("\n")
            print("  " + "=" * 46)
            print("  Satisfait de la correction ?")
            print("  " + "=" * 46)
            print("  [Entree]  = Oui, sauvegarder")
            print("  [r]       = Ré-analyser une fois de plus")
            print("  [q]       = Quitter sans sauvegarder")
            print("  " + "=" * 46)
            reponse = input("  Ton choix : ").strip().lower()

            if reponse == "r":
                print("\n  Anton réanalyse plus attentivement...")
                time.sleep(0.5)
                continue
            elif reponse == "q":
                print("  Annulation.")
                input("  Entree pour continuer...")
                return
            else:
                continuer_correction = False
        else:
            print(f"\n  Anton est confiant sur toutes les notes.")
            continuer_correction = False

    # Option de sauvegarde
    print("\n  ─" * 25)
    reponse = input("  Sauvegarder comme melodie ? (o/n) : ").strip().lower()
    if reponse == "o":
        nom = input("  Nom de la melodie : ").strip() or f"audio_{datetime.now().strftime('%H-%M-%S')}"
        melodie = {
            "nom": nom,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "sequence": sequence,
            "pauses":   pauses,
            "durees":   durees,
            "source": fichier
        }
        chemin = os.path.join(DOSSIER, f"{nom.replace(' ', '_')}.json")
        with open(chemin, "w", encoding="utf-8") as f:
            json.dump(melodie, f, ensure_ascii=False, indent=2)
        print(f"\n  Sauvegarde : {chemin}  ({len(sequence)} notes)")
    
    input("  Entree pour continuer...")

def accords_correspondent(joue, attendu):
    """
    Verifie si ce qui a ete joue correspond a ce qui etait attendu.
    Flexible : OK si toutes les notes attendues sont dans ce qui a ete joue.
    """
    noms_attendus = {n[:-1] for n in attendu}
    noms_joues    = {n[:-1] for n in joue}
    return noms_attendus.issubset(noms_joues)


# ═══════════════════════════════════════════════════════════════
# ENREGISTREMENT (Mode 1)
# ═══════════════════════════════════════════════════════════════

def enregistrer_audio(duree_sec):
    """
    Enregistre duree_sec secondes et retourne (sequence, confiances, chunks_audio, audio_brut).
    - confiances   : float 0-1 par element (proportion de frames concordantes)
    - chunks_audio : audio brut (numpy array) par element, pour re-analyse
    - audio_brut   : l'audio complet enregistré
    """
    print(f"\n  Go ! ({duree_sec}s)")
    audio = sd.rec(int(duree_sec * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype="float32")
    for s in range(duree_sec, 0, -1):
        print(f"  {s}s...", end="\r", flush=True)
        time.sleep(1)
    sd.wait()
    print("  Enregistrement termine.              ")

    audio = audio[:, 0]
    audio_brut = audio.copy()  # Garder une copie de l'audio brut

    chunk  = int(0.3 * SAMPLE_RATE)
    pas    = int(0.15 * SAMPLE_RATE)

    # Collecter frames + debut de chaque frame
    frames = []
    for debut in range(0, len(audio) - chunk, pas):
        notes = detecter_accord_audio(audio[debut:debut + chunk])
        frames.append((tuple(sorted(notes)) if notes else None, debut))

    # Regrouper les frames identiques consecutives
    sequence     = []
    confiances   = []
    chunks_audio = []

    courant, compte, debut_segment, total_frames = None, 0, 0, 0
    for f, debut in frames:
        total_frames += 1
        if f == courant:
            compte += 1
        else:
            if courant and compte >= 2:
                sequence.append(list(courant))
                confiances.append(min(1.0, compte / max(total_frames, 1)))
                fin_segment = debut_segment + compte * pas + chunk
                chunks_audio.append(audio[debut_segment:fin_segment])
            courant, compte, debut_segment = f, 1, debut
    if courant and compte >= 2:
        sequence.append(list(courant))
        confiances.append(min(1.0, compte / max(total_frames, 1)))
        fin_segment = debut_segment + compte * pas + chunk
        chunks_audio.append(audio[debut_segment:fin_segment])

    return sequence, confiances, chunks_audio, audio_brut


def mode_enregistrement():
    print("\033[2J\033[H", end="")
    print("━" * 50)
    print("  ENREGISTREMENT")
    print("━" * 50)

    nom = input("\n  Nom de la melodie : ").strip() or f"melodie_{datetime.now().strftime('%H-%M-%S')}"
    duree = input("  Duree en secondes (defaut 30) : ").strip()
    duree = int(duree) if duree.isdigit() else 30

    input(f"\n  Appuie sur Entree puis joue ({duree}s)...")
    sequence, confiances, chunks_audio, audio_brut = enregistrer_audio(duree)

    if not sequence:
        print("\n  Rien detecte. Essaie plus fort ou plus pres du micro.")
        input("  Entree pour continuer...")
        return

    # ── Sauvegarder l'audio en WAV ──
    print("\n  Sauvegarde de l'audio...")
    # Normaliser et convertir en int16
    max_val = np.abs(audio_brut).max()
    if max_val > 0:
        audio_int16 = (audio_brut / max_val * 0.9 * 32767).astype(np.int16)
    else:
        audio_int16 = audio_brut.astype(np.int16)
    
    chemin_wav = f"{nom.replace(' ', '_')}.wav"
    wavfile.write(chemin_wav, SAMPLE_RATE, audio_int16)
    print(f"  Audio : {chemin_wav}")

    # ── Anton vérifie automatiquement les notes incertaines ──
    print("\033[2J\033[H", end="")
    print("━" * 50)
    print(f"  Anton ecoute ce qu'il a detecte...")
    print("━" * 50)
    jouer_sequence(sequence)
    time.sleep(0.5)

    SEUIL_CONFIANCE = 0.15  # en dessous : Anton re-analyse l'audio enregistre
    incertaines = [i for i, c in enumerate(confiances) if c < SEUIL_CONFIANCE]

    if incertaines:
        print(f"\n  Anton n'est pas sur de {len(incertaines)} note(s). Il ré-écoute...")
        time.sleep(0.8)
        for i in incertaines:
            print(f"\n  Note {i+1} : {' + '.join(sequence[i])}  (confiance : {confiances[i]:.0%})")
            print("  Anton re-analyse l'audio enregistre...")
            # Re-analyser le chunk audio brut avec un seuil plus strict
            nouvelle = detecter_accord_audio(chunks_audio[i])
            if nouvelle and nouvelle != sequence[i]:
                print(f"  Correction : {' + '.join(sequence[i])}  →  {' + '.join(nouvelle)}")
                jouer_accord(nouvelle, duree=0.5)
                sequence[i] = nouvelle
            else:
                print(f"  Confirmation : {' + '.join(sequence[i])}")
                jouer_accord(sequence[i], duree=0.5)
            time.sleep(0.3)

        print("\n  Melodie apres correction d'Anton :")
        jouer_sequence(sequence)
        time.sleep(0.3)
    else:
        print(f"\n  Anton est confiant sur toutes les notes.")

    # Sauvegarder
    melodie = {"nom": nom, "date": datetime.now().strftime("%Y-%m-%d %H:%M"), "sequence": sequence}
    chemin = os.path.join(DOSSIER, f"{nom.replace(' ', '_')}.json")
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(melodie, f, ensure_ascii=False, indent=2)
    print(f"\n  Sauvegarde : {chemin}  ({len(sequence)} notes)")
    input("  Entree pour continuer...")


# ═══════════════════════════════════════════════════════════════
# APPRENTISSAGE (Mode 2)
# ═══════════════════════════════════════════════════════════════

def charger_melodie():
    fichiers = sorted([f for f in os.listdir(DOSSIER) if f.endswith(".json")])
    if not fichiers:
        print("  Aucune melodie. Commence par le Mode 1.")
        return None

    print("\nMelodies disponibles :")
    print("─" * 40)
    for i, f in enumerate(fichiers, 1):
        with open(os.path.join(DOSSIER, f), encoding="utf-8") as fp:
            d = json.load(fp)
        seq = d.get("sequence") or [[n] for n in d.get("notes", [])]
        print(f"  {i}. {d['nom']}  ({len(seq)} elements)  — {d.get('date', '')}")
    print("─" * 40)

    choix = input("Choix (numero) : ").strip()
    if not choix.isdigit() or not (1 <= int(choix) <= len(fichiers)):
        print("Choix invalide.")
        return None

    with open(os.path.join(DOSSIER, fichiers[int(choix)-1]), encoding="utf-8") as f:
        d = json.load(f)
    d["sequence"] = d.get("sequence") or [[n] for n in d.get("notes", [])]
    d["pauses"]   = d.get("pauses", [])
    d["durees"]   = d.get("durees", [])
    return d


def afficher_statut(i, total):
    """Affiche l'etat global de progression en haut de l'ecran."""
    print("\033[2J\033[H", end="")
    print("━" * 50)
    print(f"  PIANO LEARNER  —  {i+1}/{total}")
    print("━" * 50)

    # Progression visuelle
    barres = ""
    for j in range(total):
        if j < i:        barres += "█"
        elif j == i:     barres += "▶"
        else:            barres += "░"
    print(f"  [{barres}]")
    print()


ENCOURAGEMENTS_OK  = ["Super !", "Bravo !", "Parfait !", "Ouais !", "C'est ca !"]
ENCOURAGEMENTS_NON = ["Presque !", "Encore un essai !", "Tu y es presque !", "Essaie encore !"]

import random

def jouer_victoire():
    """Petite fanfare de victoire."""
    fanfare = [["Do4", "Mi4", "Sol4"], ["Do5"]]
    for notes in fanfare:
        jouer_accord(notes, duree=0.25)
        time.sleep(0.05)


def afficher_partition_visuelle(sequence, index_courant):
    """
    Affiche la partition comme une ligne de notes, la note courante encadrée.
    Montre une fenêtre glissante centrée sur la note à jouer.
    """
    FENETRE = 9   # notes visibles au total dans la fenêtre

    debut = max(0, index_courant - 3)
    fin   = min(len(sequence), debut + FENETRE)
    if fin - debut < FENETRE:
        debut = max(0, fin - FENETRE)

    # ── Ligne du haut : numéros ──
    ligne_num   = "  "
    ligne_notes = "  "
    ligne_fleche= "  "

    for i in range(debut, fin):
        noms  = "+".join(n[:-1] for n in sequence[i])
        largeur = max(len(noms) + 2, 7)

        num_str = str(i + 1)
        if i == index_courant:
            cell = f"[{noms}]"
            ligne_fleche += ("▲").center(largeur)
        else:
            cell = f" {noms} "
            ligne_fleche += " " * largeur

        ligne_num   += num_str.center(largeur)
        ligne_notes += cell.center(largeur)

    # Indicateurs de défilement
    prefix = "◀ " if debut > 0 else "  "
    suffix = " ▶" if fin < len(sequence) else ""

    print(f"\n{prefix}{ligne_num.strip()}{suffix}")
    print(f"  {ligne_notes.strip()}")
    print(f"  {ligne_fleche.strip()}")
    print(f"\n  Note {index_courant + 1} / {len(sequence)}")

    # Barre de progression
    barres = ""
    for j in range(len(sequence)):
        if j < index_courant:   barres += "█"
        elif j == index_courant: barres += "▶"
        else:                    barres += "░"
    print(f"  [{barres}]")


def mode_lire_partition():
    """
    Mode : Anton lit la partition complète et reproduit chaque note.
    La partition défile en haut — comme une vraie feuille de musique.
    """
    print("\033[2J\033[H", end="")
    print("━" * 50)
    print("  LIRE LA PARTITION")
    print("━" * 50)

    melodie = charger_melodie()
    if not melodie:
        return

    sequence = melodie["sequence"]
    pauses   = melodie.get("pauses", [])
    durees   = melodie.get("durees", [])
    total    = len(sequence)

    # Écoute préalable
    print("\033[2J\033[H", end="")
    print("━" * 50)
    print(f"  {melodie['nom']}")
    print("━" * 50)
    print(f"\n  [Entree] écouter la mélodie  [s] passer directement")
    if input("  : ").strip().lower() != "s":
        jouer_sequence(sequence, pauses, durees)
        ok = input("\n  C'est bien ca ? (Entree=oui / n=non) : ").strip().lower()
        if ok == "n":
            input("  Entree...")
            return

    score    = 0
    MAX_ESSAIS = 5

    for i, element in enumerate(sequence):
        reussi       = False
        premier_coup = True
        pause_apres  = pauses[i] if pauses and i < len(pauses) else 0.4
        duree_att    = durees[i] if durees and i < len(durees) else 0.3
        note_longue  = duree_att > 0.6

        for essai in range(MAX_ESSAIS):
            print("\033[2J\033[H", end="")
            print("━" * 50)
            print(f"  {melodie['nom']}")
            print("━" * 50)

            afficher_partition_visuelle(sequence, i)
            afficher_clavier(element)

            if note_longue:
                print(f"  🎵 Note longue — tiens bien !  (essai {essai+1}/{MAX_ESSAIS})")
            else:
                print(f"  Joue quand tu es prêt...  (essai {essai+1}/{MAX_ESSAIS})")

            joue, duree_jouee = ecouter_accord(duree_att)

            if accords_correspondent(joue, element):
                # Vérifier la durée si note longue
                if note_longue and duree_jouee < duree_att * 0.6:
                    print(f"\n  Bonne note ! Mais tiens-la plus longtemps ({duree_att:.1f}s)")
                    jouer_accord(element, duree=min(duree_att, 1.5))
                    time.sleep(0.2)
                else:
                    msg = random.choice(ENCOURAGEMENTS_OK)
                    print(f"\n  {msg}")
                    jouer_accord(element, duree=min(duree_att, 0.8))
                    if premier_coup:
                        score += 1
                    reussi = True
                    time.sleep(pause_apres)
                    break
            else:
                premier_coup = False
                label_joue   = " + ".join(joue) if joue else "rien"
                print(f"\n  Entendu : {label_joue}")
                print(f"  Attendu : {' + '.join(element)}")
                print(f"  {random.choice(ENCOURAGEMENTS_NON)}")
                jouer_accord(element, duree=0.5)
                time.sleep(0.3)
                print("  [Entree] réessayer  [r] ré-écouter cette note  [t] toute la mélodie")
                r = input("  : ").strip().lower()
                if r == "r":
                    print("  Écoute bien...")
                    jouer_accord(element, duree=0.8)
                    time.sleep(0.2)
                elif r == "t":
                    print("  Écoute bien...")
                    jouer_sequence(sequence[:i + 1], pauses[:i + 1] if pauses else None, durees[:i + 1] if durees else None)
                    time.sleep(0.3)

        if not reussi:
            print(f"\n  On passe. C'était : {' + '.join(element)}")
            jouer_accord(element, duree=0.5)
            time.sleep(0.8)

    # Fin
    print("\033[2J\033[H", end="")
    print("━" * 50)
    print(f"  FIN — {melodie['nom']}")
    print("━" * 50)

    pct = int(score / total * 100)
    if pct == 100:
        bilan = "Parfait ! Tu as tout réussi du premier coup !"
    elif pct >= 70:
        bilan = "Très bien ! Encore un peu et ce sera parfait !"
    elif pct >= 40:
        bilan = "Pas mal ! Continue à t'entraîner !"
    else:
        bilan = "Courage ! Ça va venir avec la pratique !"

    print(f"\n  Score : {score}/{total} notes réussies du premier coup  ({pct}%)")
    print(f"  {bilan}")
    print()

    jouer_victoire()
    time.sleep(0.3)
    jouer_sequence(sequence)
    input("\n  Entree pour revenir au menu...")


def mode_apprentissage():
    print("\n" + "━" * 50)
    print("  APPRENTISSAGE")
    print("━" * 50)

    melodie = charger_melodie()
    if not melodie:
        return

    sequence = melodie["sequence"]
    pauses   = melodie.get("pauses", [])
    durees   = melodie.get("durees", [])
    total    = len(sequence)

    # Ecouter la melodie d'abord
    print("\033[2J\033[H", end="")
    print(f"  {melodie['nom']}")
    print(f"\n  [Entree] écouter la mélodie  [s] passer directement")
    if input("  : ").strip().lower() != "s":
        jouer_sequence(sequence, pauses, durees)
        ok = input("\n  C'est bien ca ? (Entree=oui / n=non) : ").strip().lower()
        if ok == "n":
            print("  Retourne en Mode 1 pour re-enregistrer.")
            input("  Entree...")
            return

    apprises  = []
    score     = 0

    for i, element in enumerate(sequence):
        MAX_ESSAIS   = 5
        premier_coup = True
        pause_apres  = pauses[i] if pauses and i < len(pauses) else 0.4
        duree_att    = durees[i] if durees and i < len(durees) else 0.3
        note_longue  = duree_att > 0.6

        # ── Apprendre l'element seul ──
        reussi = False
        for essai in range(MAX_ESSAIS):
            afficher_statut(i, total)
            afficher_clavier(element)
            if note_longue:
                print(f"  🎵 Note longue — tiens bien !  (essai {essai+1}/{MAX_ESSAIS})")
            else:
                print(f"  Joue quand tu es prete...  (essai {essai+1}/{MAX_ESSAIS})")

            joue, duree_jouee = ecouter_accord(duree_att)

            if accords_correspondent(joue, element):
                if note_longue and duree_jouee < duree_att * 0.6:
                    print(f"\n  Bonne note ! Mais tiens-la plus longtemps ({duree_att:.1f}s)")
                    jouer_accord(element, duree=min(duree_att, 1.5))
                    time.sleep(0.2)
                else:
                    msg = random.choice(ENCOURAGEMENTS_OK)
                    print(f"\n  {msg}")
                    jouer_accord(element, duree=min(duree_att, 0.8))
                    if premier_coup:
                        score += 1
                    reussi = True
                    time.sleep(pause_apres)
                    break
            else:
                premier_coup = False
                label_joue = " + ".join(joue) if joue else "rien"
                print(f"\n  Entendu : {label_joue}")
                print(f"  Attendu : {' + '.join(element)}")
                print(f"  {random.choice(ENCOURAGEMENTS_NON)}")
                jouer_accord(element, duree=0.5)
                time.sleep(0.3)
                print("  [Entree] réessayer  [r] ré-écouter cette note  [t] toute la mélodie")
                r = input("  : ").strip().lower()
                if r == "r":
                    print("  Écoute bien...")
                    jouer_accord(element, duree=0.8)
                    time.sleep(0.2)
                elif r == "t":
                    print("  Écoute bien...")
                    jouer_sequence(sequence[:i + 1], pauses[:i + 1] if pauses else None, durees[:i + 1] if durees else None)
                    time.sleep(0.3)

        if not reussi:
            print(f"\n  On passe. C'etait : {' + '.join(element)}")
            jouer_accord(element, duree=0.5)
            time.sleep(0.8)

        apprises.append(element)

        # ── Rejouer depuis le debut ──
        if len(apprises) > 1:
            print("\033[2J\033[H", end="")
            print("━" * 50)
            print(f"  Rejoue depuis le debut ({len(apprises)} elements) !")
            print("━" * 50)
            jouer_sequence(apprises, pauses[:len(apprises)] if pauses else None, durees[:len(apprises)] if durees else None)
            time.sleep(0.3)

            for j, el_seq in enumerate(apprises):
                afficher_statut(j, len(apprises))
                afficher_clavier(el_seq)
                print("  Joue...")
                joue_seq, _ = ecouter_accord()
                if accords_correspondent(joue_seq, el_seq):
                    print(f"  {random.choice(ENCOURAGEMENTS_OK)}")
                    time.sleep(0.2)
                else:
                    print(f"\n  Hmm, cette note ne sonne pas juste...")
                    jouer_accord(el_seq, duree=0.8)
                    print("  [Entree] continuer  [t] ré-écouter toute la mélodie")
                    r = input("  : ").strip().lower()
                    if r == "t":
                        jouer_sequence(sequence, pauses, durees)
                        time.sleep(0.3)
                    break

    # ── Fin ──
    print("\033[2J\033[H", end="")
    print("━" * 50)
    print(f"  FIN — {melodie['nom']}")
    print("━" * 50)

    # Score
    pct = int(score / total * 100)
    if pct == 100:
        bilan = "Parfait ! Tu as tout reussi du premier coup !"
    elif pct >= 70:
        bilan = "Tres bien ! Encore un peu de pratique et ce sera parfait !"
    elif pct >= 40:
        bilan = "Pas mal ! Continue a t'entrainer !"
    else:
        bilan = "Courage ! Ca va venir avec la pratique !"

    print(f"\n  Score : {score}/{total} notes reussies du premier coup  ({pct}%)")
    print(f"  {bilan}")
    print()

    jouer_victoire()
    time.sleep(0.3)
    jouer_sequence(sequence)
    input("\n  Entree pour revenir au menu...")


# ═══════════════════════════════════════════════════════════════
# MENU
# ═══════════════════════════════════════════════════════════════

def main():
    while True:
        print("\033[2J\033[H", end="")
        print("━" * 50)
        print("  PIANO LEARNER")
        print("━" * 50)
        print("  1. Lire une partition PDF  ← NOUVEAU")
        print("  2. Lire la partition (melodie sauvegardee)")
        print("  3. Enregistrer une melodie")
        print("  4. Apprendre une melodie")
        print("  5. Anton ecoute un fichier audio")
        print("  6. Anton ecoute YouTube")
        print("  7. Charger une partition (.mid/.txt)")
        print("  8. Quitter")
        print("━" * 50)
        choix = input("  Choix : ").strip()
        if choix == "1":
            mode_pdf_partition()
        elif choix == "2":
            mode_lire_partition()
        elif choix == "3":
            mode_enregistrement()
        elif choix == "4":
            mode_apprentissage()
        elif choix == "5":
            mode_analyse_audio()
        elif choix == "6":
            mode_youtube()
        elif choix == "7":
            mode_partition()
        elif choix == "8":
            break


def mode_youtube():
    """
    Mode 4 : Anton télécharge et écoute une vidéo YouTube directement.
    """
    print("\033[2J\033[H", end="")
    print("━" * 50)
    print("  ANTON ÉCOUTE YOUTUBE")
    print("━" * 50)
    
    url = input("\n  Colle l'URL YouTube : ").strip()
    if not url:
        print("  URL vide.")
        input("  Entree pour continuer...")
        return
    
    fichier = telecharger_youtube(url)
    if not fichier:
        input("  Entree pour continuer...")
        return
    
    print(f"\n  Chargement et analyse...")
    sequence, confiances, chunks_audio, pauses, durees = charger_et_analyser_audio(fichier)

    if not sequence:
        print("  Aucune note detectee. Le fichier est peut-etre trop silencieux.")
        input("  Entree pour continuer...")
        return

    # Boucle de correction iterative
    iteration = 0
    continuer_correction = True

    while continuer_correction:
        iteration += 1
        print("\033[2J\033[H", end="")
        print("━" * 50)
        if iteration == 1:
            print(f"  Anton analyse la video YouTube...")
        else:
            print(f"  Anton ré-analyse ({iteration}e tentative)...")
        print("━" * 50)

        print(f"\n  Sequence detectee ({len(sequence)} notes) :")
        print("  " + " + ".join([" + ".join(n) for n in sequence]))

        print("\n  Anton joue ce qu'il a entendu...")
        time.sleep(0.5)
        jouer_sequence(sequence, pauses)
        time.sleep(0.3)

        print(f"\n  Anton est confiant sur toutes les notes.")
        continuer_correction = False

    # Option de sauvegarde
    print("\n  ─" * 25)
    reponse = input("  Sauvegarder comme melodie ? (o/n) : ").strip().lower()
    if reponse == "o":
        nom = input("  Nom de la melodie : ").strip() or f"youtube_{datetime.now().strftime('%H-%M-%S')}"
        melodie = {
            "nom": nom,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "sequence": sequence,
            "pauses":   pauses,
            "durees":   durees,
            "source": f"YouTube: {fichier}"
        }
        chemin = os.path.join(DOSSIER, f"{nom.replace(' ', '_')}.json")
        with open(chemin, "w", encoding="utf-8") as f:
            json.dump(melodie, f, ensure_ascii=False, indent=2)
        print(f"\n  Sauvegarde : {chemin}  ({len(sequence)} notes)")
    
    input("  Entree pour continuer...")





if __name__ == "__main__":
    main()
