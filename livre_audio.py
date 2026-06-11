#!/usr/bin/env python3
"""
Convertisseur Livre -> Audio IA  (100% gratuit, 100% local)
------------------------------------------------------------
Convertit un PDF ou EPUB en fichiers audio par chapitre.
Si le livre est en anglais il est traduit en français avec Ollama.

Vitesse : traduction et génération audio tournent en parallèle.

Utilisation :
    python3 livre_audio.py mon_livre.epub
    python3 livre_audio.py roman_anglais.pdf --voix remy
    python3 livre_audio.py mon_livre.epub --modele llama3.2
"""

import os
import re
import sys
import asyncio
import argparse
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import fitz
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from langdetect import detect, LangDetectException
import edge_tts
import ollama

# -----------------------------------------------------------------------
# Constantes
# -----------------------------------------------------------------------
MODELE_OLLAMA    = "mistral-nemo"
TTS_MAX_CHARS    = 4500
TRADUCTION_CHUNK = 4000
MAX_AUDIO_PARALLEL  = 4      # requêtes edge-tts simultanées (évite le rate-limit)
PODCAST_MAX_INPUT   = 7000   # chars max envoyés à Ollama pour générer le script podcast

# Podcast : deux caractères via la même voix Kokoro mais vitesses distinctes
# marie  = plus rapide → son plus aigu, plus énergique
# thomas = plus lent   → son plus grave, plus posé
VOIX_PODCAST = {
    "marie":  ("ff_siwis", 1.20),   # enthousiaste, réactive
    "thomas": ("ff_siwis", 0.85),   # posé, expert qui prend son temps
}

VOIX_DISPONIBLES = {
    "vivienne" : "fr-FR-VivienneMultilingualNeural",
    "remy"     : "fr-FR-RemyMultilingualNeural",
    "denise"   : "fr-FR-DeniseNeural",
    "henri"    : "fr-FR-HenriNeural",
    "sylvie"   : "fr-CA-SylvieNeural",
}
VOIX_DEFAUT = "vivienne"

# ElevenLabs : voix pré-faites disponibles sur le plan Starter
# Pour changer : copie l'ID depuis elevenlabs.io → My Voices ou Voice Library
ELEVENLABS_MODEL = "eleven_multilingual_v2"
ELEVENLABS_VOIX = {
    "marie":  "EXAVITQu4vr4xnSDxMaL",  # Sarah  — féminine, naturelle
    "thomas": "onwK4e9ZLuTAKqWW03F9",   # Daniel — masculine, posé
}

# XTTS v2 : clonage vocal à partir d'un court clip audio (5-30 sec)
XTTS_MODEL_ID  = "tts_models/multilingual/multi-dataset/xtts_v2"
XTTS_MAX_CHARS = 300   # blocs plus petits = moins de blocages

# Piper TTS : rapide, plusieurs voix françaises pré-entraînées
# Modèles téléchargés automatiquement depuis HuggingFace au 1er usage
PIPER_VOIX = {
    "siwis":  ("fr/fr_FR/siwis/medium",  "fr_FR-siwis-medium"),   # féminine, claire
    "gilles": ("fr/fr_FR/gilles/low",    "fr_FR-gilles-low"),     # masculine, posée
    "tom":    ("fr/fr_FR/tom/medium",    "fr_FR-tom-medium"),     # masculine, jeune
    "upmc-jessica": ("fr/fr_FR/upmc/medium", "fr_FR-upmc-medium"), # féminine naturelle (speaker 0)
    "upmc-pierre":  ("fr/fr_FR/upmc/medium", "fr_FR-upmc-medium"), # masculine profonde (speaker 1)
}
PIPER_SPEAKERS = {"upmc-jessica": 0, "upmc-pierre": 1}  # multi-speaker
PIPER_DOSSIER  = Path.home() / ".cache" / "piper_voices"
PIPER_VITESSE_DEFAUT = 0.95   # 1.0 = normal, 0.95 = légèrement plus lent, 0.8 = +15%, 0.7 = +33%

# Modifié par l'option CLI --piper-vitesse au démarrage
_piper_vitesse = PIPER_VITESSE_DEFAUT

# OpenAI TTS : payant à l'usage (~7 $ par livre standard, ~3,75 $ avec tts-1)
# Voix : alloy, echo, fable, onyx, nova, shimmer
OPENAI_MODEL_DEFAUT = "tts-1"        # ou "tts-1-hd" pour 2x la qualité et le prix
OPENAI_MAX_CHARS    = 4000           # limite stricte de l'API : 4096
OPENAI_VOIX_PODCAST = {"marie": "nova", "thomas": "onyx"}


# -----------------------------------------------------------------------
# Utilitaires
# -----------------------------------------------------------------------

def nettoyer_texte(texte: str) -> str:
    texte = re.sub(r"\n{3,}", "\n\n", texte)
    return "\n".join(l.strip() for l in texte.splitlines())


def sanitiser(nom: str) -> str:
    nom = re.sub(r'[<>:"/\\|?*]', "", nom)
    nom = re.sub(r"\s+", "_", nom.strip())
    return nom[:80]


def parser_spec_chapitres(spec: str, total: int) -> list[int]:
    """Parse '3' (3 premiers), '1-5', '3,7,9', '1-3,7,9-12' -> liste d'indices triés (1-indexés)."""
    spec = spec.strip()
    if not spec:
        return list(range(1, total + 1))

    # Cas simple : un seul nombre = les N premiers chapitres
    if spec.isdigit():
        return list(range(1, min(int(spec), total) + 1))

    result = set()
    for partie in spec.split(","):
        partie = partie.strip()
        if not partie:
            continue
        if "-" in partie:
            debut, fin = partie.split("-", 1)
            for n in range(int(debut.strip()), int(fin.strip()) + 1):
                if 1 <= n <= total:
                    result.add(n)
        else:
            n = int(partie)
            if 1 <= n <= total:
                result.add(n)
    return sorted(result)


def chapitre_deja_genere(i: int, titre: str, dossier_sortie: Path) -> bool:
    """Vrai si l'audio du chapitre i existe déjà (.mp3 ou .wav)."""
    nom_base = f"{i:02d}_{sanitiser(titre)}"
    return (dossier_sortie / f"{nom_base}.mp3").exists() or \
           (dossier_sortie / f"{nom_base}.wav").exists()


def titre_du_fichier(chemin: Path) -> str:
    nom = chemin.stem
    for prefixe in ["dokumen.pub_", "www.", "download_", "pdfdrive_"]:
        if nom.lower().startswith(prefixe):
            nom = nom[len(prefixe):]
    nom = re.sub(r"[-_]\d+(?:nd|rd|th|st)?[-_]?edition.*$", "", nom, flags=re.IGNORECASE)
    nom = re.sub(r"[-_]\d{10,}.*$", "", nom)
    nom = re.sub(r"[-_]\d+nbsped.*$", "", nom)
    nom = nom.replace("-", " ").replace("_", " ")
    return re.sub(r"\s+", " ", nom).strip().title()


def decouper_en_blocs(texte: str, max_chars: int) -> list[str]:
    blocs, bloc = [], ""
    for para in texte.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if len(para) > max_chars:
            for phrase in re.split(r"(?<=[.!?»])\s+", para):
                if len(bloc) + len(phrase) + 1 <= max_chars:
                    bloc += (" " if bloc else "") + phrase
                else:
                    if bloc:
                        blocs.append(bloc)
                    bloc = phrase[:max_chars]
        else:
            if len(bloc) + len(para) + 2 <= max_chars:
                bloc += ("\n\n" if bloc else "") + para
            else:
                if bloc:
                    blocs.append(bloc)
                bloc = para
    if bloc:
        blocs.append(bloc)
    return [b for b in blocs if b.strip()]


# -----------------------------------------------------------------------
# Extraction par chapitres
# -----------------------------------------------------------------------

def _toc_vers_map(elements) -> dict[str, str]:
    """Parcourt récursivement la TOC epub → {nom_fichier: titre_chapitre}."""
    map_titres = {}
    for el in elements:
        if isinstance(el, epub.Link):
            href = el.href.split("#")[0]   # retire le fragment (#section-id)
            if href and el.title and href not in map_titres:
                map_titres[href] = el.title
        elif isinstance(el, tuple) and len(el) == 2:
            map_titres.update(_toc_vers_map(el[1]))   # sous-niveaux
    return map_titres


def extraire_chapitres_epub(chemin: Path) -> list[tuple[str, str]]:
    livre      = epub.read_epub(str(chemin))
    titres_toc = _toc_vers_map(livre.toc)   # titres officiels de la TOC
    chapitres  = []

    for item in livre.get_items():
        if item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        soup  = BeautifulSoup(item.get_content(), "html.parser")
        texte = soup.get_text(separator="\n").strip()
        if len(texte) < 300:
            continue

        # 1. Titre depuis la TOC interne (le plus fiable)
        titre = titres_toc.get(item.get_name())

        # 2. Fallback : premier h1/h2/h3 du HTML
        if not titre:
            for tag in ("h1", "h2", "h3"):
                h = soup.find(tag)
                if h and h.get_text().strip():
                    titre = h.get_text().strip()
                    break

        # 3. Fallback : numéro d'ordre
        if not titre:
            titre = f"Partie {len(chapitres) + 1}"

        chapitres.append((titre, nettoyer_texte(texte)))
    return chapitres


def extraire_chapitres_pdf(chemin: Path) -> list[tuple[str, str]]:
    doc = fitz.open(str(chemin))
    toc = doc.get_toc()
    if toc:
        entrees   = [(lvl, t, p) for lvl, t, p in toc if lvl <= 2]
        chapitres = []
        for i, (_, titre, pg_debut) in enumerate(entrees):
            pg_fin = entrees[i + 1][2] if i + 1 < len(entrees) else len(doc) + 1
            texte  = "".join(doc[p].get_text() for p in range(pg_debut - 1, min(pg_fin - 1, len(doc))))
            texte_propre = nettoyer_texte(texte)
            # Ignore les entrées de TOC qui ne contiennent qu'un titre de section
            if len(texte_propre.strip()) >= 300:
                chapitres.append((titre, texte_propre))
        doc.close()
        if chapitres:
            return chapitres
    texte_total = "\n\n".join(p.get_text() for p in doc)
    doc.close()
    return _chapitres_par_regex(nettoyer_texte(texte_total))


def _chapitres_par_regex(texte: str) -> list[tuple[str, str]]:
    pattern = re.compile(
        r"^(?:Chapter|Chapitre|CHAPTER|CHAPITRE)\s+\d+[^\n]{0,80}$"
        r"|^\d{1,2}[.\s]+[A-ZÁÀÂÉÈÊÏÎÔÙÛÜ][^\n]{3,60}$",
        re.MULTILINE,
    )
    matches = list(pattern.finditer(texte))
    if not matches:
        return [("Livre complet", texte)]
    chapitres = []
    intro = texte[: matches[0].start()].strip()
    if len(intro) > 300:
        chapitres.append(("Introduction", intro))
    for i, m in enumerate(matches):
        fin     = matches[i + 1].start() if i + 1 < len(matches) else len(texte)
        contenu = texte[m.end():fin].strip()
        if len(contenu) >= 300:
            chapitres.append((m.group().strip(), contenu))
    return chapitres


# -----------------------------------------------------------------------
# Détection de langue
# -----------------------------------------------------------------------

def detecter_langue(texte: str) -> str:
    try:
        # On se base uniquement sur les lignes de prose longues (≥ 40 chars)
        # pour ne pas être trompé par des titres de section en anglais
        # dans un texte majoritairement français (ou vice versa).
        lignes = [l.strip() for l in texte.splitlines() if len(l.strip()) >= 40]
        echantillon = " ".join(lignes)[:4000]
        return detect(echantillon) if echantillon else "inconnu"
    except LangDetectException:
        return "inconnu"


# -----------------------------------------------------------------------
# Traduction avec Ollama (synchrone — tourne dans un thread)
# -----------------------------------------------------------------------

def _traduire_sync(texte: str, modele: str) -> str:
    """Traduit un texte complet en français, bloc par bloc."""
    blocs    = decouper_en_blocs(texte, TRADUCTION_CHUNK)
    traduits = []
    for bloc in blocs:
        rep = ollama.chat(
            model=modele,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu es un traducteur littéraire expert. "
                        "Traduis fidèlement en français en conservant style et nuances. "
                        "Renvoie UNIQUEMENT la traduction."
                    ),
                },
                {"role": "user", "content": f"Traduis en français :\n\n{bloc}"},
            ],
        )
        traduits.append(rep["message"]["content"])
    return "\n\n".join(traduits)


# -----------------------------------------------------------------------
# Mode podcast : génération du script dialogue via Ollama
# -----------------------------------------------------------------------

def _generer_script_podcast(texte: str, modele: str, est_anglais: bool) -> list[tuple[str, str]]:
    """
    Demande à Ollama de transformer le texte du chapitre en dialogue de podcast.
    Retourne une liste de (locuteur, réplique) : [("marie", "..."), ("thomas", "..."), ...]
    """
    extrait = texte[:PODCAST_MAX_INPUT]

    if est_anglais:
        consigne = "Transforme ce contenu anglais en script de podcast en FRANÇAIS."
    else:
        consigne = "Transforme ce contenu en script de podcast en français."

    prompt = f"""{consigne}

PERSONNAGES — donne-leur une vraie personnalité, pas des robots neutres :

MARIE : drôle et piquante. Elle pose des questions mais avec un regard décalé — elle compare tout à des trucs de la vie quotidienne parfois absurdes ("attends, c'est comme quand tu ranges ton frigo au dernier moment ?"). Elle exagère ses réactions, lâche des "NON MAIS SÉRIEUSEMENT" ou "ok là je comprends rien et c'est ta faute". Jamais méchante, toujours complice.

THOMAS : passionné mais conscient qu'il peut devenir indigeste. Il se moque de lui-même quand il s'emballe ("je sens que je perds tout le monde là"). Il a des analogies géniales mais parfois un peu trop poussées. Il dit des trucs comme "bon ok je vais faire simple" puis complique quand même. Il adore les mauvais jeux de mots et ne s'en excuse pas.

Ensemble ils se chamaillent, se coupent, se reprennent, rient — comme deux amis intelligents dans un bar, pas deux présentateurs de télé.

FORMAT STRICT — une réplique par ligne, rien d'autre :
MARIE: [texte]
THOMAS: [texte]

RÈGLES :
- Entre 15 et 28 échanges
- Humour, dérision, autodérision — mais les idées clés du contenu DOIVENT passer
- Vraies interruptions, vraies réactions ("ha oui attends—", "non stop, répète ça")
- Pas de "En conclusion", pas de résumé propret à la fin
- UNIQUEMENT des lignes MARIE: ou THOMAS:

CONTENU :
{extrait}"""

    rep = ollama.chat(
        model=modele,
        messages=[{"role": "user", "content": prompt}],
    )
    return _parser_script(rep["message"]["content"])


def _parser_script(script: str) -> list[tuple[str, str]]:
    """Extrait les répliques MARIE:/THOMAS: d'un script généré par Ollama."""
    repliques = []
    for ligne in script.splitlines():
        ligne = ligne.strip()
        for nom in ("MARIE", "THOMAS"):
            if ligne.upper().startswith(f"{nom}:"):
                texte = ligne[len(nom) + 1:].strip()
                if texte:
                    repliques.append((nom.lower(), texte))
                break
    return repliques


# -----------------------------------------------------------------------
# Génération audio avec edge-tts (asynchrone)
# -----------------------------------------------------------------------

async def _generer_bloc(texte: str, voix: str, chemin: Path, rate: str = "-5%"):
    await edge_tts.Communicate(texte, voix, rate=rate).save(str(chemin))


async def generer_audio_chapitre(texte: str, voix: str, chemin_sortie: Path):
    """Génère l'audio d'un chapitre : tous les blocs sont lancés en parallèle."""
    blocs = decouper_en_blocs(texte, TTS_MAX_CHARS)
    if len(blocs) == 1:
        await _generer_bloc(blocs[0], voix, chemin_sortie)
        return
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        chemins  = [tmp_path / f"bloc_{k:04d}.mp3" for k in range(len(blocs))]
        # Tous les blocs du chapitre en parallèle
        await asyncio.gather(*[_generer_bloc(b, voix, c) for b, c in zip(blocs, chemins)])
        _concatener_mp3(sorted(tmp_path.glob("*.mp3")), chemin_sortie)


def _concatener_mp3(fichiers, sortie: Path):
    try:
        from pydub import AudioSegment
        combined = AudioSegment.empty()
        for f in fichiers:
            combined += AudioSegment.from_mp3(str(f))
        combined.export(str(sortie), format="mp3", bitrate="192k")
    except (ImportError, Exception):
        with open(sortie, "wb") as f_out:
            for f in fichiers:
                f_out.write(Path(f).read_bytes())


# -----------------------------------------------------------------------
# Mode podcast : Kokoro TTS (plus expressif qu'edge-tts)
# -----------------------------------------------------------------------

_kokoro_pipeline = None   # initialisé une seule fois, partagé entre chapitres

def _get_kokoro():
    global _kokoro_pipeline
    if _kokoro_pipeline is None:
        from kokoro import KPipeline
        print("   Chargement du modele Kokoro (premiere fois seulement)...")
        _kokoro_pipeline = KPipeline(lang_code="f")
    return _kokoro_pipeline


def _generer_podcast_sync(repliques: list[tuple[str, str]]):
    """
    Genere tout l'audio du dialogue en memoire (numpy).
    Tourne dans un thread executor car Kokoro est synchrone.
    """
    import numpy as np
    pipeline = _get_kokoro()
    silence  = np.zeros(int(24000 * 0.35))   # 350 ms entre chaque replique
    segments = []

    for locuteur, texte in repliques:
        voix, vitesse = VOIX_PODCAST[locuteur]
        gen   = pipeline(texte, voice=voix, speed=vitesse)
        audio = np.concatenate([s[2] for s in gen])
        segments.append(audio)
        segments.append(silence)

    return np.concatenate(segments) if segments else np.array([])


async def generer_audio_podcast_chapitre(
    repliques: list[tuple[str, str]],
    chemin_sortie: Path,
):
    """Lance la generation Kokoro dans un thread, sauvegarde en MP3 (ou WAV)."""
    import numpy as np
    import soundfile as sf

    loop  = asyncio.get_event_loop()
    audio = await loop.run_in_executor(None, _generer_podcast_sync, repliques)

    if len(audio) == 0:
        return

    chemin_wav = chemin_sortie.with_suffix(".wav")
    sf.write(str(chemin_wav), audio, 24000)

    # Conversion en MP3 via pydub si disponible
    try:
        from pydub import AudioSegment
        AudioSegment.from_wav(str(chemin_wav)).export(
            str(chemin_sortie), format="mp3", bitrate="192k"
        )
        chemin_wav.unlink()
    except Exception:
        # Sans ffmpeg : on renomme en .wav
        chemin_wav.rename(chemin_sortie.with_suffix(".wav"))


# -----------------------------------------------------------------------
# OpenAI TTS : rapide, naturel, payant à l'usage (~7 $/livre standard)
# -----------------------------------------------------------------------

_openai_client = None


def _get_openai():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "Clé API manquante. Définis la variable OPENAI_API_KEY :\n"
                "  export OPENAI_API_KEY=sk-..."
            )
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def _openai_synthese(texte: str, voix: str, modele: str) -> bytes:
    """Génère un MP3 via OpenAI TTS, retourne les bytes."""
    response = _get_openai().audio.speech.create(
        model=modele,
        voice=voix,
        input=texte,
        response_format="mp3",
    )
    return response.content


def _concatener_mp3_bytes(segments: list[bytes], chemin: Path, silence_ms: int = 300):
    """Concatène plusieurs MP3 (bytes) en un seul fichier MP3."""
    import io
    try:
        from pydub import AudioSegment
        silence  = AudioSegment.silent(duration=silence_ms)
        combined = AudioSegment.empty()
        for s in segments:
            combined += AudioSegment.from_mp3(io.BytesIO(s)) + silence
        combined.export(str(chemin), format="mp3", bitrate="128k")
    except Exception:
        # Fallback : concatène brutalement les bytes (les lecteurs gèrent généralement)
        with open(chemin, "wb") as f:
            for s in segments:
                f.write(s)


def _openai_livre_sync(texte: str, voix: str, modele: str, label: str = "") -> list[bytes]:
    """Mode livre : découpe + génère un MP3 par bloc."""
    blocs    = decouper_en_blocs(texte, OPENAI_MAX_CHARS)
    segments = []
    for k, bloc in enumerate(blocs, 1):
        if label:
            print(f"   OpenAI {label} — bloc {k}/{len(blocs)}...", flush=True)
        segments.append(_openai_synthese(bloc, voix, modele))
    return segments


async def generer_audio_openai(
    texte: str, voix: str, chemin_sortie: Path,
    modele: str = OPENAI_MODEL_DEFAUT, label: str = "",
):
    """Mode livre : génère l'audio d'un chapitre via OpenAI TTS."""
    loop     = asyncio.get_event_loop()
    segments = await loop.run_in_executor(
        None, _openai_livre_sync, texte, voix, modele, label
    )
    _concatener_mp3_bytes(segments, chemin_sortie, silence_ms=300)


def _openai_podcast_sync(
    repliques: list[tuple[str, str]],
    voix_marie: str,
    voix_thomas: str,
    modele: str,
    label: str = "",
) -> list[bytes]:
    """Mode podcast : génère un MP3 par réplique avec la voix du locuteur."""
    voix     = {"marie": voix_marie, "thomas": voix_thomas}
    segments = []
    total    = len(repliques)
    for k, (locuteur, texte) in enumerate(repliques, 1):
        if label:
            print(f"   OpenAI {label} — réplique {k}/{total} ({locuteur})...", flush=True)
        segments.append(_openai_synthese(texte, voix[locuteur], modele))
    return segments


async def generer_audio_podcast_openai(
    repliques: list[tuple[str, str]],
    voix_marie: str,
    voix_thomas: str,
    chemin_sortie: Path,
    modele: str = OPENAI_MODEL_DEFAUT,
    label: str = "",
):
    """Mode podcast : génère l'audio du dialogue via OpenAI TTS."""
    loop     = asyncio.get_event_loop()
    segments = await loop.run_in_executor(
        None, _openai_podcast_sync, repliques, voix_marie, voix_thomas, modele, label
    )
    _concatener_mp3_bytes(segments, chemin_sortie, silence_ms=400)


# -----------------------------------------------------------------------
# Piper TTS : rapide, plusieurs voix françaises (gratuit, local)
# -----------------------------------------------------------------------

_piper_voices = {}   # cache des voix chargées


def _telecharger_piper_voix(nom: str) -> Path:
    """Télécharge le modèle .onnx et son .json si absent. Retourne le chemin du .onnx."""
    import urllib.request

    sous_chemin, prefixe = PIPER_VOIX[nom]
    dossier = PIPER_DOSSIER / sous_chemin
    dossier.mkdir(parents=True, exist_ok=True)

    onnx = dossier / f"{prefixe}.onnx"
    json = dossier / f"{prefixe}.onnx.json"
    base = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/{sous_chemin}/{prefixe}"

    for fichier, url in [(onnx, f"{base}.onnx"), (json, f"{base}.onnx.json")]:
        if not fichier.exists():
            print(f"   Téléchargement voix Piper '{nom}'...")
            urllib.request.urlretrieve(url, fichier)
    return onnx


def _get_piper_voice(nom: str):
    if nom not in _piper_voices:
        from piper import PiperVoice
        chemin = _telecharger_piper_voix(nom)
        _piper_voices[nom] = PiperVoice.load(str(chemin))
    return _piper_voices[nom]


def _piper_synthese(texte: str, nom_voix: str) -> bytes:
    """Synthétise un texte avec Piper, retourne le WAV en bytes."""
    import io
    import wave
    from piper.config import SynthesisConfig

    voice    = _get_piper_voice(nom_voix)
    speaker  = PIPER_SPEAKERS.get(nom_voix)
    # length_scale : plus c'est haut, plus c'est lent (1.0 / vitesse)
    length_scale = 1.0 / _piper_vitesse if _piper_vitesse != 1.0 else None
    config   = SynthesisConfig(speaker_id=speaker, length_scale=length_scale)
    buf      = io.BytesIO()

    with wave.open(buf, "wb") as wav_file:
        voice.synthesize_wav(texte, wav_file, syn_config=config)
    return buf.getvalue()


def _sauvegarder_piper(wav_segments: list[bytes], chemin: Path, silence_ms: int = 300):
    """Concatène plusieurs WAV Piper et sauvegarde en MP3 (WAV en fallback)."""
    import io
    import wave
    import numpy as np
    import soundfile as sf

    # Lit tous les segments et concatène
    samples_list = []
    sample_rate  = None
    for wav_bytes in wav_segments:
        with wave.open(io.BytesIO(wav_bytes), "rb") as w:
            sample_rate = w.getframerate()
            frames      = w.readframes(w.getnframes())
            samples_list.append(np.frombuffer(frames, dtype=np.int16))

    silence = np.zeros(int(sample_rate * silence_ms / 1000), dtype=np.int16)
    samples = []
    for s in samples_list:
        samples.append(s)
        samples.append(silence)
    audio_final = np.concatenate(samples) if samples else np.array([], dtype=np.int16)

    buf = io.BytesIO()
    sf.write(buf, audio_final, sample_rate, format="WAV")
    buf.seek(0)
    try:
        from pydub import AudioSegment
        AudioSegment.from_wav(buf).export(str(chemin), format="mp3", bitrate="192k")
    except Exception:
        chemin.with_suffix(".wav").write_bytes(buf.getvalue())


def _piper_samples_livre_sync(texte: str, nom_voix: str, label: str = "") -> list[bytes]:
    """Mode livre : découpe le texte et génère un WAV par bloc."""
    blocs    = decouper_en_blocs(texte, 2000)   # Piper gère bien des blocs longs
    segments = []
    for k, bloc in enumerate(blocs, 1):
        if label:
            print(f"   Piper {label} — bloc {k}/{len(blocs)}...", flush=True)
        segments.append(_piper_synthese(bloc, nom_voix))
    return segments


async def generer_audio_piper(texte: str, nom_voix: str, chemin_sortie: Path, label: str = ""):
    """Mode livre : génère l'audio d'un chapitre via Piper."""
    loop     = asyncio.get_event_loop()
    segments = await loop.run_in_executor(None, _piper_samples_livre_sync, texte, nom_voix, label)
    _sauvegarder_piper(segments, chemin_sortie, silence_ms=300)


def _generer_podcast_piper_sync(
    repliques: list[tuple[str, str]],
    voix_marie: str,
    voix_thomas: str,
    label: str = "",
) -> list[bytes]:
    """Mode podcast : génère un WAV par réplique avec la voix du locuteur."""
    voix     = {"marie": voix_marie, "thomas": voix_thomas}
    segments = []
    total    = len(repliques)
    for k, (locuteur, texte) in enumerate(repliques, 1):
        if label:
            print(f"   Piper {label} — réplique {k}/{total} ({locuteur})...", flush=True)
        segments.append(_piper_synthese(texte, voix[locuteur]))
    return segments


async def generer_audio_podcast_piper(
    repliques: list[tuple[str, str]],
    voix_marie: str,
    voix_thomas: str,
    chemin_sortie: Path,
    label: str = "",
):
    """Mode podcast : génère l'audio du dialogue via Piper."""
    loop     = asyncio.get_event_loop()
    segments = await loop.run_in_executor(
        None, _generer_podcast_piper_sync, repliques, voix_marie, voix_thomas, label
    )
    _sauvegarder_piper(segments, chemin_sortie, silence_ms=400)


# -----------------------------------------------------------------------
# XTTS v2 : clonage vocal local (gratuit, nécessite un clip de référence)
# -----------------------------------------------------------------------

_xtts_model    = None
_xtts_lock     = None   # protège le chargement du modèle
_xtts_inf_lock = None   # XTTS n'est pas thread-safe : une seule inférence à la fois


def _get_xtts():
    global _xtts_model, _xtts_lock, _xtts_inf_lock
    import threading
    if _xtts_lock is None:
        _xtts_lock     = threading.Lock()
        _xtts_inf_lock = threading.Lock()
    with _xtts_lock:
        if _xtts_model is None:
            import torch
            from TTS.api import TTS
            os.environ["COQUI_TOS_AGREED"] = "1"
            gpu = torch.cuda.is_available()
            print("   Chargement XTTS v2 (1ère fois : ~1.8 Go à télécharger)...")
            _xtts_model = TTS(XTTS_MODEL_ID, gpu=gpu)
    return _xtts_model


def _nettoyer_pour_xtts(texte: str) -> str:
    """Nettoie le texte pour éviter les blocages de XTTS."""
    import unicodedata
    # Supprime les caractères de contrôle (sauf newline)
    texte = "".join(c for c in unicodedata.normalize("NFC", texte)
                    if c == "\n" or not unicodedata.category(c).startswith("C"))
    # Remplace les éléments qui font souvent planter XTTS
    texte = re.sub(r"https?://\S+", "lien", texte)
    texte = re.sub(r"\S+@\S+\.\S+", "email", texte)
    texte = re.sub(r"(.)\1{4,}", r"\1\1\1", texte)        # AAAAAA → AAA
    texte = re.sub(r"[^\w\s.,!?;:'\"()«»–\-—\n]", " ", texte)
    texte = re.sub(r"\s+", " ", texte).strip()
    return texte


def _xtts_tts(texte: str, clip: str) -> list:
    """Appel XTTS thread-safe (une seule inférence à la fois)."""
    with _xtts_inf_lock:
        return _get_xtts().tts(text=texte, speaker_wav=clip, language="fr")


def _xtts_samples(texte: str, clip: str, label: str = "") -> list:
    """Génère les échantillons audio pour un texte via XTTS."""
    blocs   = decouper_en_blocs(texte, XTTS_MAX_CHARS)
    silence = [0.0] * int(24000 * 0.25)
    audio   = []
    ignores = 0
    for k, bloc in enumerate(blocs, 1):
        bloc_propre = _nettoyer_pour_xtts(bloc)
        if len(bloc_propre) < 3:
            ignores += 1
            continue
        if label:
            print(f"   XTTS {label} — bloc {k}/{len(blocs)}...", flush=True)
        try:
            audio.extend(_xtts_tts(bloc_propre, clip))
            audio.extend(silence)
        except Exception as e:
            print(f"   ! bloc {k} ignoré ({type(e).__name__}: {str(e)[:60]})", flush=True)
            ignores += 1
    if ignores:
        print(f"   ({ignores} bloc(s) ignoré(s) sur {len(blocs)})", flush=True)
    return audio


def _sauvegarder_xtts(audio: list, chemin: Path):
    """Sauvegarde les échantillons XTTS en MP3 (WAV en fallback)."""
    import io
    import numpy as np
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, np.array(audio, dtype=np.float32), 24000, format="WAV")
    buf.seek(0)
    try:
        from pydub import AudioSegment
        AudioSegment.from_wav(buf).export(str(chemin), format="mp3", bitrate="192k")
    except Exception:
        chemin.with_suffix(".wav").write_bytes(buf.getvalue())


async def generer_audio_xtts(texte: str, clip: str, chemin_sortie: Path, label: str = ""):
    """Génère l'audio d'un chapitre (mode livre) via XTTS."""
    loop  = asyncio.get_event_loop()
    audio = await loop.run_in_executor(None, _xtts_samples, texte, clip, label)
    _sauvegarder_xtts(audio, chemin_sortie)


def _generer_podcast_xtts_sync(
    repliques: list[tuple[str, str]],
    clip_marie: str,
    clip_thomas: str,
    label: str = "",
) -> list:
    """Génère l'audio du dialogue podcast via XTTS (une réplique à la fois)."""
    clips   = {"marie": clip_marie, "thomas": clip_thomas}
    silence = [0.0] * int(24000 * 0.4)
    audio   = []
    total   = len(repliques)
    ignores = 0
    for k, (locuteur, texte) in enumerate(repliques, 1):
        texte_propre = _nettoyer_pour_xtts(texte)
        if len(texte_propre) < 3:
            ignores += 1
            continue
        if label:
            print(f"   XTTS {label} — réplique {k}/{total} ({locuteur})...", flush=True)
        try:
            audio.extend(_xtts_tts(texte_propre, clips[locuteur]))
            audio.extend(silence)
        except Exception as e:
            print(f"   ! réplique {k} ignorée ({type(e).__name__}: {str(e)[:60]})", flush=True)
            ignores += 1
    if ignores:
        print(f"   ({ignores} réplique(s) ignorée(s) sur {total})", flush=True)
    return audio


async def generer_audio_podcast_xtts(
    repliques: list[tuple[str, str]],
    clip_marie: str,
    clip_thomas: str,
    chemin_sortie: Path,
    label: str = "",
):
    """Lance XTTS dans un thread et sauvegarde le résultat en MP3."""
    loop  = asyncio.get_event_loop()
    audio = await loop.run_in_executor(
        None, _generer_podcast_xtts_sync, repliques, clip_marie, clip_thomas, label
    )
    _sauvegarder_xtts(audio, chemin_sortie)


# -----------------------------------------------------------------------
# Mode podcast : ElevenLabs TTS (option payante, meilleure qualité)
# -----------------------------------------------------------------------

def _generer_podcast_elevenlabs_sync(repliques: list[tuple[str, str]]) -> bytes:
    """Génère l'audio du dialogue via ElevenLabs. Retourne les bytes MP3 concaténés."""
    from elevenlabs import ElevenLabs
    from pydub import AudioSegment
    import io

    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise ValueError(
            "Clé API manquante. Définis la variable d'environnement ELEVENLABS_API_KEY.\n"
            "  export ELEVENLABS_API_KEY=sk-..."
        )

    client   = ElevenLabs(api_key=api_key)
    silence  = AudioSegment.silent(duration=400)   # 400 ms entre chaque réplique
    combined = AudioSegment.empty()

    for locuteur, texte in repliques:
        voice_id    = ELEVENLABS_VOIX[locuteur]
        audio_bytes = b"".join(
            client.text_to_speech.convert(
                voice_id=voice_id,
                text=texte,
                model_id=ELEVENLABS_MODEL,
                output_format="mp3_44100_128",
            )
        )
        segment   = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
        combined += segment + silence

    buf = io.BytesIO()
    combined.export(buf, format="mp3", bitrate="128k")
    return buf.getvalue()


async def generer_audio_podcast_elevenlabs(
    repliques: list[tuple[str, str]],
    chemin_sortie: Path,
):
    """Lance ElevenLabs dans un thread et sauvegarde le résultat en MP3."""
    loop        = asyncio.get_event_loop()
    audio_bytes = await loop.run_in_executor(
        None, _generer_podcast_elevenlabs_sync, repliques
    )
    chemin_sortie.write_bytes(audio_bytes)


# -----------------------------------------------------------------------
# Orchestration principale (pipeline traduction + audio en parallèle)
# -----------------------------------------------------------------------

async def traiter_livre(
    chapitres: list[tuple[str, str, bool]],   # (titre, texte, à_traduire)
    voix_id: str,
    dossier_sortie: Path,
    modele: str,
    mode: str,                                 # "livre" ou "podcast"
    tts: str = "edge",                         # "edge" | "kokoro" | "elevenlabs" | "xtts" | "piper" | "openai"
    clip_voix: str | None = None,             # clip de référence mode livre (XTTS)
    clip_marie: str | None = None,            # clip Marie mode podcast (XTTS)
    clip_thomas: str | None = None,           # clip Thomas mode podcast (XTTS)
    piper_voix: str | None = None,            # voix Piper mode livre
    piper_marie: str | None = None,           # voix Piper Marie mode podcast
    piper_thomas: str | None = None,          # voix Piper Thomas mode podcast
    openai_voix: str | None = None,           # voix OpenAI mode livre
    openai_marie: str | None = None,          # voix OpenAI Marie mode podcast
    openai_thomas: str | None = None,         # voix OpenAI Thomas mode podcast
    openai_modele: str = OPENAI_MODEL_DEFAUT, # tts-1 ou tts-1-hd
):
    # Pré-charge XTTS avant le pipeline pour que les fichiers soient générés dès le 1er chapitre
    if tts == "xtts":
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _get_xtts)

    total         = len(chapitres)
    semaphore     = asyncio.Semaphore(MAX_AUDIO_PARALLEL)
    loop          = asyncio.get_event_loop()
    termine       = [0]
    nb_a_traiter  = sum(1 for _, _, t in chapitres if t)  # chapitres EN → à traiter

    async def audio_livre(i: int, titre: str, texte: str):
        nom    = f"{i:02d}_{sanitiser(titre)}.mp3"
        chemin = dossier_sortie / nom
        try:
            async with semaphore:
                if tts == "xtts":
                    label = f"[{i:02d}/{total}] {titre}"
                    await generer_audio_xtts(texte, clip_voix, chemin, label)
                elif tts == "piper":
                    label = f"[{i:02d}/{total}] {titre}"
                    await generer_audio_piper(texte, piper_voix, chemin, label)
                elif tts == "openai":
                    label = f"[{i:02d}/{total}] {titre}"
                    await generer_audio_openai(texte, openai_voix, chemin, openai_modele, label)
                else:
                    await generer_audio_chapitre(texte, voix_id, chemin)
            termine[0] += 1
            sortie = chemin if chemin.exists() else chemin.with_suffix(".wav")
            taille = sortie.stat().st_size / 1024
            print(f"   Audio OK [{termine[0]:02d}/{total:02d}] {titre}  ({taille:.0f} Ko)")
        except Exception as e:
            print(f"   ERREUR [{i:02d}/{total}] {titre} : {e}", flush=True)

    async def audio_podcast(i: int, titre: str, repliques: list):
        nom    = f"{i:02d}_{sanitiser(titre)}.mp3"
        chemin = dossier_sortie / nom
        try:
            async with semaphore:
                if tts == "elevenlabs":
                    await generer_audio_podcast_elevenlabs(repliques, chemin)
                elif tts == "xtts":
                    label = f"[{i:02d}/{total}] {titre}"
                    await generer_audio_podcast_xtts(repliques, clip_marie, clip_thomas, chemin, label)
                elif tts == "piper":
                    label = f"[{i:02d}/{total}] {titre}"
                    await generer_audio_podcast_piper(repliques, piper_marie, piper_thomas, chemin, label)
                elif tts == "openai":
                    label = f"[{i:02d}/{total}] {titre}"
                    await generer_audio_podcast_openai(repliques, openai_marie, openai_thomas, chemin, openai_modele, label)
                else:
                    await generer_audio_podcast_chapitre(repliques, chemin)
            termine[0] += 1
            sortie = chemin if chemin.exists() else chemin.with_suffix(".wav")
            taille = sortie.stat().st_size / 1024
            print(f"   Audio OK [{termine[0]:02d}/{total:02d}] {titre}  ({len(repliques)} répliques, {taille:.0f} Ko)")
        except Exception as e:
            print(f"   ERREUR [{i:02d}/{total}] {titre} : {e}", flush=True)

    def traiter_chapitre_sync(texte: str, est_anglais: bool) -> str | list:
        """Dans un thread : traduit si besoin, puis génère le script podcast ou retourne le texte."""
        if est_anglais and mode == "livre":
            return _traduire_sync(texte, modele)
        if mode == "podcast":
            # En podcast, Ollama génère le script (et traduit en même temps si anglais)
            return _generer_script_podcast(texte, modele, est_anglais)
        return texte  # déjà en français, mode livre

    if nb_a_traiter == 0 and mode == "livre":
        # ── Aucune traduction, mode livre : tout en parallèle ──────────
        a_faire = [
            (i, titre, texte)
            for i, (titre, texte, _) in enumerate(chapitres, start=1)
            if not chapitre_deja_genere(i, titre, dossier_sortie)
        ]
        nb_skip = total - len(a_faire)
        if nb_skip:
            print(f"\n {nb_skip} chapitre(s) déjà généré(s) — ignorés")
        if not a_faire:
            print(f"\n Tout est déjà fait dans : {dossier_sortie}/")
            return
        print(f"\n Audio en parallèle ({MAX_AUDIO_PARALLEL} à la fois)...\n")
        await asyncio.gather(*[
            audio_livre(i, titre, texte) for i, titre, texte in a_faire
        ])
    else:
        # ── Pipeline : Ollama dans un thread, edge-tts en arrière-plan ─
        if mode == "podcast":
            print(f"\n Mode podcast — génération des scripts + audio en parallèle...\n")
            if tts == "xtts":
                print(f"   Voix : Marie ({clip_marie}) & Thomas ({clip_thomas})\n")
            elif tts == "elevenlabs":
                print(f"   Voix : Marie (ElevenLabs Sarah) & Thomas (ElevenLabs Daniel)\n")
            else:
                print(f"   Voix : Marie (Kokoro ff_siwis x{VOIX_PODCAST['marie'][1]}) "
                      f"& Thomas (Kokoro ff_siwis x{VOIX_PODCAST['thomas'][1]})\n")
        else:
            print(f"\n Pipeline : {nb_a_traiter}/{total} chapitre(s) à traduire + audio en parallèle...\n")

        audio_tasks = []
        with ThreadPoolExecutor(max_workers=1) as executor:
            for i, (titre, texte, est_anglais) in enumerate(chapitres, start=1):
                if chapitre_deja_genere(i, titre, dossier_sortie):
                    print(f"   Déjà fait [{i:02d}/{total}] {titre} — ignoré")
                    continue
                besoin_ollama = est_anglais or mode == "podcast"
                if besoin_ollama:
                    label = "Podcast" if mode == "podcast" else "Traduction"
                    print(f"   {label} [{i:02d}/{total}] {titre}...", end="\r", flush=True)
                    resultat = await loop.run_in_executor(
                        executor, traiter_chapitre_sync, texte, est_anglais
                    )
                    print(f"   {label} OK [{i:02d}/{total}] {titre}")
                else:
                    print(f"   Déjà FR   [{i:02d}/{total}] {titre}")
                    resultat = texte

                if mode == "podcast":
                    task = asyncio.create_task(audio_podcast(i, titre, resultat))
                else:
                    task = asyncio.create_task(audio_livre(i, titre, resultat))
                audio_tasks.append(task)

        await asyncio.gather(*audio_tasks)


# -----------------------------------------------------------------------
# Programme principal
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convertit un PDF ou EPUB en audio IA par chapitre (gratuit, local)"
    )
    parser.add_argument("fichier")
    parser.add_argument("-o", "--sortie", default=None)
    parser.add_argument("--voix", choices=list(VOIX_DISPONIBLES.keys()), default=VOIX_DEFAUT,
                        help="Voix pour le mode livre (ignoré en mode podcast)")
    parser.add_argument("--modele", default=MODELE_OLLAMA)
    parser.add_argument("--mode", choices=["livre", "podcast"], default="livre",
                        help="livre = lecture classique | podcast = dialogue Marie & Thomas")
    parser.add_argument("--tts", choices=["edge", "kokoro", "elevenlabs", "xtts", "piper", "openai"], default=None,
                        help="Moteur TTS : edge | kokoro | elevenlabs | xtts | piper | openai")
    parser.add_argument("--clip-voix", default=None, metavar="FICHIER.wav",
                        help="Clip audio de référence pour le mode livre avec --tts xtts (5-30 sec)")
    parser.add_argument("--clip-marie", default=None, metavar="FICHIER.wav",
                        help="Clip audio pour la voix de Marie avec --tts xtts (5-30 sec)")
    parser.add_argument("--clip-thomas", default=None, metavar="FICHIER.wav",
                        help="Clip audio pour la voix de Thomas avec --tts xtts (5-30 sec)")
    parser.add_argument("--chapitres", default=None, metavar="SPEC",
                        help="Chapitres à générer. Formats : '3' (3 premiers), '1-5' (plage), "
                             "'1,3,5' (spécifiques), '1-3,7,9-12' (mixte). Vide = tous.")
    parser.add_argument("--piper-voix", choices=list(PIPER_VOIX.keys()), default="siwis",
                        help="Voix Piper pour le mode livre")
    parser.add_argument("--piper-marie", choices=list(PIPER_VOIX.keys()), default="siwis",
                        help="Voix Piper pour Marie en mode podcast")
    parser.add_argument("--piper-thomas", choices=list(PIPER_VOIX.keys()), default="gilles",
                        help="Voix Piper pour Thomas en mode podcast")
    parser.add_argument("--piper-vitesse", type=float, default=PIPER_VITESSE_DEFAUT,
                        help="Vitesse Piper : 1.0 = normal, 0.9 = un peu plus lent, 0.8 = encore plus lent")
    openai_voix_choix = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    parser.add_argument("--openai-voix", choices=openai_voix_choix, default="nova",
                        help="Voix OpenAI pour le mode livre")
    parser.add_argument("--openai-marie", choices=openai_voix_choix, default="nova",
                        help="Voix OpenAI pour Marie en mode podcast")
    parser.add_argument("--openai-thomas", choices=openai_voix_choix, default="onyx",
                        help="Voix OpenAI pour Thomas en mode podcast")
    parser.add_argument("--openai-modele", choices=["tts-1", "tts-1-hd"], default=OPENAI_MODEL_DEFAUT,
                        help="Modèle OpenAI : tts-1 (standard) ou tts-1-hd (qualité supérieure, 2x prix)")
    args = parser.parse_args()

    # Valeur par défaut du moteur TTS selon le mode
    if args.tts is None:
        args.tts = "kokoro" if args.mode == "podcast" else "edge"

    # Applique la vitesse Piper (modifie la variable globale du module)
    global _piper_vitesse
    _piper_vitesse = args.piper_vitesse

    chemin = Path(args.fichier)
    if not chemin.exists():
        print(f"Erreur : '{chemin}' introuvable.")
        sys.exit(1)
    if chemin.suffix.lower() not in (".pdf", ".epub"):
        print("Erreur : seuls les formats .pdf et .epub sont supportés.")
        sys.exit(1)

    # Validation clips XTTS
    if args.tts == "xtts":
        if args.mode == "livre" and not args.clip_voix:
            print("Erreur : --clip-voix requis avec --tts xtts en mode livre.")
            print("  Exemple : --clip-voix ma_voix.wav")
            sys.exit(1)
        if args.mode == "podcast" and (not args.clip_marie or not args.clip_thomas):
            print("Erreur : --clip-marie et --clip-thomas requis avec --tts xtts en mode podcast.")
            print("  Exemple : --clip-marie voix_f.wav --clip-thomas voix_h.wav")
            sys.exit(1)

    voix_id        = VOIX_DISPONIBLES[args.voix]
    titre_livre    = titre_du_fichier(chemin)
    dossier_parent = Path(args.sortie) if args.sortie else chemin.parent
    dossier_sortie = dossier_parent / sanitiser(titre_livre)
    dossier_sortie.mkdir(parents=True, exist_ok=True)

    # Extraction
    print(f"\n Lecture de '{chemin.name}'...")
    chapitres = extraire_chapitres_epub(chemin) if chemin.suffix.lower() == ".epub" \
                else extraire_chapitres_pdf(chemin)

    if not chapitres:
        print("Erreur : aucun contenu extrait.")
        sys.exit(1)

    # Sélection optionnelle de chapitres
    if args.chapitres:
        total_avant = len(chapitres)
        indices     = parser_spec_chapitres(args.chapitres, total_avant)
        if not indices:
            print(f"Erreur : aucun chapitre ne correspond a '{args.chapitres}'.")
            sys.exit(1)
        chapitres = [chapitres[i - 1] for i in indices]
        print(f"   Selection : chapitres {indices} ({len(chapitres)} sur {total_avant})")

    nb_mots = sum(len(t.split()) for _, t in chapitres)
    print(f"   {len(chapitres)} chapitre(s)  (~{nb_mots:,} mots)")
    print(f"   Dossier : {dossier_sortie}/")
    print(f"   Mode    : {args.mode}")
    if args.mode == "livre":
        if args.tts == "xtts":
            print(f"   Voix    : XTTS v2 — clip : {args.clip_voix}")
        elif args.tts == "piper":
            print(f"   Voix    : Piper — {args.piper_voix}")
        elif args.tts == "openai":
            print(f"   Voix    : OpenAI {args.openai_modele} — {args.openai_voix}")
        else:
            print(f"   Voix    : {args.voix} ({voix_id})")
    elif args.tts == "elevenlabs":
        print(f"   Voix    : Marie (ElevenLabs Sarah) & Thomas (ElevenLabs Daniel)")
    elif args.tts == "xtts":
        print(f"   Voix    : Marie ({args.clip_marie}) & Thomas ({args.clip_thomas})")
    elif args.tts == "piper":
        print(f"   Voix    : Marie (Piper {args.piper_marie}) & Thomas (Piper {args.piper_thomas})")
    elif args.tts == "openai":
        print(f"   Voix    : Marie (OpenAI {args.openai_marie}) & Thomas (OpenAI {args.openai_thomas}) [{args.openai_modele}]")
    else:
        print(f"   Voix    : Marie (Kokoro) & Thomas (Kokoro)")

    # Détection de langue chapitre par chapitre
    # (évite les faux positifs dus à des titres/citations en anglais dans un texte français)
    chapitres_info = []
    nb_en = 0
    for titre, texte in chapitres:
        langue = detecter_langue(texte)
        a_traduire = langue == "en"
        if a_traduire:
            nb_en += 1
        chapitres_info.append((titre, texte, a_traduire))

    if nb_en == 0:
        print("   Langue  : français — aucune traduction nécessaire")
    elif nb_en == len(chapitres):
        print(f"   Langue  : anglais — tous les chapitres seront traduits")
    else:
        print(f"   Langue  : mixte — {nb_en}/{len(chapitres)} chapitre(s) en anglais à traduire")

    print(f"\n   Appuyez sur Ctrl+C à tout moment pour arrêter.")
    print(f"   Les chapitres déjà générés seront disponibles dans : {dossier_sortie}/\n")

    try:
        asyncio.run(traiter_livre(
            chapitres_info, voix_id, dossier_sortie, args.modele, args.mode,
            tts=args.tts,
            clip_voix=args.clip_voix,
            clip_marie=args.clip_marie,
            clip_thomas=args.clip_thomas,
            piper_voix=args.piper_voix,
            piper_marie=args.piper_marie,
            piper_thomas=args.piper_thomas,
            openai_voix=args.openai_voix,
            openai_marie=args.openai_marie,
            openai_thomas=args.openai_thomas,
            openai_modele=args.openai_modele,
        ))
        print(f"\n Terminé ! {len(chapitres)} fichiers dans : {dossier_sortie}/")
    except KeyboardInterrupt:
        deja_faits = sorted(dossier_sortie.glob("*.mp3")) + sorted(dossier_sortie.glob("*.wav"))
        print(f"\n\n Arrêt demandé.")
        if deja_faits:
            print(f"   {len(deja_faits)} chapitre(s) disponibles dans : {dossier_sortie}/")
            for f in deja_faits:
                print(f"     - {f.name}")
        else:
            print("   Aucun fichier généré pour l'instant.")
        sys.exit(0)


if __name__ == "__main__":
    main()
