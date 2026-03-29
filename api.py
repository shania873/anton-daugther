import os
import re
import json
import tempfile
import certifi
import requests
import subprocess
import whisper

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import yt_dlp

load_dotenv()
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

app = FastAPI(title="StudyAI API")

# ─────────────────────────────────────────
# AUTHENTIFICATION PAR TOKEN
# ─────────────────────────────────────────

SECRET_TOKEN = os.getenv("API_SECRET_TOKEN")
api_key_header = APIKeyHeader(name="X-API-Token")

def verifier_token(token: str = Security(api_key_header)):
    if token != SECRET_TOKEN:
        raise HTTPException(status_code=401, detail="Token invalide.")

# Autorise React (localhost:3000) à appeler l'API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Charger Whisper une seule fois au démarrage (pas à chaque requête)
print("Chargement de Whisper...")
whisper_model = whisper.load_model("small")
print("✅ Whisper prêt")

OLLAMA_MODEL = "mistral-nemo"

# ─────────────────────────────────────────
# MODÈLES DE DONNÉES
# ─────────────────────────────────────────

class YoutubeRequest(BaseModel):
    url: str

class SummarizeRequest(BaseModel):
    texte: str
    mode: str = "points"  # "points" ou "complet"

class FlashcardsRequest(BaseModel):
    texte: str
    mode: str = "classique"        # "classique", "qcm", "anglais"
    nb: int = 10                   # 5, 10 ou 15
    niveau_anglais: str = "intermédiaire (B1/B2)"  # pour le mode anglais

class NotesRequest(BaseModel):
    titre: str
    contenu: str

# ─────────────────────────────────────────
# FONCTIONS UTILITAIRES
# ─────────────────────────────────────────

def transcrire(chemin_audio: str) -> dict:
    result = whisper_model.transcribe(chemin_audio, language=None, verbose=False)
    return {
        "texte": result["text"].strip(),
        "langue": result.get("language", "unknown")
    }

def ollama(prompt: str) -> str | None:
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=180
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="Ollama n'est pas lancé.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def extraire_json(raw: str):
    raw = raw.replace("```json", "").replace("```", "").strip()
    debut = raw.find("[")
    fin = raw.rfind("]") + 1
    if debut == -1 or fin == 0:
        raise HTTPException(status_code=500, detail=f"Format JSON inattendu : {raw[:200]}")
    return json.loads(raw[debut:fin])

def markdown_vers_html(texte: str) -> str:
    lignes = texte.split('\n')
    html = []
    in_list = False
    for ligne in lignes:
        if ligne.startswith('### '):
            if in_list: html.append('</ul>'); in_list = False
            html.append(f'<h3>{ligne[4:].strip()}</h3>')
        elif ligne.startswith('## '):
            if in_list: html.append('</ul>'); in_list = False
            html.append(f'<h2>{ligne[3:].strip()}</h2>')
        elif ligne.startswith('# '):
            if in_list: html.append('</ul>'); in_list = False
            html.append(f'<h1>{ligne[2:].strip()}</h1>')
        elif re.match(r'^[-*•]\s+', ligne):
            if not in_list: html.append('<ul>'); in_list = True
            item = re.sub(r'^[-*•]\s+', '', ligne)
            item = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', item)
            html.append(f'<li>{item}</li>')
        elif re.match(r'^\d+\.\s+', ligne):
            if not in_list: html.append('<ul>'); in_list = True
            item = re.sub(r'^\d+\.\s+', '', ligne)
            item = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', item)
            html.append(f'<li>{item}</li>')
        elif ligne.strip() == '':
            if in_list: html.append('</ul>'); in_list = False
            html.append('<br>')
        else:
            if in_list: html.append('</ul>'); in_list = False
            ligne = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', ligne)
            html.append(f'<p>{ligne}</p>')
    if in_list:
        html.append('</ul>')
    return '\n'.join(html)

# ─────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────

@app.post("/transcribe/youtube")
def transcribe_youtube(body: YoutubeRequest, token: str = Security(verifier_token)):
    """Télécharge l'audio d'une vidéo YouTube et le transcrit."""
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio.mp3")
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(tmpdir, 'audio.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'quiet': True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([body.url])
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Erreur téléchargement : {str(e)}")

        if not os.path.exists(audio_path):
            raise HTTPException(status_code=500, detail="Fichier audio introuvable après téléchargement.")

        return transcrire(audio_path)


@app.post("/transcribe/audio")
async def transcribe_audio(file: UploadFile = File(...), token: str = Security(verifier_token)):
    """Transcrit un fichier audio uploadé (mp3, wav, m4a...)."""
    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(file.filename)[1], delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        return transcrire(tmp_path)
    finally:
        os.remove(tmp_path)


@app.post("/summarize")
def summarize(body: SummarizeRequest, token: str = Security(verifier_token)):
    """Génère un résumé en français à partir d'un texte."""
    if body.mode == "points":
        prompt = f"""Voici une transcription audio (peut être en français ou en anglais). Crée une fiche d'étude structurée en FRANÇAIS avec :

1. **Thème principal** : une phrase qui résume le sujet
2. **Idées clés** : les 5-8 points essentiels à retenir (bullet points)
3. **Concepts importants** : termes ou notions à comprendre
4. **À retenir** : la conclusion ou message principal

Sois concis et clair. Réponds uniquement avec la fiche, sans introduction.

Transcription :
{body.texte[:6000]}"""
    else:
        prompt = f"""Voici une transcription audio (peut être en français ou en anglais). Écris un résumé complet et fluide en FRANÇAIS, en paragraphes. Couvre toutes les idées importantes dans l'ordre. Réponds uniquement avec le résumé, sans introduction.

Transcription :
{body.texte[:6000]}"""

    resume = ollama(prompt)
    return {"resume": resume}


@app.post("/flashcards")
def flashcards(body: FlashcardsRequest, token: str = Security(verifier_token)):
    """Génère des flashcards à partir d'un texte."""
    if body.mode == "classique":
        prompt = f"""À partir du texte suivant, génère exactement {body.nb} flashcards de révision en FRANÇAIS.
Réponds UNIQUEMENT avec un tableau JSON valide, sans texte avant ou après :
[
  {{"question": "...", "reponse": "..."}},
  {{"question": "...", "reponse": "..."}}
]

Texte :
{body.texte[:5000]}"""

    elif body.mode == "qcm":
        prompt = f"""À partir du texte suivant, génère exactement {body.nb} questions à choix multiples en FRANÇAIS.
Chaque question doit avoir 4 options (A, B, C, D) dont une seule est correcte.
Réponds UNIQUEMENT avec un tableau JSON valide, sans texte avant ou après :
[
  {{
    "question": "...",
    "choix": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
    "bonne_reponse": "A"
  }}
]

Texte :
{body.texte[:5000]}"""

    elif body.mode == "anglais":
        prompt = f"""Tu es un professeur d'anglais. Analyse cette transcription et repère exactement {body.nb} mots ou expressions difficiles pour un apprenant de niveau {body.niveau_anglais}.
Réponds UNIQUEMENT avec un tableau JSON valide, sans texte avant ou après :
[
  {{
    "mot": "...",
    "contexte": "...",
    "traduction": "...",
    "explication": "...",
    "exemple": "..."
  }}
]

Transcription :
{body.texte[:5000]}"""

    else:
        raise HTTPException(status_code=400, detail="Mode invalide. Utilise : classique, qcm, anglais")

    raw = ollama(prompt)
    cards = extraire_json(raw)
    return {"flashcards": cards}


@app.post("/notes")
def envoyer_notes(body: NotesRequest, token: str = Security(verifier_token)):
    """Envoie un résumé dans Apple Notes (Mac uniquement)."""
    contenu_html = markdown_vers_html(body.contenu)
    contenu_safe = contenu_html.replace('\\', '\\\\').replace('"', '\\"').replace('\r', '')
    titre_safe = body.titre.replace('"', '\\"')
    script = f'tell application "Notes" to make new note at folder "Notes" with properties {{name:"{titre_safe}", body:"{contenu_safe}"}}'
    try:
        subprocess.run(["osascript", "-e", script], check=True)
        return {"status": "ok", "message": "Note créée dans Apple Notes."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur Apple Notes : {str(e)}")


@app.get("/health")
def health():
    return {"status": "ok"}
