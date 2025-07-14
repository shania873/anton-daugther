from pytube import YouTube
import whisper
from transformers import pipeline
import os
import yt_dlp

# === Paramètres ===
VIDEO_URL = "https://www.youtube.com/watch?v=VdYQfUbjUtY"   # <-- Remplace par ton lien

# === 1. Télécharger l'audio YouTube ===
print("Téléchargement de la vidéo...")
try:
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'audio.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([VIDEO_URL])
    print("Audio téléchargé : audio.mp3")
    out_file = 'audio.mp3'
except Exception as e:
    print("Erreur lors du téléchargement :", e)
    exit(1)

# === 2. Transcription speech-to-text (Whisper) ===
print("Transcription de l'audio (ça peut prendre quelques minutes)...")
model = whisper.load_model("base")   # 'tiny', 'base', 'small', 'medium', 'large'
result = model.transcribe('audio.mp3', language=None)  # Auto language detection
full_text = result["text"]
print("\nTranscription (extrait) :", full_text[:500])

# === 3. Résumé avec un modèle open source ===
print("Résumé en cours...")
# Si la vidéo est en anglais, modèle Bart; sinon mT5 pour le français/multilingue
# Ici on détecte la langue, tu peux forcer si besoin
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")


# On découpe le texte en morceaux si trop long
chunks = [full_text[i:i+900] for i in range(0, len(full_text), 900)]
summaries = []
for chunk in chunks:
    summary = summarizer(chunk, max_length=120, min_length=40, do_sample=False)[0]['summary_text']
    summaries.append(summary)

final_summary = "\n".join(summaries)
print("\nRésumé :\n", final_summary)

# Optionnel : supprimer le fichier audio
os.remove('audio.mp3')
