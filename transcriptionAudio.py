import os
import certifi
from dotenv import load_dotenv

# Charger les variables d'environnement depuis .env
load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

import sounddevice as sd
from scipy.io.wavfile import write
from whisper import load_model

SAMPLE_RATE = 16000
OUTPUT_WAV = "recorded.wav"

print("=== Mini enregistreur IA ===")
print("Appuie sur Entrée pour commencer à enregistrer. Parle, puis appuie à nouveau sur Entrée pour stopper.")
input("Prêt ? Appuie sur Entrée pour démarrer...")

# 1. Démarrer l'enregistrement
print("Enregistrement... Parle maintenant ! (Appuie sur Entrée pour arrêter)")
sd.default.samplerate = SAMPLE_RATE
sd.default.channels = 1
recording = sd.rec(int(60 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="int16")  # 60 sec max
input()  # Attends que tu appuies sur Entrée
sd.stop()
write(OUTPUT_WAV, SAMPLE_RATE, recording)

print("Enregistrement terminé. Transcription en cours...")

# 2. Transcription avec Whisper
print("\n" + "=" * 60)
print("ÉTAPE 2: TRANSCRIPTION")
print("=" * 60)
print("Chargement du modèle Whisper...")
print("  [10%] Initialisation...")
model = load_model("small")  # "tiny", "base", "small", "medium", "large"
print("  [50%] Modèle chargé")

print("  [60%] Transcription en cours...")
result = model.transcribe(OUTPUT_WAV, language=None, verbose=False)
print("  [90%] Traitement des résultats...")
transcript = result["text"]
detected_language = result.get("language", "unknown")
print("  [100%] ✅ Transcription terminée")
print(f"\n📝 Langue détectée: {detected_language.upper()}")
print("\nTranscription (extrait) :", transcript[:500])

# 3. Résumé avec Ollama
import requests

print("\n" + "=" * 60)
print("ÉTAPE 3: RÉSUMÉ")
print("=" * 60)

print("\nQuel type de résumé veux-tu ?")
print("  1 - Fiche par points (thème, idées clés, concepts, à retenir)")
print("  2 - Résumé complet en paragraphes")
choix = input("Ton choix (1 ou 2) : ").strip()
mode_resume = "complet" if choix == "2" else "points"
print(f"  ➜ Mode choisi : {'résumé complet' if mode_resume == 'complet' else 'fiche par points'}")

print("Connexion à Ollama...")

def summarize_with_ollama(text, mode="points", model="mistral-nemo"):
    if mode == "points":
        prompt = f"""Voici une transcription audio (peut être en français ou en anglais). Crée une fiche d'étude structurée en FRANÇAIS avec :

1. **Thème principal** : une phrase qui résume le sujet
2. **Idées clés** : les 5-8 points essentiels à retenir (bullet points)
3. **Concepts importants** : termes ou notions à comprendre
4. **À retenir** : la conclusion ou message principal

Sois concis et clair. Réponds uniquement avec la fiche, sans introduction.

Transcription :
{text[:6000]}"""
    else:
        prompt = f"""Voici une transcription audio (peut être en français ou en anglais). Écris un résumé complet et fluide en FRANÇAIS, en paragraphes. Couvre toutes les idées importantes dans l'ordre. Réponds uniquement avec le résumé, sans introduction.

Transcription :
{text[:6000]}"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        print("  ❌ Ollama n'est pas lancé. Ouvre l'app Ollama et réessaie.")
        return None
    except Exception as e:
        print(f"  ❌ Erreur Ollama : {e}")
        return None

print("  [50%] Résumé en cours (peut prendre 1-2 minutes)...")
final_summary = summarize_with_ollama(transcript, mode=mode_resume)

if final_summary is None:
    final_summary = transcript[:300]

print("\nRésumé :\n", final_summary)

# Optionnel : supprimer le fichier audio
os.remove(OUTPUT_WAV)

# === FONCTION POUR REFORMATER LE TEXTE EN BEAUX PARAGRAPHES ===
def format_text_into_paragraphs(text, sentences_per_para=5):
    sentences = []
    for s in text.replace('!', '.').replace('?', '.').split('.'):
        s = s.strip()
        if s:
            sentences.append(s + '.')
    paragraphs = []
    for i in range(0, len(sentences), sentences_per_para):
        paragraphs.append(' '.join(sentences[i:i+sentences_per_para]))
    return '\n\n'.join(paragraphs)

# 4. Sauvegarder transcription + résumé dans un fichier texte
print("\n" + "=" * 60)
print("ÉTAPE 4: SAUVEGARDE")
print("=" * 60)
print("Sauvegarde en cours...")
from datetime import datetime
output_file = f"enregistrement_et_resume_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"

formatted_text = format_text_into_paragraphs(transcript)
formatted_summary = format_text_into_paragraphs(final_summary)

with open(output_file, 'w', encoding='utf-8') as f:
    f.write("╔" + "═" * 78 + "╗\n")
    f.write("║" + " TRANSCRIPTION ET RÉSUMÉ AUTOMATIQUE ".center(78) + "║\n")
    f.write("╚" + "═" * 78 + "╝\n\n")

    f.write("📋 Métadonnées:\n")
    f.write("-" * 80 + "\n")
    f.write(f"  • Langue: {detected_language.upper()}\n")
    f.write(f"  • Longueur: {len(transcript.split())} mots\n\n")

    f.write("\n" + "─" * 80 + "\n")
    f.write("📝 TRANSCRIPTION COMPLÈTE\n")
    f.write("─" * 80 + "\n\n")
    for para in formatted_text.split('\n\n'):
        f.write("    " + para + "\n\n")

    f.write("\n" + "─" * 80 + "\n")
    f.write("✨ RÉSUMÉ\n")
    f.write("─" * 80 + "\n\n")
    for para in formatted_summary.split('\n\n'):
        f.write("    " + para + "\n\n")

    f.write("\n" + "═" * 80 + "\n")

print(f"[100%] ✅ Fichier sauvegardé: {output_file}")

# === 5. Envoyer le résumé dans Apple Notes ===
import subprocess
import re

def markdown_vers_html(texte):
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
            contenu_item = re.sub(r'^[-*•]\s+', '', ligne)
            contenu_item = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', contenu_item)
            html.append(f'<li>{contenu_item}</li>')
        elif re.match(r'^\d+\.\s+', ligne):
            if not in_list: html.append('<ul>'); in_list = True
            contenu_item = re.sub(r'^\d+\.\s+', '', ligne)
            contenu_item = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', contenu_item)
            html.append(f'<li>{contenu_item}</li>')
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

def envoyer_dans_notes(titre, contenu):
    contenu_html = markdown_vers_html(contenu)
    contenu_safe = contenu_html.replace('\\', '\\\\').replace('"', '\\"').replace('\r', '')
    script = f'tell application "Notes" to make new note at folder "Notes" with properties {{name:"{titre}", body:"{contenu_safe}"}}'
    try:
        subprocess.run(["osascript", "-e", script], check=True)
        print("✅ Note créée dans Apple Notes !")
    except Exception as e:
        print(f"⚠️  Impossible de créer la note : {e}")

print("\n" + "=" * 60)
print("ÉTAPE 5: APPLE NOTES")
print("=" * 60)
note_titre = f"Transcription - {datetime.now().strftime('%d/%m/%Y %H:%M')}"
envoyer_dans_notes(note_titre, final_summary)

print("\n" + "=" * 60)
print("✅ TRAITEMENT TERMINÉ AVEC SUCCÈS!")
print("=" * 60)
print(f"✓ Transcription: {len(transcript.split())} mots")
print(f"✓ Fichier: {output_file}")
print("=" * 60)
