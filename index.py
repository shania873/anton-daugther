import os
import certifi
from dotenv import load_dotenv

# Charger les variables d'environnement depuis .env
load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from pytube import YouTube
import whisper
from transformers import pipeline
import yt_dlp

# === Paramètres ===
VIDEO_URL = "https://www.youtube.com/watch?v=SaHHgzoXceU&t=231s"   # <-- Remplace par ton lien

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
print("\n" + "=" * 60)
print("ÉTAPE 2: TRANSCRIPTION")
print("=" * 60)
print("Chargement du modèle Whisper...")
print("  [10%] Initialisation...")
model = whisper.load_model("large")   # 'tiny', 'base', 'small', 'medium', 'large'
print("  [50%] Modèle chargé")

print("  [60%] Transcription en cours... (ça peut prendre quelques minutes)")
result = model.transcribe('audio.mp3', language=None, verbose=False)  # Auto language detection
print("  [90%] Traitement des résultats...")
full_text = result["text"]
detected_language = result.get("language", "unknown")
print("  [100%] ✅ Transcription terminée")
print(f"\n📝 Langue détectée: {detected_language.upper()}")

# === 2.5 AMÉLIORATION DE LA TRANSCRIPTION ===
print("\n" + "=" * 60)
print("ÉTAPE 2.5: AMÉLIORATION DU TEXTE")
print("=" * 60)
print("Formatage et correction en cours...")
print("  [10%] Initialisation...")

try:
    # Charger le modèle de ponctuation
    print("  [30%] Chargement du modèle de ponctuation...")
    punct_model = pipeline("text2text-generation", model="oliverguhr/fullstop-punctuation-multilingual-sobert")
    
    # Découper en chunks et appliquer la ponctuation
    chunks = [full_text[i:i+300] for i in range(0, len(full_text), 300)]
    corrected_chunks = []
    
    for i, chunk in enumerate(chunks):
        pct = 40 + int((i / len(chunks)) * 40)
        print(f"  [{pct}%] Correction chunk {i+1}/{len(chunks)}")
        try:
            # Appliquer la ponctuation
            result = punct_model(chunk, max_length=512, num_beams=2, do_sample=False)
            corrected_text = result[0]['generated_text']
            corrected_chunks.append(corrected_text)
        except Exception as e:
            print(f"    ⚠️  Erreur: {str(e)[:50]}")
            # Fallback: application basique de majuscules et ponctuation
            corrected = chunk.strip()
            if corrected and not corrected.endswith(('.', '!', '?')):
                corrected += '.'
            corrected_chunks.append(corrected)
    
    full_text = " ".join(corrected_chunks).strip()
    
    # Appliquer des améliorations basiques
    # Capitaliser le début des phrases
    sentences = full_text.replace('? ', '?\\n').replace('! ', '!\\n').replace('. ', '.\\n').split('\\n')
    sentences = [s.strip() for s in sentences if s.strip()]
    formatted_sentences = []
    for sent in sentences:
        if sent:
            # Capitaliser la première lettre
            sent = sent[0].upper() + sent[1:] if len(sent) > 1 else sent.upper()
            formatted_sentences.append(sent)
    
    full_text = " ".join(formatted_sentences)
    
    print("  [90%] Finalisation...")
    print("  [100%] ✅ Texte amélioré")
    print("\n📝 Transcription améliorée (extrait) :", full_text[:500])
except Exception as e:
    print(f"  ⚠️  Amélioration échouée: {str(e)[:100]}")
    print("  ➜ Utilisation de la transcription originale")

# === 3. TRADUCTION (si anglais) ===
if detected_language == "en":
    print("\n" + "=" * 60)
    print("ÉTAPE 3: TRADUCTION ANGLAIS → FRANÇAIS")
    print("=" * 60)
    print("Traduction en cours...")
    print("  [10%] Chargement du modèle de traduction...")
    try:
        # Essayer d'abord Helsinki-NLP
        try:
            translator = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")
            print("  [20%] Modèle Helsinki-NLP chargé")
        except Exception as e1:
            print(f"  ⚠️  Helsinki-NLP échoue: {str(e1)[:80]}")
            print("  [20%] Essai alternative...")
            from transformers import MarianMTModel, MarianTokenizer
            model_name = "Helsinki-NLP/opus-mt-en-fr"
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            print("  [20%] Modèle alternatif chargé")
            translator = None
        
        print("  [30%] Modèle chargé")
        
        # Découper le texte en chunks (max 512 chars pour le modèle)
        chunks = [full_text[i:i+512] for i in range(0, len(full_text), 512)]
        translated_chunks = []
        
        for i, chunk in enumerate(chunks):
            pct = 30 + int((i / len(chunks)) * 60)
            print(f"  [{pct}%] Traduction chunk {i+1}/{len(chunks)}")
            try:
                if translator:
                    result_translation = translator(chunk, max_length=512)
                    if isinstance(result_translation, list) and len(result_translation) > 0:
                        translated_text = result_translation[0].get('translation_text', chunk)
                    else:
                        translated_text = chunk
                else:
                    # Utiliser pipeline manuel
                    inputs = tokenizer(chunk, return_tensors="pt", max_length=512, truncation=True)
                    translated = model.generate(**inputs)
                    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
                
                translated_chunks.append(translated_text)
            except Exception as e:
                print(f"    ⚠️  Erreur chunk {i+1}: {str(e)[:60]}")
                translated_chunks.append(chunk)
        
        full_text = " ".join(translated_chunks)
        print("  [90%] Finalisation...")
        print("  [100%] ✅ Traduction terminée")
        print("\n📝 Transcription traduite (extrait) :", full_text[:500])
    except Exception as e:
        print(f"  ❌ Traduction échouée: {str(e)[:100]}")
        print("  ➜ Utilisation de la transcription originale en anglais")
else:
    print("\n📝 Transcription (extrait) :", full_text[:500])

# === 4. Résumé avec un modèle open source ===
print("\n" + "=" * 60)
print("ÉTAPE 4: RÉSUMÉ")
print("=" * 60)
print("Chargement du modèle de résumé...")
print("  [10%] Initialisation...")
model_name = "t5-small"

print("  [30%] Modèle chargé")
try:
    summarizer = pipeline("summarization", model=model_name)
except KeyError:
    try:
        summarizer = pipeline("text-generation", model=model_name)
    except KeyError:
        summarizer = None

print("  [50%] Résumé en cours...")
# On découpe le texte en morceaux si trop long
chunks = [full_text[i:i+1024] for i in range(0, len(full_text), 1024)]
summaries = []

if summarizer:
    for i, chunk in enumerate(chunks):
        pct = 50 + int((i / len(chunks)) * 40)
        print(f"  [{pct}%] Résumé du chunk {i+1}/{len(chunks)}")
        try:
            output = summarizer(chunk, max_length=150, min_length=50, do_sample=False)
            if isinstance(output, list) and len(output) > 0:
                summary = output[0].get('summary_text', output[0].get('generated_text', chunk[:100]))
            else:
                summary = str(output)[:150]
            summaries.append(summary)
        except Exception as e:
            print(f"  Erreur lors du résumé d'un chunk: {e}")
            summaries.append(chunk[:200])
else:
    summaries = [full_text[:200]]

final_summary = "\n\n".join(summaries)
print("  [90%] Traitement du résumé final...")

# === FONCTION POUR FORMATER LE TEXTE EN PARAGRAPHES ===
def format_text_into_paragraphs(text, width=100):
    """Formate le texte en paragraphes lisibles avec sauts de ligne appropriés."""
    paragraphs = []
    
    # Diviser en phrases
    sentences = text.replace('.', '.\n').replace('!', '!\n').replace('?', '?\n').split('\n')
    current_para = []
    current_length = 0
    
    for i, sent in enumerate(sentences):
        sent = sent.strip()
        if not sent:
            if current_para:
                paragraphs.append(' '.join(current_para))
                current_para = []
                current_length = 0
            continue
        
        sent_with_space = sent + ' ' if i < len(sentences) - 1 else sent
        
        if current_length + len(sent_with_space) > width and current_para:
            paragraphs.append(' '.join(current_para))
            current_para = [sent]
            current_length = len(sent)
        else:
            current_para.append(sent)
            current_length += len(sent_with_space)
    
    if current_para:
        paragraphs.append(' '.join(current_para))
    
    return '\n\n'.join(paragraphs)

# === 5. Sauvegarder transcription + résumé dans un fichier texte ===
print("\n" + "=" * 60)
print("ÉTAPE 5: SAUVEGARDE")
print("=" * 60)
print("Sauvegarde en cours...")
output_file = "transcription_et_resume.txt"

# Formater les textes en paragraphes
formatted_text = format_text_into_paragraphs(full_text)
formatted_summary = format_text_into_paragraphs(final_summary)

with open(output_file, 'w', encoding='utf-8') as f:
    f.write("╔" + "═" * 78 + "╗\n")
    f.write("║" + " TRANSCRIPTION ET RÉSUMÉ AUTOMATIQUE ".center(78) + "║\n")
    f.write("╚" + "═" * 78 + "╝\n\n")
    
    f.write("📋 Métadonnées:\n")
    f.write("-" * 80 + "\n")
    f.write(f"  • URL: {VIDEO_URL}\n")
    f.write(f"  • Langue: {detected_language.upper()}\n")
    f.write(f"  • Longueur: {len(full_text.split())} mots\n\n")
    
    f.write("\n" + "─" * 80 + "\n")
    f.write("📝 TRANSCRIPTION COMPLÈTE\n")
    f.write("─" * 80 + "\n\n")
    # Ajouter indentation aux paragraphes
    for para in formatted_text.split('\n\n'):
        f.write("    " + para + "\n\n")
    
    f.write("\n" + "─" * 80 + "\n")
    f.write("✨ RÉSUMÉ\n")
    f.write("─" * 80 + "\n\n")
    # Ajouter indentation aux paragraphes du résumé
    for para in formatted_summary.split('\n\n'):
        f.write("    " + para + "\n\n")
    
    f.write("\n" + "═" * 80 + "\n")

print(f"[100%] ✅ Fichier sauvegardé: {output_file}")

print("\n" + "=" * 60)
print("✅ TRAITEMENT TERMINÉ AVEC SUCCÈS!")
print("=" * 60)
print(f"✓ Transcription: {len(full_text.split())} mots")
print(f"✓ Fichier: {output_file}")
print("=" * 60)
