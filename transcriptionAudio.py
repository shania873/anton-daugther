import sounddevice as sd
from scipy.io.wavfile import write
import whisper
from transformers import pipeline

SAMPLE_RATE = 16000
OUTPUT_WAV = "recorded.wav"

print("=== Mini enregistreur IA ===")
print("Appuie sur Entrée pour commencer à enregistrer. Parle, puis appuie à nouveau sur Entrée pour stopper.")
input("Prêt ? Appuie sur Entrée pour démarrer...")

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
model = whisper.load_model("base")  # "tiny", "base", "small", etc.
result = model.transcribe(OUTPUT_WAV, language=None)
transcript = result["text"]
print("\nTranscription :", transcript[:500])

# 3. Résumé automatique
print("Résumé en cours...")
if result['language'] == 'fr':
    summarizer = pipeline("summarization", model="csebuetnlp/mT5_multilingual_XLSum")
else:
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# On découpe le texte si trop long (au cas où)
chunks = [transcript[i:i+900] for i in range(0, len(transcript), 900)]
summaries = []
for chunk in chunks:
    summary = summarizer(chunk, max_length=120, min_length=40, do_sample=False)[0]['summary_text']
    summaries.append(summary)

final_summary = "\n".join(summaries)
print("\nRésumé :\n", final_summary)
# Optionnel : supprimer le fichier audio
os.remove(OUTPUT_WAV)