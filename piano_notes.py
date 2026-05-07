"""
Reconnaissance de notes de piano avec librosa + Ollama
------------------------------------------------------
1. Charge un fichier audio (.wav, .mp3, etc.)
2. Détecte les notes jouées (fréquences → noms de notes)
3. Envoie la séquence à Ollama pour analyse musicale
"""

import librosa
import numpy as np
import requests
import json
import sys

# === Configuration ===
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"  # Change selon les modèles installés sur ton Ollama


# === 1. Détection des notes ===

def frequence_vers_note(freq_hz):
    """Convertit une fréquence (Hz) en nom de note musical (ex: La4, Do5)."""
    if freq_hz <= 0:
        return None

    # La4 = 440 Hz, référence standard
    demi_tons = round(12 * np.log2(freq_hz / 440.0))
    noms = ["La", "La#", "Si", "Do", "Do#", "Ré", "Ré#", "Mi", "Fa", "Fa#", "Sol", "Sol#"]

    index_note = demi_tons % 12
    octave = 4 + (demi_tons + 9) // 12  # +9 car La est la 10e note (index 9 depuis Do)
    return f"{noms[index_note]}{octave}"


def detecter_notes(fichier_audio, duree_min_note=0.1, seuil_confiance=0.8):
    """
    Analyse un fichier audio et retourne la liste des notes détectées.

    Args:
        fichier_audio: chemin vers le fichier (.wav, .mp3, .ogg, etc.)
        duree_min_note: durée minimale d'une note pour être gardée (en secondes)
        seuil_confiance: entre 0 et 1, filtre les détections peu fiables

    Returns:
        Liste de tuples (temps_en_sec, nom_note, frequence_hz)
    """
    print(f"Chargement du fichier : {fichier_audio}")
    audio, taux_echantillonnage = librosa.load(fichier_audio, sr=None, mono=True)

    print("Détection des hauteurs (pitch)...")
    # pyin : algorithme robuste pour la détection de hauteur monophonique
    frequences, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),   # Do2 ~ 65 Hz (limite basse piano)
        fmax=librosa.note_to_hz("C8"),   # Do8 ~ 4186 Hz (limite haute piano)
        sr=taux_echantillonnage
    )

    # Convertir les indices de temps en secondes
    temps = librosa.times_like(frequences, sr=taux_echantillonnage)

    notes_detectees = []
    note_courante = None
    debut_note = None

    for t, freq, flag, prob in zip(temps, frequences, voiced_flag, voiced_probs):
        if flag and prob >= seuil_confiance and freq is not None and not np.isnan(freq):
            note = frequence_vers_note(freq)
            if note != note_courante:
                # Nouvelle note détectée
                if note_courante is not None and (t - debut_note) >= duree_min_note:
                    notes_detectees.append((round(debut_note, 2), note_courante))
                note_courante = note
                debut_note = t
        else:
            # Silence ou note peu fiable
            if note_courante is not None and debut_note is not None:
                if (t - debut_note) >= duree_min_note:
                    notes_detectees.append((round(debut_note, 2), note_courante))
            note_courante = None
            debut_note = None

    return notes_detectees


# === 2. Analyse avec Ollama ===

def analyser_avec_ollama(notes):
    """
    Envoie la séquence de notes à Ollama pour une analyse musicale.

    Args:
        notes: liste de tuples (temps, note)

    Returns:
        Texte d'analyse généré par Ollama
    """
    if not notes:
        return "Aucune note détectée dans l'audio."

    # Formater la séquence de notes pour Ollama
    sequence = ", ".join([f"{note} (à {t}s)" for t, note in notes])

    prompt = f"""Tu es un expert en musique. Voici une séquence de notes de piano détectées dans un enregistrement :

{sequence}

Analyse cette séquence et réponds en français :
1. Quelle gamme ou tonalité semble être utilisée ?
2. Est-ce que tu reconnais un motif musical (arpège, gamme, mélodie connue) ?
3. Des observations sur le rythme ou la structure ?
Sois concis et pédagogique."""

    print("\nEnvoi à Ollama pour analyse...")
    try:
        reponse = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=60
        )
        reponse.raise_for_status()
        return reponse.json()["response"]
    except requests.exceptions.ConnectionError:
        return "Erreur : Ollama n'est pas lancé. Lance-le avec la commande : ollama serve"
    except requests.exceptions.Timeout:
        return "Erreur : Ollama a mis trop de temps à répondre."
    except Exception as e:
        return f"Erreur Ollama : {e}"


# === 3. Programme principal ===

def main():
    # Récupérer le fichier depuis les arguments ou demander
    if len(sys.argv) > 1:
        fichier = sys.argv[1]
    else:
        fichier = input("Chemin vers le fichier audio (ex: piano.wav) : ").strip()

    # Détecter les notes
    try:
        notes = detecter_notes(fichier)
    except FileNotFoundError:
        print(f"Erreur : fichier '{fichier}' introuvable.")
        return
    except Exception as e:
        print(f"Erreur lors de l'analyse audio : {e}")
        return

    # Afficher les notes
    print(f"\n=== {len(notes)} notes détectées ===")
    if notes:
        for temps, note in notes:
            print(f"  {temps:6.2f}s  →  {note}")
    else:
        print("  Aucune note claire détectée. Vérifie que l'audio contient bien du piano.")
        return

    # Analyse Ollama
    analyse = analyser_avec_ollama(notes)
    print("\n=== Analyse musicale (Ollama) ===")
    print(analyse)


if __name__ == "__main__":
    main()
