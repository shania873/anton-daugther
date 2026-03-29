import os
import glob
import json
import requests

# === 1. Choisir le fichier de transcription ===
print("=== Générateur de Flashcards ===\n")

fichiers = sorted(
    glob.glob("transcription_et_resume_*.txt") + glob.glob("enregistrement_et_resume_*.txt"),
    reverse=True
)

if not fichiers:
    print("❌ Aucun fichier de transcription trouvé dans ce dossier.")
    exit(1)

print("Fichiers disponibles :")
for i, f in enumerate(fichiers):
    print(f"  {i + 1} - {f}")

choix = input("\nChoisis un fichier (numéro) : ").strip()
try:
    fichier_choisi = fichiers[int(choix) - 1]
except (ValueError, IndexError):
    print("❌ Choix invalide.")
    exit(1)

print(f"  ➜ {fichier_choisi}\n")

with open(fichier_choisi, 'r', encoding='utf-8') as f:
    contenu = f.read()

# Pour le mode anglais on veut la TRANSCRIPTION complète, pas juste le résumé
if "TRANSCRIPTION COMPLÈTE" in contenu:
    texte_transcription = contenu.split("TRANSCRIPTION COMPLÈTE")[-1].split("RÉSUMÉ")[0].strip()
else:
    texte_transcription = contenu.strip()

if "RÉSUMÉ" in contenu:
    texte_resume = contenu.split("RÉSUMÉ")[-1].strip()
else:
    texte_resume = contenu.strip()

# Détecter la langue depuis les métadonnées du fichier
langue_detectee = "unknown"
for ligne in contenu.splitlines():
    if "Langue:" in ligne:
        langue_detectee = ligne.split("Langue:")[-1].strip().upper()
        break

# === 2. Choisir le mode ===
print("Quel mode de révision ?")
print("  1 - Classique  (question → réponse)")
print("  2 - QCM        (question → 4 choix)")
if langue_detectee == "EN":
    print("  3 - Anglais    (vocabulaire difficile selon ton niveau)")
else:
    print("  3 - Anglais    ⚠️  non disponible (transcription en français)")
mode = input("Ton choix (1 / 2 / 3) : ").strip()

if mode == "3" and langue_detectee != "EN":
    print("\n⚠️  Le mode anglais nécessite une transcription en anglais.")
    print(f"   Ce fichier est en : {langue_detectee}")
    print("   Choisis un fichier avec une vidéo en anglais.\n")
    exit(1)

mode_qcm = (mode == "2")
mode_anglais = (mode == "3")
print(f"  ➜ Mode {'QCM' if mode_qcm else 'anglais' if mode_anglais else 'classique'}\n")

# === 3. Niveau d'anglais (si mode anglais) ===
niveau_anglais = None
if mode_anglais:
    print("Quel est ton niveau d'anglais ?")
    print("  1 - Débutant    (A1/A2) — vocabulaire du quotidien")
    print("  2 - Intermédiaire (B1/B2) — expressions et phrasal verbs")
    print("  3 - Avancé      (C1/C2) — idiomes, nuances, registres")
    choix_niveau = input("Ton niveau (1 / 2 / 3) : ").strip()
    niveaux = {"1": "débutant (A1/A2)", "2": "intermédiaire (B1/B2)", "3": "avancé (C1/C2)"}
    niveau_anglais = niveaux.get(choix_niveau, "intermédiaire (B1/B2)")
    print(f"  ➜ Niveau : {niveau_anglais}\n")

# === 4. Choisir le nombre de flashcards ===
choix_nb = input("Combien de flashcards ? (5 / 10 / 15) : ").strip()
nb_flashcards = {"5": 5, "10": 10, "15": 15}.get(choix_nb, 10)
print(f"  ➜ {nb_flashcards} flashcards\n")

# === 5. Générer avec Ollama ===
print("Génération en cours...")

def extraire_json(raw):
    # Supprimer les blocs markdown ```json ... ``` ou ``` ... ```
    raw = raw.replace("```json", "").replace("```", "").strip()
    debut = raw.find("[")
    fin = raw.rfind("]") + 1
    if debut == -1 or fin == 0:
        return None
    return json.loads(raw[debut:fin])

def ollama(prompt, model="mistral-nemo"):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=180
        )
        response.raise_for_status()
        raw = response.json().get("response", "").strip()
        resultat = extraire_json(raw)
        if resultat is None:
            print("⚠️  Format inattendu dans la réponse d'Ollama.")
            print("   Réponse reçue :", raw[:300])
        return resultat
    except requests.exceptions.ConnectionError:
        print("❌ Ollama n'est pas lancé. Ouvre l'app Ollama et réessaie.")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Impossible de lire le JSON : {e}")
        print("   Réponse reçue :", raw[:300])
        return None
    except Exception as e:
        print(f"❌ Erreur : {e}")
        return None

def generer_classique(texte, nb=10):
    prompt = f"""À partir du texte suivant, génère exactement {nb} flashcards de révision en FRANÇAIS.
Réponds UNIQUEMENT avec un tableau JSON valide, sans texte avant ou après :
[
  {{"question": "...", "reponse": "..."}},
  {{"question": "...", "reponse": "..."}}
]

Texte :
{texte[:5000]}"""
    return ollama(prompt)

def generer_qcm(texte, nb=10):
    prompt = f"""À partir du texte suivant, génère exactement {nb} questions à choix multiples en FRANÇAIS.
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
{texte[:5000]}"""
    return ollama(prompt)

def generer_anglais(texte, nb=10, niveau="intermédiaire (B1/B2)"):
    prompt = f"""Tu es un professeur d'anglais. Analyse cette transcription en anglais et repère exactement {nb} mots, expressions ou structures grammaticales qui seraient difficiles pour un apprenant de niveau {niveau}.

Pour chaque élément, crée une flashcard avec :
- "mot" : le mot ou l'expression en anglais tel qu'il apparaît dans le texte
- "contexte" : la phrase exacte du texte où il apparaît (courte)
- "traduction" : la traduction en français
- "explication" : une explication courte en français (sens, usage, nuance)
- "exemple" : un autre exemple d'utilisation en anglais

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
{texte[:5000]}"""
    return ollama(prompt)

if mode_anglais:
    flashcards = generer_anglais(texte_transcription, nb=nb_flashcards, niveau=niveau_anglais)
elif mode_qcm:
    flashcards = generer_qcm(texte_resume, nb=nb_flashcards)
else:
    flashcards = generer_classique(texte_resume, nb=nb_flashcards)

if not flashcards:
    exit(1)

print(f"✅ {len(flashcards)} flashcards prêtes !\n")

# === 6. Session de révision ===
print("=" * 60)
print("        DÉBUT DE LA SESSION DE RÉVISION")
print("=" * 60)
if mode_anglais:
    print("  → Appuie sur Entrée pour voir la traduction et l'explication")
elif mode_qcm:
    print("  → Tape la lettre de ton choix (A / B / C / D)")
else:
    print("  → Appuie sur Entrée pour voir la réponse")
print("  → Tape 'q' pour quitter")
print("=" * 60 + "\n")

score = 0

for i, card in enumerate(flashcards):

    if mode_anglais:
        print(f"[{i + 1}/{len(flashcards)}] 🇬🇧 \"{card.get('mot', '')}\"")
        print(f"   Contexte : {card.get('contexte', '')}")
        reponse = input("   Entrée pour la traduction... ").strip()
        if reponse.lower() == 'q':
            print("\n👋 Session terminée.")
            break
        print(f"   🇫🇷 Traduction  : {card.get('traduction', '')}")
        print(f"   💡 Explication : {card.get('explication', '')}")
        print(f"   📝 Exemple     : {card.get('exemple', '')}\n")

    elif mode_qcm:
        print(f"[{i + 1}/{len(flashcards)}] ❓ {card.get('question', '')}")
        choix_map = card.get('choix', {})
        for lettre, texte_choix in choix_map.items():
            print(f"   {lettre}) {texte_choix}")
        reponse = input("\n   Ton choix : ").strip().upper()
        if reponse == 'Q':
            print("\n👋 Session terminée.")
            break
        bonne = card.get('bonne_reponse', '').upper()
        if reponse == bonne:
            print(f"   ✅ Bonne réponse !\n")
            score += 1
        else:
            print(f"   ❌ Mauvais. La bonne réponse était : {bonne}) {choix_map.get(bonne, '')}\n")

    else:
        print(f"[{i + 1}/{len(flashcards)}] ❓ {card.get('question', '')}")
        reponse = input("   Entrée pour la réponse... ").strip()
        if reponse.lower() == 'q':
            print("\n👋 Session terminée.")
            break
        print(f"   ✅ {card.get('reponse', '')}\n")

else:
    print("=" * 60)
    if mode_qcm:
        print(f"🎉 Session terminée ! Score : {score}/{len(flashcards)}")
        if score == len(flashcards):
            print("   Parfait, tu maîtrises le sujet !")
        elif score >= len(flashcards) * 0.7:
            print("   Bien joué, continue comme ça !")
        else:
            print("   Continue à réviser, tu vas y arriver !")
    else:
        print(f"🎉 Bravo ! Tu as révisé les {len(flashcards)} flashcards.")
    print("=" * 60)
