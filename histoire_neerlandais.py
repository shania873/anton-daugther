"""
histoire_neerlandais.py
-----------------------
Génère une histoire d'enfant en néerlandais,
la décortique mot par mot avec traduction française,
et envoie le tout dans Apple Notes.

Usage :
    python histoire_neerlandais.py
"""

import re
import requests
import subprocess
from datetime import datetime


# ──────────────────────────────────────────────
# 1. OLLAMA — génération et traduction
# ──────────────────────────────────────────────

MODELE = "mistral-nemo"


def appeler_ollama(prompt, modele=MODELE, timeout=120):
    """Envoie un prompt à Ollama et renvoie la réponse texte."""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": modele, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        print("  ❌ Ollama n'est pas lancé. Ouvre l'app Ollama et réessaie.")
        return None
    except Exception as e:
        print(f"  ❌ Erreur Ollama : {e}")
        return None


def generer_histoire(theme="un petit lapin dans la forêt", nb_phrases=8):
    """Génère une histoire d'enfant en néerlandais (NIVEAU A0)."""
    prompt = f"""Écris une histoire SIMPLE en néerlandais ({nb_phrases} phrases) NIVEAU A0.
Thème : {theme}

IMPORTANT - Respecte ABSOLUMENT :
- Utilise UNIQUEMENT : présent simple, "is/zijn", "heb/hebben"
- Vocabulaire : BASIQUE (maison, chat, chien, pain, eau, soleil, herbe, fleur, forêt, famille)
- Sujet verbe objet uniquement
- PAS de temps complexes
- PAS d'adjectifs compliqués
- Phrases max 12 mots
- Construis une mini-histoire avec début, milieu et fin
- Exemple : "De kat eet brood. De hond speelt in het park. De zon schijnt. Het is een mooie dag."

Réponds UNIQUEMENT avec l'histoire, sans titre ni traduction."""
    return appeler_ollama(prompt)


def decortique_phrase(phrase_nl):
    """
    Pour une phrase néerlandaise, renvoie la décomposition mot par mot
    sous forme de liste de dict : [{"mot": ..., "traduction": ..., "nature": ...}, ...]
    """
    prompt = f"""Voici une phrase en néerlandais : « {phrase_nl} »

Pour chaque mot de cette phrase, donne exactement (une ligne par mot) :
MOT | TRADUCTION_FR | NATURE_GRAMMATICALE

Exemple :
De | Le/La | article
hond | chien | nom
loopt | court/marche | verbe

Réponds UNIQUEMENT avec le tableau, un mot par ligne, sans explication ni en-tête."""

    reponse = appeler_ollama(prompt, timeout=60)
    if not reponse:
        return []

    mots = []
    for ligne in reponse.strip().splitlines():
        parties = [p.strip() for p in ligne.split("|")]
        if len(parties) >= 2:
            mots.append({
                "mot":        parties[0],
                "traduction": parties[1] if len(parties) > 1 else "?",
                "nature":     parties[2] if len(parties) > 2 else "",
            })
    return mots


def decouper_en_phrases(texte):
    """Découpe un texte en phrases (. ! ?)."""
    phrases = re.split(r'(?<=[.!?])\s+', texte.strip())
    return [p.strip() for p in phrases if p.strip()]


# ──────────────────────────────────────────────
# 2. FORMATAGE
# ──────────────────────────────────────────────

def formater_pour_affichage(histoire, analyses):
    """Affichage terminal : histoire + tableau mot par mot."""
    lignes = []
    lignes.append("\n" + "═" * 60)
    lignes.append("📖  HISTOIRE EN NÉERLANDAIS")
    lignes.append("═" * 60)
    lignes.append(histoire)
    lignes.append("\n" + "─" * 60)
    lignes.append("🔍  DÉCORTIQUÉ MOT PAR MOT")
    lignes.append("─" * 60)

    for i, (phrase, mots) in enumerate(analyses, 1):
        lignes.append(f"\n[Phrase {i}]  {phrase}")
        lignes.append(f"  {'MOT':<20} {'TRADUCTION':<25} NATURE")
        lignes.append(f"  {'─'*20} {'─'*25} {'─'*15}")
        for m in mots:
            lignes.append(f"  {m['mot']:<20} {m['traduction']:<25} {m['nature']}")

    return "\n".join(lignes)


def formater_pour_notes(histoire, analyses, theme):
    """Génère le contenu Markdown pour Apple Notes."""
    lignes = []
    lignes.append(f"## 📖 Histoire : {theme}")
    lignes.append(f"*Générée le {datetime.now().strftime('%d/%m/%Y à %H:%M')}*")
    lignes.append("")
    lignes.append(histoire)
    lignes.append("")
    lignes.append("---")
    lignes.append("")
    lignes.append("## 🔍 Décortiqué mot par mot")
    lignes.append("")

    for i, (phrase, mots) in enumerate(analyses, 1):
        lignes.append(f"### Phrase {i}")
        lignes.append(f"**{phrase}**")
        lignes.append("")
        for m in mots:
            nature = f" *({m['nature']})*" if m['nature'] else ""
            lignes.append(f"- **{m['mot']}** → {m['traduction']}{nature}")
        lignes.append("")

    return "\n".join(lignes)


# ──────────────────────────────────────────────
# 3. APPLE NOTES (même logique que les autres scripts)
# ──────────────────────────────────────────────

PARA_EMOJIS = {
    "1 - Projets":    "🎯",
    "2 - Domaines":   "🏠",
    "3 - Ressources": "📚",
    "4 - Archives":   "🗄️",
}


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
            contenu = re.sub(r'^[-*•]\s+', '', ligne)
            contenu = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', contenu)
            contenu = re.sub(r'\*(.+?)\*', r'<i>\1</i>', contenu)
            html.append(f'<li>{contenu}</li>')
        elif ligne.strip() in ('', '---'):
            if in_list: html.append('</ul>'); in_list = False
            html.append('<br>')
        else:
            if in_list: html.append('</ul>'); in_list = False
            ligne = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', ligne)
            ligne = re.sub(r'\*(.+?)\*', r'<i>\1</i>', ligne)
            html.append(f'<p>{ligne}</p>')
    if in_list:
        html.append('</ul>')
    return '\n'.join(html)


def lister_dossiers_notes():
    script = '''tell application "Notes"
        set noms to {}
        repeat with f in folders
            set end of noms to name of f
        end repeat
        return noms
    end tell'''
    try:
        result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
        brut = result.stdout.strip()
        return [d.strip() for d in brut.split(",") if d.strip()]
    except Exception:
        return []


def envoyer_dans_notes(titre, contenu_md, dossier="3 - Ressources"):
    emoji = PARA_EMOJIS.get(dossier, "📝")
    contenu_html = markdown_vers_html(contenu_md)
    contenu_safe = contenu_html.replace('\\', '\\\\').replace('"', '\\"').replace('\r', '')
    titre_safe = f"{emoji} {titre}".replace('"', '\\"')
    dossier_safe = dossier.replace('"', '\\"')

    script = f'''tell application "Notes"
        set targetFolder to missing value
        repeat with f in folders
            if name of f is "{dossier_safe}" then
                set targetFolder to f
                exit repeat
            end if
        end repeat
        if targetFolder is missing value then
            set targetFolder to make new folder with properties {{name:"{dossier_safe}"}}
        end if
        make new note at targetFolder with properties {{name:"{titre_safe}", body:"{contenu_safe}"}}
    end tell'''

    try:
        subprocess.run(["osascript", "-e", script], check=True)
        print(f"  ✅ Note créée dans '{dossier}'")
    except Exception as e:
        print(f"  ⚠️  Impossible de créer la note : {e}")


# ──────────────────────────────────────────────
# 4. PROGRAMME PRINCIPAL
# ──────────────────────────────────────────────

def main():
    print("╔" + "═" * 54 + "╗")
    print("║" + "  🇳🇱  HISTOIRE EN NÉERLANDAIS — Apprendre  ".center(54) + "║")
    print("╚" + "═" * 54 + "╝\n")

    # ── Paramètres ──
    print("  📍 NIVEAU A0 (Débutant absolu)")
    print("  Vocabulaire très simple, phrases courtes\n")
    
    theme = input("📝 Thème (Entrée = chat avec une souris) : ").strip()
    if not theme:
        theme = "de kat en de muis"  # le chat et la souris

    nb_str = input("📏 Nombre de phrases (Entrée = 8) : ").strip()
    nb_phrases = int(nb_str) if nb_str.isdigit() else 8

    # ── ÉTAPE 1 : Générer l'histoire ──
    print("\n" + "=" * 60)
    print("ÉTAPE 1 : GÉNÉRATION DE L'HISTOIRE (Ollama)")
    print("=" * 60)
    print("  ⏳ Génération en cours...")

    histoire = generer_histoire(theme=theme, nb_phrases=nb_phrases)
    if not histoire:
        print("❌ Impossible de générer l'histoire.")
        return

    print("  ✅ Histoire générée !\n")
    print("─" * 60)
    print(histoire)
    print("─" * 60)

    # ── ÉTAPE 2 : Décomposer mot par mot ──
    print("\n" + "=" * 60)
    print("ÉTAPE 2 : DÉCORTICAGE MOT PAR MOT")
    print("=" * 60)

    phrases = decouper_en_phrases(histoire)
    print(f"  📋 {len(phrases)} phrase(s) détectée(s)")

    analyses = []
    for i, phrase in enumerate(phrases, 1):
        print(f"  🔍 Analyse phrase {i}/{len(phrases)} : {phrase[:50]}...")
        mots = decortique_phrase(phrase)
        analyses.append((phrase, mots))

    # Affichage terminal
    print(formater_pour_affichage(histoire, analyses))

    # ── ÉTAPE 3 : Apple Notes ──
    print("\n" + "=" * 60)
    print("ÉTAPE 3 : APPLE NOTES")
    print("=" * 60)

    envoyer = input("\nEnvoyer dans Apple Notes ? (o/n) : ").strip().lower()
    if envoyer in ("o", "oui", "y", "yes"):
        dossiers = lister_dossiers_notes()
        para_noms = ["1 - Projets", "2 - Domaines", "3 - Ressources", "4 - Archives"]
        para_presents = [d for d in para_noms if d in dossiers]
        autres = [d for d in dossiers if d not in para_noms]
        tous = para_presents + autres

        print("\nOù envoyer la note ?")
        print("  0 - ✨ Créer un nouveau dossier")
        for i, nom in enumerate(tous, 1):
            emoji_d = PARA_EMOJIS.get(nom, "📁")
            print(f"  {i} - {emoji_d} {nom}")

        choix = input("\nTon choix : ").strip()
        if choix == "0":
            dossier_choisi = input("Nom du nouveau dossier : ").strip() or "3 - Ressources"
        elif choix.isdigit() and 1 <= int(choix) <= len(tous):
            dossier_choisi = tous[int(choix) - 1]
        else:
            dossier_choisi = "3 - Ressources"

        contenu_md = formater_pour_notes(histoire, analyses, theme)
        titre_note = f"Néerlandais — {theme[:40]} ({datetime.now().strftime('%d/%m/%Y')})"
        envoyer_dans_notes(titre_note, contenu_md, dossier=dossier_choisi)

    print("\n" + "=" * 60)
    print("✅ TERMINÉ ! Bonne révision 🇳🇱")
    print("=" * 60)


if __name__ == "__main__":
    main()
