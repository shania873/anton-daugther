"""
flashcards_neerlandais.py
------------------------
Génère des flashcards en néerlandais NIVEAU A0.
Envoie dans Apple Notes.

Usage :
    python flashcards_neerlandais.py
"""

import subprocess
from datetime import datetime
import re


# ──────────────────────────────────────────────
# 1. FLASHCARDS NIVEAU A0
# ──────────────────────────────────────────────

FLASHCARDS_A0 = {
    "CHIFFRES": [
        ("nul", "zéro"),
        ("een", "un"),
        ("twee", "deux"),
        ("drie", "trois"),
        ("vier", "quatre"),
        ("vijf", "cinq"),
        ("zes", "six"),
        ("zeven", "sept"),
        ("acht", "huit"),
        ("negen", "neuf"),
        ("tien", "dix"),
    ],
    "COULEURS": [
        ("rood", "rouge"),
        ("blauw", "bleu"),
        ("geel", "jaune"),
        ("groen", "vert"),
        ("wit", "blanc"),
        ("zwart", "noir"),
        ("oranje", "orange"),
        ("roze", "rose"),
    ],
    "ANIMAUX": [
        ("kat", "chat"),
        ("hond", "chien"),
        ("vogel", "oiseau"),
        ("vis", "poisson"),
        ("konijntje", "petit lapin"),
        ("muis", "souris"),
        ("koe", "vache"),
        ("paard", "cheval"),
    ],
    "OBJETS QUOTIDIENS": [
        ("huis", "maison"),
        ("boom", "arbre"),
        ("bloem", "fleur"),
        ("water", "eau"),
        ("brood", "pain"),
        ("appel", "pomme"),
        ("tafel", "table"),
        ("stoel", "chaise"),
        ("boek", "livre"),
        ("pen", "stylo"),
    ],
    "PARTIES DU CORPS": [
        ("hoofd", "tête"),
        ("oog", "oeil"),
        ("neus", "nez"),
        ("mond", "bouche"),
        ("oor", "oreille"),
        ("hand", "main"),
        ("voet", "pied"),
        ("been", "jambe"),
    ],
    "VERBES SIMPLES": [
        ("eten", "manger"),
        ("drinken", "boire"),
        ("slapen", "dormir"),
        ("spelen", "jouer"),
        ("lopen", "marcher"),
        ("kijken", "regarder"),
        ("luisteren", "écouter"),
        ("spreken", "parler"),
    ],
    "EXPRESSIONS BASIQUES": [
        ("Hoi", "Bonjour / Salut"),
        ("Goedemorgen", "Bon matin"),
        ("Goedenacht", "Bonne nuit"),
        ("Tot ziens", "Au revoir"),
        ("Dank je", "Merci"),
        ("Alsjeblieft", "S'il te plaît"),
        ("Ja", "Oui"),
        ("Nee", "Non"),
    ],
    "PHRASES SIMPLES": [
        ("Ik ben [naam]", "Je m'appelle [nom]"),
        ("Ik hou van [iets]", "J'aime [quelque chose]"),
        ("Ik eet brood", "Je mange du pain"),
        ("De kat speelt", "Le chat joue"),
        ("Je bent aardig", "Tu es gentil"),
        ("Het is mooi weer", "Le temps est beau"),
        ("Ik ben moe", "Je suis fatigué"),
        ("Hoe gaat het?", "Comment ça va?"),
    ],
}


# ──────────────────────────────────────────────
# 2. FORMATAGE MARKDOWN
# ──────────────────────────────────────────────

def generer_markdown_flashcards():
    """Génère le contenu Markdown des flashcards."""
    lignes = []
    lignes.append("## 🇳🇱 Flashcards Néerlandais — NIVEAU A0")
    lignes.append(f"*Générées le {datetime.now().strftime('%d/%m/%Y à %H:%M')}*")
    lignes.append("")
    lignes.append("### 📚 Contenu")
    lignes.append("")

    for i, categorie in enumerate(FLASHCARDS_A0.keys(), 1):
        nb_cartes = len(FLASHCARDS_A0[categorie])
        lignes.append(f"{i}. **{categorie}** ({nb_cartes} cartes)")

    lignes.append("")
    lignes.append("---")
    lignes.append("")

    # Contenu détaillé
    for categorie, cartes in FLASHCARDS_A0.items():
        lignes.append(f"## {categorie}")
        lignes.append("")
        for nl, fr in cartes:
            lignes.append(f"### Carte")
            lignes.append(f"**Avant (Néerlandais):** {nl}")
            lignes.append(f"**Arrière (Français):** {fr}")
            lignes.append("")

    return "\n".join(lignes)


def generer_csv_flashcards():
    """Génère un CSV pour Anki ou autre app."""
    lignes = []
    lignes.append("Néerlandais,Français,Catégorie")

    for categorie, cartes in FLASHCARDS_A0.items():
        for nl, fr in cartes:
            # Échappe les guillemets
            nl_safe = f'"{nl}"'
            fr_safe = f'"{fr}"'
            cat_safe = f'"{categorie}"'
            lignes.append(f"{nl_safe},{fr_safe},{cat_safe}")

    return "\n".join(lignes)


# ──────────────────────────────────────────────
# 3. MARKDOWN → HTML (conversion pour Notes)
# ──────────────────────────────────────────────

def markdown_vers_html(texte):
    lignes = texte.split('\n')
    html = []
    in_list = False

    for ligne in lignes:
        if ligne.startswith('### Carte'):
            if in_list: html.append('</ul>')
            in_list = False
            html.append('<h3 style="margin-top: 20px; color: #0066cc;">Carte</h3>')

        elif ligne.startswith('### '):
            if in_list: html.append('</ul>')
            in_list = False
            html.append(f'<h3>{ligne[4:].strip()}</h3>')

        elif ligne.startswith('## '):
            if in_list: html.append('</ul>')
            in_list = False
            html.append(f'<h2 style="color: #334455; margin-top: 30px;">{ligne[3:].strip()}</h2>')

        elif ligne.startswith('# '):
            if in_list: html.append('</ul>')
            in_list = False
            html.append(f'<h1>{ligne[2:].strip()}</h1>')

        elif ligne.startswith('**Avant'):
            if in_list: html.append('</ul>')
            in_list = False
            contenu = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', ligne)
            html.append(f'<p style="color: #ff6600;"><b>{contenu[2:-2]}</b></p>')

        elif ligne.startswith('**Arrière'):
            if in_list: html.append('</ul>')
            in_list = False
            contenu = re.sub(r'\*\*(.+?)\*\*', r'<b style="color: green;">\1</b>', ligne)
            html.append(f'<p>{contenu[2:-2]}</p>')

        elif re.match(r'^[-*•]\s+', ligne):
            if not in_list: html.append('<ul>')
            in_list = True
            contenu = re.sub(r'^[-*•]\s+', '', ligne)
            contenu = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', contenu)
            html.append(f'<li>{contenu}</li>')

        elif ligne.strip() in ('', '---'):
            if in_list: html.append('</ul>')
            in_list = False
            html.append('<br/>')

        else:
            if in_list: html.append('</ul>')
            in_list = False
            ligne = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', ligne)
            ligne = re.sub(r'\*(.+?)\*', r'<i>\1</i>', ligne)
            if ligne.strip():
                html.append(f'<p>{ligne}</p>')

    if in_list:
        html.append('</ul>')

    return '\n'.join(html)


# ──────────────────────────────────────────────
# 4. APPLE NOTES
# ──────────────────────────────────────────────

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
    contenu_html = markdown_vers_html(contenu_md)
    contenu_safe = contenu_html.replace('\\', '\\\\').replace('"', '\\"').replace('\r', '')
    titre_safe = titre.replace('"', '\\"')
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
        return True
    except Exception as e:
        print(f"  ⚠️  Impossible de créer la note : {e}")
        return False


# ──────────────────────────────────────────────
# 5. PROGRAMME PRINCIPAL
# ──────────────────────────────────────────────

def main():
    print("╔" + "═" * 54 + "╗")
    print("║" + "  🇳🇱  FLASHCARDS NÉERLANDAIS A0  ".center(54) + "║")
    print("╚" + "═" * 54 + "╝\n")

    # Total de cartes
    nb_total = sum(len(cartes) for cartes in FLASHCARDS_A0.values())
    print(f"📊 Total : {nb_total} flashcards répartis en {len(FLASHCARDS_A0)} catégories\n")

    # Affichage de la liste
    print("📚 Catégories :")
    for i, (cat, cartes) in enumerate(FLASHCARDS_A0.items(), 1):
        print(f"   {i}. {cat} ({len(cartes)} cartes)")

    print("\n" + "=" * 60)
    print("ÉTAPE 1 : AFFICHAGE TERMINAL")
    print("=" * 60)

    # Affiche quelques exemples
    print("\n📋 Exemples (premières cartes de chaque catégorie):\n")
    for cat, cartes in FLASHCARDS_A0.items():
        if cartes:
            nl, fr = cartes[0]
            print(f"  {cat:25} → {nl:20} | {fr}")

    # Demande d'action
    print("\n" + "=" * 60)
    print("ÉTAPE 2 : EXPORT")
    print("=" * 60)

    actions = input("""
Que veux-tu faire ?
  1 - Envoyer dans Apple Notes
  2 - Sauver en CSV (pour Anki, etc.)
  3 - Les deux
  0 - Rien (quitter)

Ton choix : """).strip()

    if actions in ("1", "3"):
        print("\n" + "─" * 60)
        print("EXPORT APPLE NOTES")
        print("─" * 60)

        md = generer_markdown_flashcards()

        dossiers = lister_dossiers_notes()
        para_noms = ["1 - Projets", "2 - Domaines", "3 - Ressources", "4 - Archives"]
        para_presents = [d for d in para_noms if d in dossiers]
        autres = [d for d in dossiers if d not in para_noms]
        tous = para_presents + autres

        print("\nOù envoyer la note ?")
        print("  0 - ✨ Créer un nouveau dossier")
        for i, nom in enumerate(tous, 1):
            emojis = {"3 - Ressources": "📚", "1 - Projets": "🎯", "2 - Domaines": "🏠"}
            emoji = emojis.get(nom, "📁")
            print(f"  {i} - {emoji} {nom}")

        choix = input("\nTon choix : ").strip()
        if choix == "0":
            dossier_choisi = input("Nom du nouveau dossier : ").strip() or "3 - Ressources"
        elif choix.isdigit() and 1 <= int(choix) <= len(tous):
            dossier_choisi = tous[int(choix) - 1]
        else:
            dossier_choisi = "3 - Ressources"

        titre = f"🇳🇱 Flashcards A0 ({datetime.now().strftime('%d/%m/%Y')})"
        envoyer_dans_notes(titre, md, dossier=dossier_choisi)

    if actions in ("2", "3"):
        print("\n" + "─" * 60)
        print("EXPORT CSV")
        print("─" * 60)

        csv = generer_csv_flashcards()
        nom_fichier = "flashcards_neerlandais_A0.csv"
        try:
            with open(nom_fichier, "w", encoding="utf-8") as f:
                f.write(csv)
            print(f"  ✅ Fichier créé : {nom_fichier}")
            print(f"  📂 Localisation : ./{nom_fichier}")
        except Exception as e:
            print(f"  ❌ Erreur : {e}")

    print("\n" + "=" * 60)
    print("✅ TERMINÉ ! Bonne révision 🇳🇱")
    print("=" * 60)


if __name__ == "__main__":
    main()
