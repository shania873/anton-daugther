"""
doc_scraper.py
--------------
Récupère la documentation d'un site web et génère une fiche de révision
facile à retenir, avec Ollama + sauvegarde dans Apple Notes.

Usage :
    python doc_scraper.py
    → puis entre l'URL de la doc quand demandé
"""

import os
import re
import requests
import subprocess
from datetime import datetime
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# 1. SCRAPING — récupérer le texte de la doc
# ──────────────────────────────────────────────

BALISES_INUTILES = ["script", "style", "nav", "footer", "header", "aside", "form"]

def nettoyer_html(soup):
    """Supprime le bruit (nav, scripts…) et extrait le texte utile."""
    for tag in soup(BALISES_INUTILES):
        tag.decompose()

    # Cherche la zone principale de contenu
    zone = (
        soup.find("main")
        or soup.find("article")
        or soup.find(id=re.compile(r"content|main|doc", re.I))
        or soup.find(class_=re.compile(r"content|main|doc", re.I))
        or soup.body
    )
    if not zone:
        return ""

    lignes = []
    for el in zone.find_all(["h1", "h2", "h3", "h4", "p", "li", "code", "pre"]):
        texte = el.get_text(separator=" ", strip=True)
        if texte:
            if el.name in ("h1", "h2", "h3", "h4"):
                lignes.append(f"\n## {texte}\n")
            elif el.name in ("code", "pre"):
                lignes.append(f"`{texte}`")
            else:
                lignes.append(texte)
    return "\n".join(lignes)


def scraper_page(url, session):
    """Télécharge une page et renvoie son texte nettoyé."""
    try:
        r = session.get(url, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        return nettoyer_html(soup), soup
    except Exception as e:
        print(f"  ⚠️  Impossible de charger {url} : {e}")
        return "", None


def trouver_liens_doc(soup, base_url, domaine):
    """Trouve tous les liens internes qui ressemblent à de la doc."""
    if not soup:
        return []
    liens = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        url_complete = urljoin(base_url, href)
        parsed = urlparse(url_complete)
        # Garde uniquement les liens du même domaine, sans ancre pure
        if parsed.netloc == domaine and not parsed.fragment:
            liens.add(url_complete.split("#")[0])
    return list(liens)


def scraper_doc(url_depart, max_pages=20):
    """
    Parcourt le site de documentation à partir de l'URL de départ.
    Renvoie le texte combiné de toutes les pages visitées.
    """
    domaine = urlparse(url_depart).netloc
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (compatible; DocScraper/1.0)"

    visites = set()
    a_visiter = [url_depart]
    pages = []  # liste de (url, texte)

    print(f"\n  🌐 Domaine : {domaine}")
    print(f"  📄 Pages max : {max_pages}\n")

    while a_visiter and len(visites) < max_pages:
        url = a_visiter.pop(0)
        if url in visites:
            continue

        print(f"  [{len(visites)+1}/{max_pages}] {url}")
        texte, soup = scraper_page(url, session)
        visites.add(url)

        if texte:
            pages.append((url, texte))

        # Ajoute les nouveaux liens trouvés
        if soup:
            nouveaux = trouver_liens_doc(soup, url, domaine)
            for lien in nouveaux:
                if lien not in visites and lien not in a_visiter:
                    a_visiter.append(lien)

    print(f"\n  ✅ {len(visites)} page(s) récupérée(s)")
    return pages, list(visites)


# ──────────────────────────────────────────────
# 2. RÉSUMÉ — Ollama
# ──────────────────────────────────────────────

def resumer_avec_ollama(texte, mode="fiche", modele="mistral-nemo"):
    """Envoie le texte à Ollama et renvoie la fiche ou le résumé."""

    texte_tronque = texte[:12000]

    if mode == "fiche":
        prompt = f"""Tu es un expert en pédagogie. Voici la documentation d'un outil.
Crée une FICHE DE RÉVISION structurée en FRANÇAIS avec :

1. **Outil** : Nom + en une phrase à quoi ça sert
2. **Problème résolu** : Pourquoi on utilise cet outil ? (2-3 lignes)
3. **Concepts clés** : Les 5-8 notions fondamentales à comprendre (bullet points)
4. **Commandes / fonctions essentielles** : Les choses les plus utilisées, avec une courte explication
5. **Workflow typique** : Les étapes d'une utilisation classique (numérotées)
6. **Pièges à éviter** : 2-3 erreurs fréquentes des débutants
7. **À retenir en 1 phrase** : Le message principal de cette doc

Réponds UNIQUEMENT avec la fiche, sans introduction ni conclusion.

Documentation :
{texte_tronque}"""
    else:
        prompt = f"""Voici la documentation d'un outil. Écris un résumé complet en FRANÇAIS, en paragraphes fluides.
Couvre : le but de l'outil, ses concepts principaux, et comment l'utiliser.
Réponds UNIQUEMENT avec le résumé.

Documentation :
{texte_tronque}"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": modele, "prompt": prompt, "stream": False},
            timeout=180
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        print("  ❌ Ollama n'est pas lancé. Ouvre l'app Ollama et réessaie.")
        return None
    except Exception as e:
        print(f"  ❌ Erreur Ollama : {e}")
        return None


def resumer_toutes_pages(pages, mode="fiche", modele="mistral-nemo"):
    """Résume chaque page séparément, puis fait un résumé final de tout."""
    if not pages:
        return None

    if len(pages) == 1:
        return resumer_avec_ollama(pages[0][1], mode=mode, modele=modele)

    # Étape 1 : mini-résumé de chaque page
    mini_resumes = []
    for i, (url, texte) in enumerate(pages, 1):
        print(f"  📝 Résumé page {i}/{len(pages)} : {url}")
        mini = resumer_avec_ollama(texte, mode="complet", modele=modele)
        if mini:
            mini_resumes.append(f"--- Page {i} ({url}) ---\n{mini}")

    if not mini_resumes:
        return None

    # Étape 2 : résumé final à partir de tous les mini-résumés
    print(f"\n  🔗 Fusion de {len(mini_resumes)} résumés en une fiche finale...")
    texte_combine = "\n\n".join(mini_resumes)
    return resumer_avec_ollama(texte_combine, mode=mode, modele=modele)


# ──────────────────────────────────────────────
# 3. APPLE NOTES (repris de index.py)
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
            html.append(f'<li>{contenu}</li>')
        elif re.match(r'^\d+\.\s+', ligne):
            if not in_list: html.append('<ul>'); in_list = True
            contenu = re.sub(r'^\d+\.\s+', '', ligne)
            contenu = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', contenu)
            html.append(f'<li>{contenu}</li>')
        elif ligne.strip() == '':
            if in_list: html.append('</ul>'); in_list = False
            html.append('<br>')
        else:
            if in_list: html.append('</ul>'); in_list = False
            ligne = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', ligne)
            ligne = re.sub(r'`(.+?)`', r'<code>\1</code>', ligne)
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


def envoyer_dans_notes(titre, contenu, dossier="3 - Ressources"):
    emoji = PARA_EMOJIS.get(dossier, "📝")
    contenu_html = markdown_vers_html(contenu)
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
    print("╔" + "═" * 50 + "╗")
    print("║" + "  📖 DOC SCRAPER — Fiche de révision IA  ".center(50) + "║")
    print("╚" + "═" * 50 + "╝\n")

    # ── URL ──
    url = input("🌐 URL de la documentation : ").strip()
    if not url.startswith("http"):
        url = "https://" + url

    # ── Nombre de pages ──
    nb_pages_str = input("📄 Nombre de pages max à lire (Entrée = 15) : ").strip()
    max_pages = int(nb_pages_str) if nb_pages_str.isdigit() else 15

    # ── Mode résumé ──
    print("\nQuel format de sortie ?")
    print("  1 - Fiche de révision structurée (recommandé)")
    print("  2 - Résumé complet en paragraphes")
    choix_mode = input("Ton choix (1 ou 2) : ").strip()
    mode = "complet" if choix_mode == "2" else "fiche"

    # ── ÉTAPE 1 : Scraping ──
    print("\n" + "=" * 60)
    print("ÉTAPE 1 : RÉCUPÉRATION DE LA DOC")
    print("=" * 60)
    pages, pages_visitees = scraper_doc(url, max_pages=max_pages)

    if not pages:
        print("❌ Aucun texte récupéré. Vérifie l'URL et réessaie.")
        return

    total_mots = sum(len(t.split()) for _, t in pages)
    print(f"\n  📊 Texte extrait : {total_mots} mots sur {len(pages)} page(s)")

    # ── ÉTAPE 2 : Résumé IA ──
    print("\n" + "=" * 60)
    print("ÉTAPE 2 : RÉSUMÉ AVEC L'IA (Ollama)")
    print("=" * 60)
    print(f"  ⏳ Résumé de {len(pages)} page(s) en cours (peut prendre plusieurs minutes)...")

    fiche = resumer_toutes_pages(pages, mode=mode)

    if not fiche:
        print("❌ Impossible de générer le résumé.")
        return

    print("  ✅ Résumé généré !\n")
    print("─" * 60)
    print(fiche)
    print("─" * 60)

    # ── ÉTAPE 3 : Sauvegarde fichier ──
    print("\n" + "=" * 60)
    print("ÉTAPE 3 : SAUVEGARDE")
    print("=" * 60)

    nom_outil = urlparse(url).netloc.replace("www.", "")
    pages_visitees = [u for u, _ in pages]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fichier = f"doc_resume_{nom_outil}_{timestamp}.txt"

    with open(fichier, "w", encoding="utf-8") as f:
        f.write(f"SOURCE : {url}\n")
        f.write(f"DATE   : {datetime.now().strftime('%d/%m/%Y %H:%M')}\n")
        f.write(f"PAGES  : {len(pages_visitees)}\n")
        f.write("\n" + "─" * 60 + "\n")
        f.write("FICHE DE RÉVISION\n")
        f.write("─" * 60 + "\n\n")
        f.write(fiche)
        f.write("\n\n" + "─" * 60 + "\n")
        f.write("PAGES VISITÉES\n")
        f.write("─" * 60 + "\n")
        for p in pages_visitees:
            f.write(f"  • {p}\n")

    print(f"  ✅ Fichier sauvegardé : {fichier}")

    # ── ÉTAPE 4 : Apple Notes (optionnel) ──
    print("\n" + "=" * 60)
    print("ÉTAPE 4 : APPLE NOTES")
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
            emoji = PARA_EMOJIS.get(nom, "📁")
            print(f"  {i} - {emoji} {nom}")

        choix = input("\nTon choix : ").strip()
        if choix == "0":
            dossier_choisi = input("Nom du nouveau dossier : ").strip() or "3 - Ressources"
        elif choix.isdigit() and 1 <= int(choix) <= len(tous):
            dossier_choisi = tous[int(choix) - 1]
        else:
            dossier_choisi = "3 - Ressources"

        titre_note = f"Doc {nom_outil} — {datetime.now().strftime('%d/%m/%Y')}"
        envoyer_dans_notes(titre_note, fiche, dossier=dossier_choisi)

    print("\n" + "=" * 60)
    print("✅ TERMINÉ !")
    print("=" * 60)


if __name__ == "__main__":
    main()
