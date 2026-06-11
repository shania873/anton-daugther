#!/usr/bin/env python3
"""
Lanceur interactif pour livre_audio.py
Pose les questions, télécharge les clips vocaux depuis YouTube, puis lance la génération.
"""

import sys
import subprocess
from pathlib import Path

# Utilise le Python du venv si disponible (TTS et autres dépendances y sont installées)
_venv = Path(__file__).parent / ".venv" / "bin" / "python"
PYTHON = str(_venv) if _venv.exists() else sys.executable

PIPER_VOIX_LISTE = [
    ("siwis",        "Féminine claire (recommandée)"),
    ("gilles",       "Masculine posée"),
    ("tom",          "Masculine jeune"),
    ("upmc-jessica", "Féminine naturelle (UPMC)"),
    ("upmc-pierre",  "Masculine profonde (UPMC)"),
]

OPENAI_VOIX_LISTE = [
    ("nova",    "Féminine, énergique"),
    ("shimmer", "Féminine, douce"),
    ("alloy",   "Neutre, polyvalente"),
    ("fable",   "Masculine, narrative"),
    ("onyx",    "Masculine, grave"),
    ("echo",    "Masculine, posée"),
]


def lister_livres() -> list[Path]:
    dossier = Path(".")
    return sorted(dossier.glob("*.pdf")) + sorted(dossier.glob("*.epub"))


def choisir(prompt: str, options: list[str]) -> int:
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    while True:
        rep = input(f"\n{prompt} : ").strip()
        if rep.isdigit() and 1 <= int(rep) <= len(options):
            return int(rep) - 1
        print(f"  Entrez un nombre entre 1 et {len(options)}.")


def lister_clips() -> list[Path]:
    dossier = Path("clips")
    if not dossier.exists():
        return []
    return sorted(f for f in dossier.iterdir() if f.suffix in (".wav", ".mp3", ".m4a"))


def choisir_clip(label: str, nom_defaut: str) -> str:
    """Demande à l'utilisateur d'utiliser un clip existant ou d'en télécharger un nouveau."""
    clips_existants = lister_clips()

    print(f"\nVoix pour {label} :")
    options = []
    if clips_existants:
        options.append("Utiliser un clip existant du dossier clips/")
    options.append("Télécharger depuis YouTube")

    idx = choisir("Source", options)

    if clips_existants and idx == 0:
        print(f"\n  Clips disponibles :")
        idx_clip = choisir("Votre choix", [c.name for c in clips_existants])
        chemin = str(clips_existants[idx_clip])
        print(f"   OK — clip sélectionné : {chemin}")
        return chemin
    else:
        url = input("  URL YouTube : ").strip()
        return telecharger_clip(url, nom_defaut)


def telecharger_clip(url: str, nom: str) -> str:
    """Télécharge les 25 premières secondes d'une vidéo YouTube comme clip vocal WAV."""
    import yt_dlp

    dossier = Path("clips")
    dossier.mkdir(exist_ok=True)
    chemin = dossier / f"{nom}.wav"
    tmp    = dossier / f"{nom}_full.%(ext)s"

    print(f"   Téléchargement de la voix '{nom}' depuis YouTube...")

    base_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(tmp),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        "quiet": True,
        "no_warnings": True,
    }

    # YouTube exige des cookies pour confirmer qu'on n'est pas un bot.
    # On essaie les navigateurs dans l'ordre jusqu'à ce que ça marche.
    navigateurs = ["chrome", "safari", "firefox", "edge"]
    telechargé  = False

    for nav in navigateurs:
        try:
            opts = {**base_opts, "cookiesfrombrowser": (nav,)}
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
            telechargé = True
            break
        except Exception:
            continue

    if not telechargé:
        print(f"Erreur : impossible de télécharger depuis YouTube.")
        print("Astuce : ouvre la vidéo dans ton navigateur puis réessaie.")
        sys.exit(1)

    # Trouve le fichier téléchargé
    candidats = [f for f in dossier.glob(f"{nom}_full.*") if f.suffix in (".wav", ".mp3", ".m4a", ".webm")]
    if not candidats:
        print(f"Erreur : impossible de télécharger la voix depuis {url}")
        sys.exit(1)

    fichier_source = candidats[0]

    # Coupe à 25 secondes via ffmpeg (pas besoin de pydub)
    subprocess.run(
        ["ffmpeg", "-i", str(fichier_source), "-t", "25", "-y", str(chemin)],
        capture_output=True, check=True
    )
    fichier_source.unlink(missing_ok=True)

    print(f"   OK — clip prêt ({chemin})")
    return str(chemin)


def main():
    print("\n" + "=" * 40)
    print("        LIVRE AUDIO IA")
    print("=" * 40 + "\n")

    # 1. Choix du livre
    livres = lister_livres()
    if not livres:
        print("Aucun fichier PDF ou EPUB trouvé dans ce dossier.")
        sys.exit(1)

    print("Quel livre voulez-vous traiter ?")
    idx = choisir("Votre choix", [l.name for l in livres])
    fichier = str(livres[idx])
    print(f"\n  Livre selectionne : {livres[idx].name}")

    # 2. Choix du mode
    print("\nQuel mode ?")
    modes = [
        "Livre audio  (lecture classique avec une seule voix)",
        "Podcast      (dialogue entre Marie et Thomas)",
    ]
    idx_mode = choisir("Votre choix", modes)
    mode = "livre" if idx_mode == 0 else "podcast"
    print(f"\n  Mode selectionne : {mode}\n")

    # 2bis. Quels chapitres ?
    print("Quels chapitres voulez-vous generer ?")
    choix_qte = [
        "1 chapitre (test rapide)",
        "3 chapitres",
        "5 chapitres",
        "Tous les chapitres",
        "Choix personnalise (plage ou liste)",
    ]
    idx_qte = choisir("Votre choix", choix_qte)
    spec_chapitres = {0: "1", 1: "3", 2: "5", 3: None}.get(idx_qte)

    if idx_qte == 4:
        print()
        print("  Formats acceptes :")
        print("    1-5      -> chapitres 1 a 5")
        print("    3,7,9    -> chapitres 3, 7 et 9")
        print("    1-3,7,9-12 -> melange (1 a 3, 7, 9 a 12)")
        spec_chapitres = input("\n  Votre selection : ").strip()
        if not spec_chapitres:
            spec_chapitres = None

    # 3. Choix du moteur TTS
    print("Quel moteur de synthese vocale ?")
    moteurs = [
        "Piper       (gratuit, rapide, voix francaises predefinies)",
        "OpenAI TTS  (payant ~7 $/livre, tres naturel, rapide)",
        "XTTS v2     (gratuit, lent, clonage de voix depuis YouTube)",
    ]
    idx_moteur = choisir("Votre choix", moteurs)
    moteur = ["piper", "openai", "xtts"][idx_moteur]
    print(f"\n  Moteur selectionne : {moteur}\n")

    # 4. Configuration des voix selon le moteur
    if moteur == "piper":
        if mode == "livre":
            print("Voix pour la lecture du livre :")
            idx_v = choisir("Votre choix", [f"{n} — {d}" for n, d in PIPER_VOIX_LISTE])
            voix_livre = PIPER_VOIX_LISTE[idx_v][0]
            cmd = [
                PYTHON, "livre_audio.py", fichier,
                "--mode", "livre",
                "--tts", "piper",
                "--piper-voix", voix_livre,
            ]
        else:
            print("Voix de Marie (feminine, energique) :")
            idx_m = choisir("Votre choix", [f"{n} — {d}" for n, d in PIPER_VOIX_LISTE])
            voix_marie = PIPER_VOIX_LISTE[idx_m][0]
            print("\nVoix de Thomas (masculine, posee) :")
            idx_t = choisir("Votre choix", [f"{n} — {d}" for n, d in PIPER_VOIX_LISTE])
            voix_thomas = PIPER_VOIX_LISTE[idx_t][0]
            cmd = [
                PYTHON, "livre_audio.py", fichier,
                "--mode", "podcast",
                "--tts", "piper",
                "--piper-marie", voix_marie,
                "--piper-thomas", voix_thomas,
            ]
    elif moteur == "openai":
        import os
        if not os.environ.get("OPENAI_API_KEY"):
            print("\nERREUR : la variable OPENAI_API_KEY n'est pas definie.")
            print("  1. Va sur https://platform.openai.com/api-keys")
            print("  2. Cree une cle, ajoute du credit (5-10 $)")
            print("  3. Lance dans le terminal :")
            print("       export OPENAI_API_KEY=sk-...")
            sys.exit(1)

        print("Modele de qualite ?")
        modeles = [
            "tts-1     (standard, 0,015 $/1k chars) [RECOMMANDE]",
            "tts-1-hd  (HD, 0,030 $/1k chars, 2x plus cher)",
        ]
        idx_mod = choisir("Votre choix", modeles)
        modele_openai = "tts-1" if idx_mod == 0 else "tts-1-hd"

        if mode == "livre":
            print("\nVoix pour la lecture du livre :")
            idx_v = choisir("Votre choix", [f"{n} — {d}" for n, d in OPENAI_VOIX_LISTE])
            voix_livre = OPENAI_VOIX_LISTE[idx_v][0]
            cmd = [
                PYTHON, "livre_audio.py", fichier,
                "--mode", "livre",
                "--tts", "openai",
                "--openai-voix", voix_livre,
                "--openai-modele", modele_openai,
            ]
        else:
            print("\nVoix de Marie (feminine, energique) :")
            idx_m = choisir("Votre choix", [f"{n} — {d}" for n, d in OPENAI_VOIX_LISTE])
            voix_marie = OPENAI_VOIX_LISTE[idx_m][0]
            print("\nVoix de Thomas (masculine, posee) :")
            idx_t = choisir("Votre choix", [f"{n} — {d}" for n, d in OPENAI_VOIX_LISTE])
            voix_thomas = OPENAI_VOIX_LISTE[idx_t][0]
            cmd = [
                PYTHON, "livre_audio.py", fichier,
                "--mode", "podcast",
                "--tts", "openai",
                "--openai-marie", voix_marie,
                "--openai-thomas", voix_thomas,
                "--openai-modele", modele_openai,
            ]
    else:
        print("-" * 40)
        print("Pour chaque voix : choisissez un clip existant")
        print("ou telechargez-en un depuis YouTube.")
        print("(Un clip = 5 a 30 sec d'une voix claire)")
        print("-" * 40)

        if mode == "livre":
            clip = choisir_clip("la voix du livre", "voix_livre")
            cmd = [
                PYTHON, "livre_audio.py", fichier,
                "--mode", "livre",
                "--tts", "xtts",
                "--clip-voix", clip,
            ]
        else:
            clip_marie  = choisir_clip("Marie (feminine, energique)", "voix_marie")
            clip_thomas = choisir_clip("Thomas (masculine, pose)", "voix_thomas")
            cmd = [
                PYTHON, "livre_audio.py", fichier,
                "--mode", "podcast",
                "--tts", "xtts",
                "--clip-marie", clip_marie,
                "--clip-thomas", clip_thomas,
            ]

    if spec_chapitres:
        cmd += ["--chapitres", spec_chapitres]

    print("\n" + "=" * 40)
    print("  Lancement de la generation...")
    print("  (Ctrl+C pour arreter a tout moment)")
    print("=" * 40 + "\n")
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
