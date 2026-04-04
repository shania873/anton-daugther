import os
import subprocess
from pathlib import Path

PROJECTS_DIR = Path.home() / "Documents" / "Projects"

# ─── Couleurs ANSI ───
RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
GRAY   = "\033[90m"
WHITE  = "\033[97m"
BG_GREEN  = "\033[42m"
BG_RED    = "\033[41m"
BG_YELLOW = "\033[43m"
BG_DARK   = "\033[100m"

def clear():
    os.system("clear")

def header():
    print(f"{BOLD}{BG_DARK}                                                  {RESET}")
    print(f"{BOLD}{BG_DARK}   🛠  StudyAI — Lanceur de projets .NET 9         {RESET}")
    print(f"{BOLD}{BG_DARK}                                                  {RESET}")
    print()

def choisir_projet():
    # Trouver tous les projets .NET dans Documents/Projects
    projets = []
    for dossier in sorted(PROJECTS_DIR.iterdir()):
        if dossier.is_dir():
            csprojs = list(dossier.rglob("*.csproj"))
            if csprojs:
                projets.append((dossier.name, csprojs[0].parent))

    if not projets:
        print(f"{RED}❌ Aucun projet .NET trouvé dans {PROJECTS_DIR}{RESET}")
        exit(1)

    print(f"{CYAN}{BOLD}  Projets disponibles :{RESET}")
    print(f"  {GRAY}{'─' * 40}{RESET}")
    for i, (nom, _) in enumerate(projets):
        print(f"  {GRAY}[{i+1}]{RESET}  📁  {WHITE}{nom}{RESET}")
    print(f"  {GRAY}{'─' * 40}{RESET}")

    choix = input(f"\n  {BOLD}Choisis un projet : {RESET}").strip()
    try:
        nom, chemin = projets[int(choix) - 1]
        return nom, chemin
    except (ValueError, IndexError):
        print(f"{RED}❌ Choix invalide.{RESET}")
        exit(1)

def menu_action(nom_projet):
    print(f"\n  {BOLD}Projet : {GREEN}{nom_projet}{RESET}\n")
    print(f"  {BG_GREEN}{BOLD}  ▶   Exécuter       {RESET}  {GRAY}[1]{RESET}")
    print(f"  {BG_YELLOW}{BOLD}  🐛  Debug           {RESET}  {GRAY}[2]{RESET}")
    print(f"  {BG_DARK}{BOLD}  🔨  Build seulement {RESET}  {GRAY}[3]{RESET}")
    print(f"  {RED}  ✕   Quitter        {RESET}  {GRAY}[q]{RESET}")
    print()

    choix = input(f"  {BOLD}Ton choix : {RESET}").strip().lower()
    return choix

def trouver_url(chemin):
    """Lit le launchSettings.json pour trouver l'URL du projet."""
    launch = chemin / "Properties" / "launchSettings.json"
    if not launch.exists():
        return None
    try:
        import json
        data = json.loads(launch.read_text())
        profiles = data.get("profiles", {})
        for profile in profiles.values():
            urls = profile.get("applicationUrl", "")
            for url in urls.split(";"):
                url = url.strip()
                if url.startswith("http://"):
                    return url
        return None
    except Exception:
        return None

def ouvrir_swagger(url):
    """Ouvre Edge avec le Swagger après un délai pour laisser le serveur démarrer."""
    import time, threading
    swagger_url = url.rstrip("/") + "/swagger"

    def _ouvrir():
        time.sleep(4)
        subprocess.Popen(
            ["open", "-a", "Microsoft Edge", swagger_url],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        print(f"\n  🌐 Edge ouvert : {swagger_url}")

    threading.Thread(target=_ouvrir, daemon=True).start()

def lancer(chemin, config, debug=False):
    clear()
    header()

    if debug:
        print(f"  {YELLOW}{BOLD}🐛 Mode DEBUG — {chemin.name}{RESET}")
    else:
        print(f"  {GREEN}{BOLD}▶  Lancement — {chemin.name}{RESET}")

    print(f"  {GRAY}{'─' * 50}{RESET}\n")

    env = os.environ.copy()
    if debug:
        env["LOGGING__LOGLEVEL__DEFAULT"] = "Debug"
        env["LOGGING__LOGLEVEL__MICROSOFT"] = "Debug"

    cmd = f"dotnet run --configuration {config} --framework net9.0"
    if debug:
        cmd += " --verbosity detailed"

    # Ouvrir Swagger dans Edge si c'est une webapi
    url = trouver_url(chemin)
    if url:
        ouvrir_swagger(url)
        print(f"  {GRAY}→ Swagger : {url.rstrip('/')}/swagger{RESET}\n")

    process = subprocess.Popen(
        cmd, shell=True, cwd=chemin, env=env
    )
    try:
        process.wait()
    except KeyboardInterrupt:
        process.terminate()
        print(f"\n\n  {YELLOW}⏹  Arrêté par l'utilisateur.{RESET}\n")
        return

    if process.returncode == 0:
        print(f"\n  {GREEN}✅ Terminé avec succès.{RESET}\n")
    else:
        print(f"\n  {RED}❌ Terminé avec le code {process.returncode}.{RESET}\n")

def build_seulement(chemin):
    clear()
    header()
    print(f"  {CYAN}{BOLD}🔨 Build — {chemin.name}{RESET}")
    print(f"  {GRAY}{'─' * 50}{RESET}\n")

    result = subprocess.run("dotnet build --framework net9.0", shell=True, cwd=chemin)
    if result.returncode == 0:
        print(f"\n  {GREEN}✅ Build réussi.{RESET}\n")
    else:
        print(f"\n  {RED}❌ Build échoué.{RESET}\n")

# ─── Main ───
while True:
    clear()
    header()
    nom_projet, chemin_projet = choisir_projet()

    clear()
    header()
    choix = menu_action(nom_projet)

    if choix == "1":
        lancer(chemin_projet, config="Release")
    elif choix == "2":
        lancer(chemin_projet, config="Debug", debug=True)
    elif choix == "3":
        build_seulement(chemin_projet)
    elif choix == "q":
        clear()
        print(f"\n  {GRAY}Au revoir !{RESET}\n")
        break
    else:
        print(f"\n  {RED}Choix invalide.{RESET}\n")

    input(f"\n  {GRAY}Appuie sur Entrée pour revenir au menu...{RESET}")
