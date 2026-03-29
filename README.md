# StudyAI

Transcription audio, résumés et flashcards générés par IA.

## Lancer l'API

```bash
uvicorn api:app --reload
```

L'API tourne sur `http://localhost:8000`
La doc interactive est disponible sur `http://localhost:8000/docs`

## Prérequis

- [Ollama](https://ollama.com) doit être lancé avec le modèle `mistral-nemo`
- Un fichier `.env` avec les variables suivantes :

```
API_SECRET_TOKEN=ton_token_ici
HF_TOKEN=ton_token_huggingface
```
