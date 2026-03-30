# Agentic-RAG-Rust-Core-PFE-26
Prototype RAG minimal qui charge des PDFs, découpe le texte en Rust, génère des embeddings avec
`sentence-transformers` (bge-small-en-v1.5), stocke les vecteurs dans LanceDB, et répond aux questions à partir du contexte récupéré via OpenRouter.

**Fonctionnalités**
- Ingestion de PDFs depuis `pdfs/`
- Découpage intelligent basé sur Rust (`rag_rust`)
- Chargeur de PDFs en lot via Rust (un seul appel, multi-fichiers) avec parallélisme
- Recherche vectorielle avec LanceDB (gérée en Rust)
- Embeddings locaux avec `sentence-transformers`
- Génération de réponses LLM optionnelle avec OpenRouter

**Structure du projet**
- `api.py` : Serveur FastAPI pour le frontend
- `rag_rust/` : Module d'extension Rust (PyO3)
- `pdfs/` : Fichiers PDF locaux (ignorés par git)
- `lancedb/` : Stockage vectoriel local (ignoré par git)

**Installation**
1. Créer un environnement virtuel (optionnel).
2. Installer les dépendances Python :
   `pip install -r requirements.txt`
3. Définir votre clé OpenRouter :
   `setx OPENROUTER_API_KEY "<VOTRE_CLÉ>"`
   Optionnel (requis pour certains modèles gratuits) :
   `setx OPENROUTER_HTTP_REFERER "http://localhost:3000,https://agentic-rag-rust-core-frontend-pfe.vercel.app/"`
   `setx OPENROUTER_TITLE "Agentic-RAG-Rust-Core-PFE-26"`
4. Compiler l'extension Rust :
   `cd rag_rust`
   `maturin develop`
5. Placer vos PDFs dans `pdfs/`.

**CORS (Frontend)**
- Correction rapide en développement :
  `setx CORS_ALLOW_ALL true`
- Ou de façon explicite :
  `setx CORS_ORIGINS "https://agentic-rag-rust-core-frontend-pfe.vercel.app/,http://localhost:3000,http://127.0.0.1:3000"`

**Hugging Face (optionnel)**
- En cas d'erreur 401 lors du téléchargement du modèle d'embedding :
  `setx HF_TOKEN "<VOTRE_TOKEN_HF>"`

**Lancement (API)**
`uvicorn api:app --reload`

**Remarques**
- Assurez-vous que `OPENROUTER_API_KEY` est défini avant de lancer l'API.
- Utilisez `POST /index` pour reconstruire la base de données vectorielle.

**Endpoints de l'API**
- `GET /health`
  - Vérification de l'état du serveur.
- `POST /documents`
  - Envoyer un ou plusieurs fichiers PDF (formulaire multipart).
- `GET /documents`
  - Lister les PDFs uploadés.
- `POST /index`
  - Construire ou reconstruire l'index vectoriel à partir des PDFs uploadés.
  - Corps : `{"rebuild": true, "max_pages": null}`
- `POST /query`
  - Interroger l'index et appeler optionnellement OpenRouter pour une réponse.
  - Corps :
    `{"question": "...", "top_k": 3, "chat_model": "openrouter/free", "use_llm": true}`

**Exemples de requêtes**
```bash
curl -X POST http://localhost:8000/documents \
  -F "files=@./data/pdfs/doc1.pdf" \
  -F "files=@./data/pdfs/doc2.pdf"

curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"rebuild": true, "max_pages": null}'

curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"Quel est le sujet ?", "top_k":3, "use_llm":true}'
```