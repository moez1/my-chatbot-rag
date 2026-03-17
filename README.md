## personal-rag

Projet de RAG personnel (FastAPI + Postgres + Docker) pour centraliser ta base de connaissance (code, PDF, notes, etc.).

### Stack technique

- FastAPI / Pydantic
- Postgres (+ extensions pour vecteurs)
- Alembic (migrations)
- Docker / Docker Compose
- Intégration multi-providers LLM : OpenAI, Anthropic (Claude), Deepseek

### Structure prévue

- `app/` : code applicatif (API, RAG, providers, stockage)
- `config/settings.yaml` : configuration non sensible
- `.env` : secrets locaux (non commit)
- `docker-compose.yml` : orchestrer API + Postgres

