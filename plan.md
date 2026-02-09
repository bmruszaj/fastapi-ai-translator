# Translation App Plan

## 1. Scope
- Build a local web app translating between: `de`, `en`, `el`, `es`, `fr`, `it`, `pl`, `pt`, `ro`, `nl`.
- Deliver backend, frontend, local run setup, Docker run, scalability note, and risk/ethics note.

## 2. Fixed Decisions
- Model: `facebook/nllb-200-distilled-600M`.
- Backend: FastAPI, hexagonal architecture.
- Frontend: static HTML/CSS/JS.
- Dependency management: root `requirements.txt` pinned from `.venv` using `uv pip freeze`.
- Runtime model lifecycle: load once at app startup.

## 3. Backend Structure (Hexagonal)
```text
backend/app/
  main.py
  core/{config.py,dependencies.py}
  domain/{language_rules.py,errors.py}
  application/{dto.py,ports/translator_port.py,use_cases/translate_text.py}
  adapters/
    inbound/http/{schemas.py,routes.py}
    outbound/nllb_translator.py
  bootstrap/container.py
backend/tests/{conftest.py,test_health.py,test_translate_validation.py,test_translate_integration.py}
```

## 4. Backend Implementation
- `domain/language_rules.py`:
  - `SUPPORTED = ["de","en","el","es","fr","it","pl","pt","ro","nl"]`
  - ISO -> NLLB mapping.
  - `validate_language`, `validate_pair`, `to_model_lang`.
- `domain/errors.py`: explicit domain exceptions for validation and limits.
- `application/ports/translator_port.py`:
  - define `TranslatorPort` as `ABC`,
  - `translate(text: str, src: str, tgt: str) -> str` is an `@abstractmethod`.
- `application/use_cases/translate_text.py`:
  - input normalization and validation,
  - invoke translator port,
  - return translated text.
- `adapters/outbound/nllb_translator.py`:
  - implement translator port,
  - manage tokenizer/model inference,
  - no model loading inside request handlers.
- `adapters/inbound/http/routes.py`:
  - `GET /health`
  - `GET /languages`
  - `POST /translate`
  - map domain errors to 400/422; unexpected errors to 500.
- `bootstrap/container.py` + `main.py`:
  - compose adapter + use case,
  - store container/singletons in app state at startup.

## 5. Frontend
- `frontend/index.html`: text input, source/target selectors, submit, output, error area.
- `frontend/app.js`: call `/translate`, handle loading/error/success states.

## 6. Docker
- `backend/Dockerfile` (CPU default):
  - base `python:3.11-slim`,
  - install from root `requirements.txt`,
  - run `uvicorn app.main:app`.
- `backend/Dockerfile.gpu` (optional local GPU run).
- `docker-compose.yml`:
  - backend service, `8000:8000`,
  - optional HF cache mount.

## 7. Documentation Deliverables
- `README.md` must include:
  - business problem approach,
  - architecture,
  - local run steps,
  - Docker run steps,
  - API examples,
  - scalability/deployment approach,
  - risks/bias/ethics,
  - frontend run steps.
- `docs/architecture.md`: component diagram and runtime flow.
- `docs/risks_ethics.md`: model limitations, bias, privacy, human-review constraints.

## 8. Validation and Completion
- Tests:
  - `test_health.py`
  - `test_translate_validation.py`
  - optional `test_translate_integration.py` (`slow` marker).
- Completion criteria:
  - `pytest` passes,
  - `docker compose up --build` works,
  - frontend works with backend locally,
  - no model binaries committed,
  - PR to `main`.
