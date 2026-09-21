# TOPAS Studio

TOPAS Studio is a presentable, persistent web workspace for the TOPAS pipeline. It currently implements **late-fusion extraction and refinement** while showing the full product roadmap: extraction, refinement, annotation, RL training, and interactive deployment.

## What is implemented

- Email + access-code entry and returning-user restoration
- Persistent projects, source metadata, progress events, and run history in SQLite
- Domain, system-party, user-party, interaction-unit, and model setup
- Separate multi-PDF upload lanes for system and user textbooks
- Upload either role first, add the other later, or upload both
- Azure model selector: `gpt-5.1`, `gpt-4.1`, and `DeepSeek-V4-Pro`
- The existing `TOPAOurExtractor` **Late** fusion pipeline and its prompts
- Live artifact discovery: book summaries, per-book components, and final fused components appear independently
- Presentational structured renderers for macro actions, micro actions, conversation states, cautions, user profiles, and the knowledge graph
- Profile-scoped component editing with validation, atomic writes, and retained version history
- Full-screen, same-page knowledge-graph workspace with draggable nodes and editable concepts and relationships
- TOPAS refinement with live progress, refined component review/editing, and audit reports
- JSON viewing and download
- Per-user showcase copies seeded from the bundled `demo_data/` directory
- Rules omitted from prompts, extraction, manifests, and the interface

The project is self-contained: the TOPA Python package is bundled in `topa/`
and the CBT showcase seed is stored in `demo_data/`. Running the website does
not require a sibling `TOPA-main` checkout.

## Architecture

```text
React + Vite frontend
        │ authenticated REST + 1.s live polling
        ▼
FastAPI backend ── SQLite profile/project store
        │
        ├── persisted PDF uploads and run artifacts
        └── TOPAOurExtractor (Late fusion) ── Azure AI endpoint
```

The Python backend is intentional: PDF parsing and the long-running TOPAS process cannot execute on GitHub Pages. The frontend can still be deployed to GitHub Pages; point it to a separately hosted API with `VITE_API_BASE`.

The production deployment included in this repository uses a serverless split:
GitHub Pages serves the interface, a Cloudflare Worker exposes the same REST
contract, D1 stores profiles/projects/events, KV stores PDFs and artifacts, and
an authenticated GitHub Actions runner executes the unchanged Python TOPA code.
This keeps Azure credentials out of the browser while allowing the interface to
poll and reveal artifacts as each one is completed.

## Run locally

1. Configure the backend:

```powershell
Copy-Item backend/.env.example backend/.env
```

Set `AZURE_OPENAI_API_KEY` in `backend/.env`. The default local preview access code is `topas-preview`; replace it before sharing the app.

2. Install and start the API from the `website` folder:

```powershell
python -m pip install -r backend/requirements.txt
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

3. In another terminal, install and start the frontend:

```powershell
Set-Location frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`.

## Standalone layout

```text
website/
|-- backend/       FastAPI application and local credentials
|-- frontend/      React application
|-- topa/          bundled TOPA source, prompts, and pipeline code
|-- demo_data/     bundled CBT showcase seed
|-- data/          generated profiles, uploads, and project artifacts
`-- scripts/       local QA utilities
```

You can copy the complete `website/` directory elsewhere and use the same
installation and startup commands. Runtime paths are resolved from that folder.

After `npm --prefix frontend run build`, FastAPI also serves the production bundle at `http://127.0.0.1:8000`, so a one-server local or container deployment is available.

## Container deployment

The root `Dockerfile` builds the React application and serves it from the FastAPI
process. Mount durable storage at `/data`; this holds the SQLite database,
uploaded PDFs, extracted artifacts, and edit history.

```powershell
docker build -t topas-studio .
docker run --rm -p 8000:8000 --env-file backend/.env -v topas-data:/data topas-studio
```

In production, configure these server-side environment variables:

- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_VERSION`
- `TOPAS_MODEL_GPT_5_1`, `TOPAS_MODEL_GPT_4_1`, and `TOPAS_MODEL_DEEPSEEK_V4_PRO`
- `TOPAS_ACCESS_CODE_HASHES` (preferred over a plain access code)
- `TOPAS_TOKEN_SECRET`
- `TOPAS_ALLOWED_ORIGINS`
- `TOPAS_DATA_DIR=/data`

Do not expose any of these through a `VITE_*` variable. The only frontend build
variable is `VITE_API_BASE`, and it is unnecessary when the container serves
both the UI and API from the same origin.

## Access-code configuration

For local work, set a plain code:

```text
TOPAS_ACCESS_CODE=your-local-code
```

For deployment, remove the plain value and provide one or more comma-separated SHA-256 hashes:

```text
TOPAS_ENV=production
TOPAS_ACCESS_CODE=
TOPAS_ACCESS_CODE_HASHES=<sha256-hex>,<another-sha256-hex>
TOPAS_TOKEN_SECRET=<long-random-secret>
```

Azure deployment names can be remapped without changing the labels in the interface:

```text
TOPAS_MODEL_GPT_5_1=gpt-5.1
TOPAS_MODEL_GPT_4_1=gpt-4.1
TOPAS_MODEL_DEEPSEEK_V4_PRO=DeepSeek-V4-Pro
```

## Verification

```powershell
npm --prefix frontend run build
python -m unittest backend.tests.test_api -v
```

The checked API tests cover access validation, project creation, one-sided PDF
upload, bundled showcase loading, component editing, refinement availability,
and complete rules exclusion.

## GitHub Pages frontend

Build with the repository base path and hosted API URL:

```powershell
$env:VITE_BASE_PATH = "/your-repository/"
$env:VITE_API_BASE = "https://your-api.example.org"
npm --prefix frontend run build
```

Deploy `frontend/dist`. Make sure `TOPAS_ALLOWED_ORIGINS` on the API contains the exact GitHub Pages origin. Azure keys and access-code hashes belong only on the backend—never in `VITE_*` variables.

## Production cloud resources

The checked-in deployment definitions are:

- `.github/workflows/deploy_pages.yml` for the React interface
- `.github/workflows/topas_job.yml` for Python extraction/refinement jobs
- `worker/` for the API, D1 schema, and KV bindings
- `scripts/cloud_job.py` for prompt-preserving TOPA execution

Required GitHub Actions secrets:

- `AZURE_OPENAI_API_KEY`
- `TOPAS_JOB_TOKEN` (must match the Worker secret)

Required GitHub Actions variables:

- `VITE_API_BASE`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_VERSION`
- `TOPAS_MODEL_GPT_5_1`
- `TOPAS_MODEL_GPT_4_1`
- `TOPAS_MODEL_DEEPSEEK_V4_PRO`

Required Worker secrets:

- `ACCESS_CODE_HASHES`
- `TOKEN_SECRET`
- `JOB_TOKEN`
- `GITHUB_TOKEN`

Production PDFs are limited to 24 MB each because Cloudflare KV values have a
per-object size ceiling. Local FastAPI uploads retain the configurable local
limit.
