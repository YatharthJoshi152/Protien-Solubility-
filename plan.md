# E. coli Protein Solubility Predictor plan

## Implementation approach

The project is a browser client backed by a Python FastAPI service. The client sends sequence-derived and manually entered features over HTTP. The service loads joblib artifacts created by the training script and uses the single shared `backend/features.py` function for sequence calculations. If a model-card field is absent, the UI displays `Not available` instead of inventing a metric.

The supplied repository is the data source. `data/features_all.csv` is the only training table, and `data/source_README.md` records the dataset and model protocol. The runtime never loads the legacy random-forest pickle. Training uses gene-grouped holdout and cross-validation, then writes the model artifacts, ranges, categories, feature order, and model card under `artifacts/`.

The Vite React client and FastAPI server run together in development with one command. Vite proxies API calls to FastAPI. The production Docker image builds the client, then FastAPI serves the compiled client and the API on the configured port.

## Design direction

- **Design movement:** quiet laboratory instrument, influenced by research software and technical field notebooks rather than a marketing landing page.
- **Core principles:** evidence before decoration, dense but readable controls, explicit uncertainty, and accessible flat color encoding.
- **Color philosophy:** near-black ink on a cool white work surface, deep teal for actions and focus, and three distinct muted band colors paired with text labels so color is never the only signal.
- **Layout paradigm:** a left-aligned instrument console with a narrow information rail and a broad working surface. The predictor uses a two-column reading order on desktop and a single vertical sequence on mobile.
- **Signature elements:** monospaced numeric readouts, thin ruled section headers, and a score scale with a physical marker and labeled bands.
- **Interaction philosophy:** every state change is explained in text. Sequence calculations lock derived fields until the user chooses `Edit manually`; edits are visibly marked. Motion is limited to short focus and hover transitions and respects reduced-motion preferences.
- **Animation:** no scroll choreography or large movement. Use only 150ms color and border transitions for focus, hover, validation, and tab state.
- **Typography system:** a system sans-serif stack for labels and explanatory copy, with `ui-monospace` for scores, probabilities, sequences, and feature values. Headings use weight and spacing rather than decorative display type.
- **Brand essence:** a transparent screening instrument for researchers checking protein solubility risk in E. coli. Personality: measured, precise, candid.
- **Brand voice:** short, direct, and explicit about uncertainty. Example lines: `Predicted probability of measured solubility at least 50 percent.` and `Model disagreement makes this score less reliable.`
- **Wordmark and mark:** the wordmark is paired with a small outlined three-bar instrument mark representing the three model outputs. It is drawn in CSS and does not claim affiliation with any company.
- **Signature brand color:** deep teal `#0b5b5b`, reserved for primary actions, links, focus outlines, and the instrument mark.

## Project structure

- `src/`: React entry point, application routes, form state, API calls, and CSS.
- `public/manus-routes.json`: declared page routes for the managed website.
- `backend/features.py`: copied source-of-truth sequence feature function.
- `backend/main.py`: FastAPI API, artifact loading, validation, scoring, batch CSV responses, and static serving.
- `backend/tests/`: feature parity and API contract tests.
- `training/build_dataset.py`: reproducible source table checks and feature parity report.
- `training/train_ensemble.py`: gene-grouped training, OOF stacking, held-out evaluation, and artifact export.
- `training/model_saving_snippet.py`: notebook-ready model persistence example.
- `artifacts/`: generated joblib models and JSON metadata used by the runtime.
- `data/`: supplied data and report, plus a small batch example.
- `Dockerfile` and `scripts/dev.py`: production and one-command development workflows.

## Complete build workflow

1. Install Python dependencies from `backend/requirements.txt` and frontend dependencies from the pinned package manifest.
2. Run `python training/build_dataset.py` to verify table structure and sequence-feature parity.
3. Run `python training/train_ensemble.py` to create all runtime artifacts and the data-derived model card.
4. Run `python scripts/dev.py` for FastAPI on port 8000 and Vite on port 3000, or build the client and use the Docker entrypoint for production.
5. Run the backend tests, frontend build, route manifest request, and repository policy scans before checkpointing.
