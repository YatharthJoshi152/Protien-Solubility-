# E. coli Protein Solubility Predictor

This is a scientific screening tool for estimating whether a protein is soluble when expressed in E. coli. Enter a protein name and parameters, or paste an amino acid sequence and calculate its parameters. The service scores the input with logistic regression, a calibrated RBF SVM, and a random forest, then combines the three probabilities.

The score is the predicted probability that the protein is soluble when expressed in E. coli. In the training data, soluble means measured solubility of at least 50 percent. It is not a predicted solubility percentage and it is not a lab guarantee. The models are trained on E. coli K-12 proteins only.

## Run locally

Install Python 3.11, Node 22, pnpm, and the dependencies:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r backend/requirements.txt
corepack enable
pnpm install
```

Create or refresh the artifacts from the supplied table:

```bash
python training/build_dataset.py
python training/train_ensemble.py
```

Start both services with one command:

```bash
python scripts/dev.py
```

Open `http://localhost:3000`. Vite serves the client and proxies `/meta`, `/features`, `/predict`, and `/predict_batch` to FastAPI on port 8000.

## API

- `GET /meta` returns feature order, category labels, training ranges, library versions, and the model card.
- `POST /features` accepts `{ "sequence": "..." }` and returns the shared feature function output.
- `POST /predict` accepts a protein name, parameters, categories, optional sequence, and edited field names.
- `POST /predict_batch` accepts a UTF-8 CSV upload. The sample input columns are documented in `data/sample_batch.csv`.

## Training protocol

`data/features_all.csv` is the only training table. Labeled rows are split by `B_number` with a 20 percent gene-grouped holdout. The remaining rows use five grouped folds for out-of-fold probabilities. The stacker is a logistic regression trained only on those out-of-fold probabilities. It is used only when its grouped cross-validation ROC-AUC beats every base model; otherwise the final score is the simple average. The three production models are then refit on all labeled rows.

The runtime never loads the legacy random-forest pickle. The shared sequence code is `backend/features.py`, and it is used by both the parity checker and the training script.

## Model-card data

All accuracy, ROC-AUC, precision, recall, and F1 values shown in the interface come from `artifacts/model_card.json`, which is written by the training script. If a metric is absent, the interface says `Not available`.

Known limitations are shown in the interface: the training set contains E. coli K-12 proteins only, related proteins can sit on both sides of a gene-grouped split, and the unlabeled set contains more membrane proteins than the labeled training set, which makes those scores less reliable.

## Test and build

```bash
pytest -q backend/tests
pnpm typecheck
pnpm build
```

The container build compiles the Vite client and starts FastAPI on port 3000. The health endpoint is `/health`.
