# Protein solubility predictor: project package

Read this file first. It explains every file in this package and the order to work in.
The website itself is described in the main prompt. This file covers the data, the models and the checks.

## Goal
Train logistic regression, SVM and random forest on ONE shared dataset, with ONE feature function and ONE gene-grouped split. Stack them into a final solubility score from 0.0 (not soluble) to 1.0 (fully soluble), and serve it through the website. Soluble means measured solubility of 50 percent or higher in E. coli.

## Files in this package
data/features_all.csv
  Ready-to-use table, 4,087 E. coli K-12 proteins: 3,151 labeled (has_solubility_label = True) and 936 unlabeled. Use it as the only data file.
  Columns: B_number, JW_ID, Gene_name, has_solubility_label, Solubility_pct, Soluble (1 = soluble, 0 = not, blank for unlabeled), MW_kDa, pI, Cell_location, Gene_type, the 20 amino acid percentages (A_pct to Y_pct), Length, GRAVY, Aromaticity, Instability_Index, Aliphatic_Index, Hydrophobic_pct, Charged_pct, Polar_pct, Acidic_pct, Basic_pct, log_MW, log_Length, pI_dist7, Sequence.
  Train and test ONLY on labeled rows. Unlabeled rows are for the batch-mode demo and are never used to fit anything.
data/data_report.txt
  How the table was built: what was dropped and why, class balance, and the location mix of labeled versus unlabeled proteins. Show the limitations from this report on the "How this works" page.
code/features.py
  The single feature function. Import it in training AND in the backend. Do not rewrite it. Run `python code/features.py` to run its self-test.
reference/logreg_notebook.ipynb, reference/svm_notebook_mst.ipynb, reference/rf_notebook.ipynb
  The author's earlier notebooks. Read them for model settings and ideas only. Do NOT reuse their data loading, splits or feature code. Specific notes below.

## Feature rules (already applied in features_all.csv, and implemented in code/features.py)
- Aliphatic_Index = A + 2.9*V + 3.9*(I + L), each term a percentage of length. (An older script left leucine out; that is wrong and is not used here.)
- Hydrophobic_pct = A, I, L, M, F, W, V, Y. Charged_pct = D, E, K, R. Polar_pct = S, T, N, Q. Acidic = D + E. Basic = K + R.
- For a protein typed in by a user: validate the sequence (20 standard letters, at least 10 residues) and calculate MW and pI from it (compute_features returns MW_kDa_calc and pI_calc). Tell the user these two values were calculated. If the user types the numbers manually, label them "entered by user".
- Models:
  * Random forest and SVM: 27 numeric features: MW_kDa, pI, GRAVY, Instability_Index, Aliphatic_Index, Hydrophobic_pct, Charged_pct, and the 20 amino acid percentages.
  * Logistic regression: log_MW, pI, log_Length, GRAVY, Aromaticity, Instability_Index, Aliphatic_Index, Charged_pct, Hydrophobic_pct, Polar_pct, Acidic_pct, Basic_pct, the 20 amino acid percentages, pI_dist7, cubic spline terms on log_MW and pI, plus one-hot Cell_location and Gene_type (categories with fewer than 30 labeled proteins grouped as "Other").
  * The input form must collect the union of all of these.

## Notes on the reference notebooks
- logreg_notebook.ipynb: reuse its pipeline design (scaling, spline terms, one-hot categories, regularisation chosen by gene-grouped cross-validation).
- svm_notebook_mst.ipynb: reuse only the idea of median imputation, StandardScaler and an RBF kernel. Ignore the Google Colab cells and the PLM_Sol repository. Its 15-feature list was chosen using the test set, so do not reuse it; use all 27 features or select features inside each cross-validation fold. Its reported 78.58 percent accuracy is optimistic.
- rf_notebook.ipynb: reuse the settings only: 300 trees, class_weight="balanced", max_features="sqrt", random_state=42, threshold 0.5. Retrain from scratch.

## Work order
Step 1. Run the self-test in code/features.py. Check that for 20 proteins in features_all.csv, compute_features(Sequence) reproduces the feature columns within 0.0001.
Step 2. train_ensemble.py (a script that can also run as a notebook):
  a. Hold out 20 percent of GENES (group by B_number) as a final test set: GroupShuffleSplit, random_state=42. Do not touch it until step e.
  b. On the other 80 percent run 5-fold gene-grouped cross-validation. Choose all settings inside the folds only.
  c. Fit logistic regression, SVM (probabilities from CalibratedClassifierCV with method="sigmoid", or SVC(probability=True); never use raw decision values) and random forest.
  d. Collect out-of-fold probabilities. Train the combiner (a logistic regression on the three probabilities) on those only. Also compute the simple average. Use the combiner only if it beats the best single model on grouped cross-validation ROC-AUC; otherwise use the simple average.
  e. Refit the three models on the full 80 percent, evaluate each model and the ensemble ONCE on the held-out test set. Then refit on all labeled data for production.
  f. Save with joblib: lr.joblib, svm.joblib, rf.joblib, meta.joblib, plus feature_order.json, training_ranges.json (min and max per feature), categories.json, and model_card.json with the real accuracy, ROC-AUC, precision, recall and F1 of each model and of the ensemble (held-out test and cross-validation), dataset sizes and library versions.
Step 3. Backend (FastAPI) that loads only those artifacts, as in the API contract in the main prompt.
Step 4. Frontend as in the design rules in the main prompt. Every number on the "How this works" page comes from model_card.json.
Step 5. Run the acceptance tests.

## Acceptance tests
- For 20 training proteins, the backend's per-model probability equals the training script's probability to 4 decimals.
- Every probability and the final score are between 0.0 and 1.0.
- No held-out test gene was used to fit any model, imputer, scaler, calibrator or the combiner (assert on B_number).
- Batch mode runs on the 936 unlabeled proteins in features_all.csv and returns the three model scores, the final score and the agreement label for each.
- No hard-coded accuracy, AUC or other metric number appears anywhere outside model_card.json.
- The site states its limits: E. coli K-12 proteins only; related proteins can appear on both sides of a split; the unlabeled set contains many more membrane proteins (integral membrane proteins are about 46 percent of it versus about 10 percent of the labeled set), so scores there are less reliable.

## Do not
- Do not use any lab-measured column, or Solubility_pct as a feature (Soluble is defined from it).
- Do not select features, tune settings or choose the threshold using the test set.
- Do not claim any accuracy that is not in model_card.json.
