# Predicting Nuclide Decay Mode From Nuclear Structure

Project Question: Given a nuclide's proton and neutron counts (potential to include mass-derived energies), can a model predict how its ground state primarily decays?

## Data Information

Source: IAEA Nuclear Data Section, Live Chart of Nuclides (ground-state table)
URL: https://nds.iaea.org/relnsd/v1/data?fields=ground_states&nuclides=all
Downloaded on 10/12/2026 manually through a browser (the API rejects scripted requests); each file contains an extraction date column as well for traceability
Contains 3,386 nuclides with 55 columns; raw file included in git, so no download is needed to reproduce

## Setup

- Python 3.14.7, virtual environment in `.venv/`
- Install PyTorch separately from pytorch.org, choosing the build that matches your GPU (newer GPUs need newer CUDA builds; the CPU build also works at this dataset size), then run `python -m pip install -r requirements.txt`
- Run scripts from the project root, e.g. `python src/get_data.py`

## File Setup

```
MLNuclearPhysics/
├── data/raw/              # untouched download
├── data/processed/
├── notebooks/             # exploration
├── results/               # hyperparameter search results (CSV)
├── src/
│   ├── get_data.py        # load data, build labels
│   ├── features.py        # Tier 1 features, load_main_dataset()
│   ├── split.py           # random and element train/test splits
│   ├── baselines.py       # majority class, logistic regression, random forest
│   ├── error_analysis.py  # per-class errors of the default forest
│   ├── mlp.py             # PyTorch MLP and training loop
│   ├── mlp_cv.py          # MLP cross validation
│   ├── tune_mlp.py        # MLP random search
│   ├── tune_rf.py         # random forest random search
│   └── compare.py         # repeated-CV model comparison
├── requirements.txt
└── README.md
```

## Target and its labels

Target: the primary decay mode of the nuclide, 'decay_1' in the data; expected in 5 classes so results are concise yet still meaningful

1. beta_minus; B-; 1,344 appearances in data
2. beta_plus_ec; EC+B+, EC, B+; 1,082 appearances in data
3. alpha; A; 479 appearances in data
4. stable; STABLE; 244 appearances in data
5. other; P, N, SF, double decays, delayed emission, etc.; 193 appearances in data

Counts above are for all 3,342 labeled rows. The main dataset (measured half-life only, 3,191 rows) has: beta_minus 1,225 / beta_plus_ec 1,070 / alpha 473 / stable 244 / other 179.

Notes regarding specific target outcomes:
- For value 2. I grouped the 3 types of decay as one because they all represent the same process (proton to neutron).
- For value 4. I check if a nuclide is STABLE in the data first, since some rows do actually have a decay_1 value yet decay so slowly they are treated as STABLE
- For value 5. I group all rare types of decay since there are not enough rows to evaluate (some decays only had 1 row)
- 44 rows were dropped due to having no decay_1 value, yet they do have a half-life so they can't be grouped as stable

## Data notes and exclusions

- Columns: half-life, decay branching, and uncertainty have an immediate relationship with decay and are thus excluded. Furthermore, columns: abundance, radius, and nuclear moments are only filled for long-lasting nuclides, so even their missingness reveals the label, and they are also dropped. Columns with few entries are also dropped
- Some nuclide masses appear to be estimated from systematics rather than measured (interpreting `me_systematics == 'Y'` in the data this way, based on the column name and pattern, not documentation). None of the stable nuclides have estimated masses, so we eliminate this column along with the mass uncertainty columns as features. It is kept only as an evaluation flag. Potential to revisit in evaluation
- 151 labeled rows have a decay mode but no measured half-life. 119 of them also have estimated masses and 119 are beta-minus, which suggests these modes are assumed or predicted rather than directly observed (my inference from the pattern, not stated by the database). Their labels are less reliable, so they are excluded from training and testing. Comparing the model's predictions against their listed modes is left as possible future work

## Project Approach

1. Dataset: 3,191 rows with a measured half-life
2. Main (Tier 1) features: Z, N, A (mass number, Z + N), N-Z, neutron fraction, even/odd parity of Z and N, distance to nearest magic number
    - Ensured in features.py testing that beta_minus nuclides had higher neutron fraction, alpha emitters had largest mass number, and the majority of stable nuclides are even-even. These are all commonly accepted nuclear physics principles, checked to confirm our data relatively fits assumptions
3. Establishing Train/Test split: Set aside 20% of data as test (stratified by label), then train on the remaining 80% using 5 fold cross validation
    - Some isotopes are near identical (Sn-120 and Sn-121), so I experimented with two different kinds of splits. RANDOM split is a traditional split that is purely random in separating train and test data. ELEMENT split addresses the similar isotope issue by sending all isotopes that share the same number of protons entirely to train or entirely to test (24 elements held out). It tests generalization to unseen elements, though neighboring elements remain in training, so it does not test unseen regions of the chart
    - The test sets are touched once, at the end; all model selection uses cross validation on the training portion
4. Baselines: I use 3 baseline models (majority class, logistic regression, and random forest from sklearn, all with default settings) to compare against our final model created with PyTorch. Results of a single 5 fold cross validation on the training portion:

    | Model | Random split (acc / macro-F1) | Element split (acc / macro-F1) |
    |---|---|---|
    | Majority class | 0.384 / 0.111 | 0.383 / 0.111 |
    | Logistic regression | 0.803 / 0.594 | 0.810 / 0.619 |
    | Random forest | 0.864 / 0.744 | 0.841 / 0.717 |

    Linear boundaries already reach about 80% accuracy and the forest's edge is mainly in macro-F1 (the small classes).

    Per-class results for the default random forest (cross validated predictions on the training portion; precision / recall / F1):

    | Class | Random split | Element split |
    |---|---|---|
    | beta_minus | 0.913 / 0.951 / 0.932 | 0.900 / 0.956 / 0.927 |
    | beta_plus_ec | 0.875 / 0.932 / 0.903 | 0.859 / 0.903 / 0.880 |
    | alpha | 0.824 / 0.844 / 0.834 | 0.789 / 0.789 / 0.789 |
    | stable | 0.699 / 0.595 / 0.643 | 0.678 / 0.534 / 0.597 |
    | other | 0.652 / 0.301 / 0.411 | 0.568 / 0.319 / 0.409 |

    The macro-F1 gap comes almost entirely from stable and other. Stable nuclides are confused with their beta-minus and beta-plus/EC neighbors on the chart, and alpha and beta-plus/EC are confused with each other.

    Specific "other" decay_1 modes (element split with cross validated predictions on train; found / total): SF 21/48, P 13/47, 2P 9/11, N 2/8, double beta (2B-, 2B+, 2EC) 0/10, delayed emission (B-N, B-2N) 0/12, remaining modes 1/8. Spontaneous fission and proton emission are two thirds of the class. Hypothesis for the Tier 2 ablation: separation energies (sp, sn) should lift proton and neutron emission while having little effect on fission predictions.

5. Models and tuning:
    - **MLP (PyTorch):** 2 hidden layers of 64 (ReLU, dropout 0.1), Adam with weight decay, early stopping on macro-F1 using an inner validation split (15% of each fold's training rows), features standardized using training rows only.
    - **MLP search:** random search of 32 configurations (hidden size, layers, dropout, weight decay, learning rate, batch size, class-weight power) scored with 5-fold element-grouped CV (`results/mlp_tuning.csv`). The best configuration reached macro-F1 0.775 versus 0.758 to 0.765 for the defaults, within fold-to-fold noise (std 0.01 to 0.04), and most configurations landed between 0.74 and 0.77. Only a learning rate of 3e-4 was clearly worse, and extra depth did not help (a single hidden layer matched 2 to 3).
    - **Random forest search:** random search of 31 configurations (`results/rf_tuning.csv`). Class weighting was the main lever: unweighted configurations averaged roughly 0.71 macro-F1 versus about 0.75 for balanced weighting, and the best configuration (300 trees, max depth 20, min samples per leaf 3, sqrt features, balanced_subsample weights) reached 0.760 versus 0.722 for the default.
    - **Comparison:** each contender re-scored over 3 CV shuffles x 5 folds (15 folds per split) on the training portion only. These numbers supersede the single-shuffle ones above (the default forest differs slightly because of different fold shuffles).

    | Model | Random split macro-F1 | Element split macro-F1 |
    |---|---|---|
    | Random forest, default | 0.752 +/- 0.023 | 0.721 +/- 0.036 |
    | Random forest, tuned | 0.769 +/- 0.024 | 0.762 +/- 0.037 |
    | MLP, default, no class weights | 0.772 +/- 0.019 | 0.759 +/- 0.030 |
    | MLP, default, inverse-frequency class weights | 0.760 +/- 0.025 | 0.760 +/- 0.026 |
    | MLP, 3 best searched configurations | 0.775 to 0.782 | 0.764 to 0.772 |

    The tuned forest and the default MLP are statistically tied (the MLP wins 9/15 folds on the random split and 6/15 on the element split). Tuning mattered for the forest, mainly through class weighting, and not for the MLP: before looking at results I set a bar of a mean gain of at least about 0.01 on both splits over the default, and none of the three best configurations cleared it. Class weights raise recall on rare classes at a large cost in precision, so they are a different operating point and not a better one. With class weighting, the forest's drop from the random to the element split shrinks to about 0.007, so there is little evidence of neighbor memorization.

    Final settings, frozen before the Tier 2 ablation (chosen using Tier 1 features): MLP with the defaults above and no class weights; random forest with the tuned settings above.

6. Tier 2 ablation (planned): add mass-derived energies to measure how much they help, in layers because Q-values are close to tautological for this task: Tier 1; + binding energy and mass excess; + neutron and proton separation energies (sn, sp); + decay Q-values (qa, qec, qbm, qbm_n). Hyperparameters stay frozen so differences come from the features. All tiers will be compared on the same rows (those complete in every Tier 2 column), after checking whether missingness in these columns tracks the label. Metrics will also be reported on the measured-mass subset (`extrapolated == False`) and per class, to test the hypothesis above.
7. Final evaluation (planned): score the final models once on each test set (accuracy, macro-F1, per-class precision and recall), plus learning curves and permutation feature importance.

## Progress

- [x] Data downloaded and inspected
- [x] Labels defined, data issues addressed (data issues discovered using 'get_data.py')
- [x] Main features addressed ('src/features.py')
- [x] Train/test split, baselines
- [x] Error analysis ('src/error_analysis.py')
- [x] PyTorch MLP creation and tuning, random forest tuning, repeated-CV comparison
- [ ] Tier 2 features and ablation
- [ ] Final test-set evaluation, learning curves, feature importance
- [ ] Final write-up