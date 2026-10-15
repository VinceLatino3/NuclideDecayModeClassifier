# Predicting Nuclide Decay Mode From Nuclear Structure

Project Question: Given a nuclide's proton and neutron counts (potential to include mass-derived energies), can a model predict how its ground state primarily decays?

## Data Information

Source: IAEA Nuclear Data Section, Live Chart of Nuclides (ground-state table)
URL: https://nds.iaea.org/relnsd/v1/data?fields=ground_states&nuclides=all
Downloaded on 10/12/2026 manually through a browser (the API rejects scripted requests); each file contains an extraction date column as well for traceability
Contains 3,386 nuclides with 55 columns; raw file included in git

## Setup

- Python 3.14.7, virtual environment in `.venv/`
- Install PyTorch separately from pytorch.org
- Run scripts from the project root, e.g. `python src/get_data.py`

## File Setup

```
MLNuclearPhysics/
├── data/raw/          # untouched download
├── data/processed/
├── notebooks/         # exploration
├── src/               # get_data.py, features.py, split.py, baselines.py,
│                      # error_analysis.py...
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
- 151 labeled rows have a decay mode but no measured half-life. 119 of them also have estimated masses and 119 are beta-minus, which suggests these modes are assumed or predicted rather than directly observed (my inference from the pattern, not stated by the database). Their labels are less reliable, so they are excluded from training and testing and kept for a later comparison against the model's predictions

## Project Approach

1. Dataset: 3,191 rows with a measured half-life
2. Main (Tier 1) features: Z, N, A (mass number, Z + N), N-Z, neutron fraction, even/odd parity of Z and N, distance to nearest magic number
    - Ensured in features.py testing that beta_minus nuclides had higher neutron fraction, alpha emitters had largest mass number, and the majority of stable nuclides are even-even. These are all commonly accepted nuclear physics principles, checked to confirm our data relatively fits assumptions
3. Establishing Train/Test split: Set aside 20% of data as test (stratified by label), then train on the remaining 80% using 5 fold cross validation
    - Some isotopes are near identical (Sn-120 and Sn-121), so I experimented with two different kinds of splits. RANDOM split is a traditional split that is purely random in separating train and test data. ELEMENT split addresses the similar isotope issue by sending all isotopes that share the same number of protons entirely to train or entirely to test (24 elements held out). It tests generalization to unseen elements, though neighboring elements remain in training, so it does not test unseen regions of the chart
    - The test sets are touched once, at the end; all model selection uses cross validation on the training portion
4. Baselines: I use 3 baseline models (majority class, logistic regression, and random forest from sklearn) to compare against our final model created with PyTorch. Results of 5 fold cross validation on the training portion:
    | Model | Random split (acc / macro-F1) | Element split (acc / macro-F1) |
    |---|---|---|
    | Majority class | 0.384 / 0.111 | 0.383 / 0.111 |
    | Logistic regression | 0.803 / 0.594 | 0.810 / 0.619 |
    | Random forest | 0.864 / 0.744 | 0.841 / 0.717 |

    Linear boundaries already reach about 80% accuracy and the forest's edge is mainly in macro-F1 (the small classes). The forest loses a few points when whole elements are held out while the linear model does not, which suggests a small neighbor-memorization effect, though the difference is close to fold-to-fold noise. Given these results our PyTorch model has the bar set at a macro-F1 of about 0.72 to 0.74 (random forest) depending on the split, with 0.59 to 0.62 as the linear floor

    We compare against the random forest model with cross validated predictions on the training portion.
    | Class | Random split | Element split |
    |---|---|---|
    | beta_minus | 0.913 / 0.951 / 0.932 | 0.900 / 0.956 / 0.927 |
    | beta_plus_ec | 0.875 / 0.932 / 0.903 | 0.859 / 0.903 / 0.880 |
    | alpha | 0.824 / 0.844 / 0.834 | 0.789 / 0.789 / 0.789 |
    | stable | 0.699 / 0.595 / 0.643 | 0.678 / 0.534 / 0.597 |
    | other | 0.652 / 0.301 / 0.411 | 0.568 / 0.319 / 0.409 |
    Specific "other" decay_1 modes (element split with cross validated predictions on train; found / total): SF 21/48, P 13/47, 2P 9/11, N 2/8, double beta (2B-, 2B+, 2EC) 0/10, delayed emission (B-N, B-2N) 0/12, remaining modes 1/8. Spontaneous fission and proton emission are two thirds of the class. For Tier 2 ablation: expect adding separation energies (sp, sn) should lift proton and neutron emission while having little effect on fission predictions.

6. Evaluation and Comparison: Include results of main features (accuracy, macro-F1, per-class precision and recall) on both splits, and the same statistics for the Tier 2 ablation. Tier 2 adds mass-derived energies (binding energy, mass excess, separation energies, possibly Q-values) to measure how much they help; Q-values are close to tautological for this task, so that comparison is part of the analysis. Metrics will also be reported on the measured-mass subset (`extrapolated == False`)
    MLP (PyTorch, 2 hidden layers of 64, early stopping on inner validation split; untuned; same 5 folds):

    | Model | Random split (acc / macro-F1) | Element split (acc / macro-F1) |
    |---|---|---|
    | Random forest | 0.864 / 0.744 | 0.841 / 0.717 |
    | MLP, class weights | 0.827 / 0.749 | 0.834 / 0.765 |
    | MLP, no class weights | 0.870 / 0.774 | 0.857 / 0.758 |

    The MLP appears to match/modestly exceed the forest baseline, with gains concentrated in the 'stable' and 'other' classes. Inverse-frequency class weights raise recall on rare classes at a large cost in precision and do not meaningully improve macro-F1

## Progress

- [x] Data downloaded and inspected
- [x] Labels defined, data issues addressed (data issues discovered using 'get_data.py')
- [x] Main features addressed ('src/features.py')
- [x] Train/test split, baselines
- [X] Error analysis ('src/error_analysis.py')
- [ ] PyTorch model, tuning, analysis