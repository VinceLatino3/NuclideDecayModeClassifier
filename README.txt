# Predicting Nuclide Decay Mode From Nuclear Structure

Project Question: Given a nuclide's proton and nuetron counts (potential to include mass-derived energies), can a model predict how its ground state primarily decays.

## Data Information

Source: IAEA Nuclear Data Section, Live Chart of Nuclides (ground-state table)
URL: https://nds.iaea.org/relnsd/v1/data?fields=ground_states&nuclides=all
Downloaded on 10/12/2026 (API rejects scripted requests), each file contains an extraction date column as well for traceability
Contains 3,386 nuclides with 55 columns; raw file is not included in git so it must be downloaded to reproduce

## File Setup
MLNuclearPhysics/
├── data/raw/          # untouched download
├── data/processed/
├── notebooks/         # exploration
├── src/               # get_data.py, features.py, (later) dataset/model/train
├── requirements.txt
└── README.md

## Target and its labels

Target: the primary decay mode of the nuclide, 'decay_1' in the data; expected in 5 classes so results are concise yet still meaningful

1. beta_minus; B-; 1,344 appearances in data
2. beta_plus_ec; EC+B, EC, B+; 1,082 appearances in data
3. alpha; A; 479 appearances in data
4. stable; STABLE; 244 appearances in data;
5. other; P, N, SF, double decays, delayed emission, etc.; 193 appearances in data

Notes regarding specific target outcomes:
- For value 2. I grouped the 3 types of decay as one because they all represent the same process (proton to nuetron).
- For value 4. I check if a nuclide is STABLE in the data first, since some rows do actually have a decay_1 value yet decay so slowly they are treated as STABLE
- For value 5. I group all rare types of decay since there are not enough rows to evaluate (some decays only had 1 row)
- 44 rows were dropped due to having no decay_1 value, yet they do have a half-life so they can't be grouped as stable

## Data notes and exclusions

- Columns: half-life, decay branching, and uncertainty have an immediate relationship with decay and are thus excluded. Furthermore, columns: abundance, radis, and nuclear moments are only filled for long-lasting nuclides and are also dropped. Columns with few entries are also dropped
- Some nuclide masses are estimated rather than measured due to them being impossible to 'weigh' experimentally. Furthermore, none of the nuclides with estimated mass (me_systematics == 'Y' in data) are stable so we also eliminate this feature along with mass uncertainty. Potential to revisit in evaluation
- 151 rows have a decay mode but no half-life (they do decay but in extreme amounts of time). Due to the nature of these nuclides representing stable however scientifically proven as decaying, we treat them as less reliable and will also exclude them from training data

## Project Approach

- Dataset: 3,191 rows remaining after exclusions (see Data notes and exclusions)
- Main features: Z, N, A, N-Z, nuetron fraction, even/odd parity of Z and N, distance to nearest magic number 
- Exploratory features (ablation): (mass-derived energies) binding energy, mass excess, separation energies
- Evaluation: Include results of main features (accuracy, precision, recall, f1) and the same statistics for data including the mass derived features
- Models: Start with scikit-learn baselines logistic regression and random forest, then try PyTorch MLP with hyperparameter tuning, learning curves, and feature importance
- Attempting to predict unusual data: predict decay modes for the 151 nuclides removed from data due to having no half-life

## Progress 

- [X] Data downloaded and inspected
- [X] Labels defined, data issues addressed (data issues discovered using 'get_data.py')
- [] Main features adressed ('src/features.py')
- [] Train/test split, baselines
- [] PyTorch model, tuning, analysis