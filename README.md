# TAC-Net

TAC-Net is a transition-aware adaptive-depth feedforward neural network for tabular classification. An auxiliary classifier is attached after each candidate hidden layer. A learned gate estimates whether later computation is likely to correct or harm the current prediction. The model exits only when the calibrated policy accepts the intermediate prediction. Otherwise, execution continues to the next layer.

This repository contains the implementation and frozen evidence used in the paper. Raw datasets, trained weights, and generated output files are excluded.

## Reproduce the reported tables

The exact manuscript tables come from frozen experiment summaries committed under `evidence/`. This command requires only Python 3.11 or newer. It does not train a model or read a dataset.

```powershell
python run.py verify
```

The command writes four CSV files under `outputs/paper_tables/`:

| File | Paper result |
|---|---|
| `table_main.csv` | Five-dataset predictive preservation and computation savings |
| `table_depth.csv` | Depth 6, 8, and 12 pilot |
| `table_ablation.csv` | Stability-gate ablation |
| `table_baselines.csv` | ML, plain ANN, full-depth ANN, and TAC-Net comparison |

The main frozen result is 13 preserved runs out of 15 under a 0.005 headline-score tolerance. Mean executed ANN computation fell by 36.4% to 55.0%, depending on the dataset. These are measured experiment outputs, not values written into source code.

## Run the experiments again

Install the recorded package versions:

```powershell
python -m pip install -r requirements.txt
```

Place the prepared partitions under `data/model_ready/`, or point `TACNET_DATA_ROOT` to an external directory. Then run:

```powershell
python run.py train --datasets credit diabetes unsw nf_unsw nf_toniot_sanitized --seeds 42 73 101 --depth 6 --width 128 --epochs 20 --patience 4
```

Retraining writes additive files under `outputs/phase3/`. It does not overwrite frozen evidence. Small numeric differences remain possible across CPUs, operating systems, BLAS builds, and PyTorch versions. The frozen prediction artifacts preserve the exact runs reported in the paper.

## Expected data layout

```text
data/model_ready/
  Credit_Default/
    credit_train.csv
    credit_calibration.csv
    credit_test.csv
  Diabetes/
    diabetes_train.csv
    diabetes_calibration.csv
    diabetes_test.csv
  UNSW_NB15/
    unsw_train.csv
    unsw_calibration.csv
    unsw_test_official.csv
  NF_UNSW/
    nf_unsw_train.csv
    nf_unsw_calibration.csv
    nf_unsw_test.csv
  NF_TONIOT_SANITIZED/
    nf_toniot_train.csv
    nf_toniot_calibration.csv
    nf_toniot_test.csv
```

Training, calibration, and test files must remain separate. Preprocessing is fitted on the training split. Gate fitting and threshold selection use the calibration split. Test labels are used only after the policy is frozen.

## Dataset sources

Datasets are not redistributed in this repository. Download them from their official sources and retain the source licences and citation requirements.

| Repository key | Dataset | Official source |
|---|---|---|
| `credit` | Default of Credit Card Clients | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) |
| `diabetes` | CDC Diabetes Health Indicators | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators) |
| `unsw` | UNSW-NB15 | [UNSW Research](https://research.unsw.edu.au/projects/unsw-nb15-dataset) |
| `nf_toniot_sanitized` | NF-ToN-IoT | [UQ Cyber Research Centre](https://www.cyber.uq.edu.au/node/824) |
| `nf_unsw` | NF-UNSW-NB15 | [UQ Cyber Research Centre](https://www.cyber.uq.edu.au/node/824) |

`nf_toniot_sanitized` refers to the leakage-controlled partition used in the paper. Target-derived fields must not enter model inputs.

## Six-file code map

| File | Responsibility |
|---|---|
| `run.py` | Verification and retraining entry point |
| `tacnet/datasets.py` | Dataset registry and training-fitted preprocessing |
| `tacnet/model.py` | Feedforward ANN, auxiliary exits, calibration, metrics, and baselines |
| `tacnet/gate.py` | Correction, harm, stability, and continuation policy |
| `tacnet/experiment.py` | Frozen-test protocol, dynamic execution, latency, and artifact writing |
| `tacnet/report.py` | Exact manuscript table generation from frozen evidence |

## Decision rule

At hidden layer `l`, TAC-Net estimates correction benefit, harm risk, and remaining normalised compute. It continues when expected correction exceeds expected harm plus the cost term. Exit also requires calibrated confidence, class credibility, prediction stability, and the calibration-selected tolerance constraints. No test label enters this decision.

## Evidence policy

Each committed NPZ file stores row-level full-depth and TAC-Net predictions from one frozen run. JSON records contain metrics, exit counts, latency measurements, configuration, and artifact hashes. CSV summaries are derived from those records. The `verify` command converts the frozen summaries into paper-ready tables without manual transcription.

## Repository scope

The repository does not include raw data, virtual environments, cached packages, trained model weights, paper drafts, or generated tables. This keeps the review surface small and prevents accidental dataset redistribution.
