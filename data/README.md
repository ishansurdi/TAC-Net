# Local dataset directory

Keep downloaded or prepared data in this directory. Git ignores every data file here. Only this instruction file is tracked.

TAC-Net expects three fixed CSV partitions per dataset: training, calibration, and test. Follow the directory and file names listed in the repository README. The training split fits preprocessing and model parameters. The calibration split fits the gate and exit thresholds. The test split is opened after the policy is frozen.

You may keep data elsewhere. Set `TACNET_DATA_ROOT` to the directory containing the prepared dataset subfolders before running `python run.py benchmark`.
