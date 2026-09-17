"""Dataset locations and leakage-safe tabular encoding."""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("TACNET_DATA_ROOT", ROOT / "data" / "model_ready"))


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    train_file: Path
    calibration_file: Path
    test_file: Path
    target: str
    categorical: tuple[str, ...]
    drop_columns: tuple[str, ...] = field(default_factory=tuple)


def _dataset(name, folder, prefix, target, categorical, test_suffix="test", drop_columns=()):
    base = DATA_ROOT / folder
    return DatasetConfig(
        name=name,
        train_file=base / f"{prefix}_train.csv",
        calibration_file=base / f"{prefix}_calibration.csv",
        test_file=base / f"{prefix}_{test_suffix}.csv",
        target=target,
        categorical=tuple(categorical),
        drop_columns=tuple(drop_columns),
    )


FLOW_CATEGORICAL = (
    "PROTOCOL", "L7_PROTO", "TCP_FLAGS", "CLIENT_TCP_FLAGS",
    "SERVER_TCP_FLAGS", "ICMP_TYPE", "ICMP_IPV4_TYPE",
    "DNS_QUERY_TYPE", "FTP_COMMAND_RET_CODE",
)

DATASETS = {
    "credit": _dataset("credit", "Credit_Default", "credit", "default payment next month", (
        "SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3",
        "PAY_4", "PAY_5", "PAY_6",
    )),
    "diabetes": _dataset("diabetes", "Diabetes", "diabetes", "Diabetes_binary", (
        "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
        "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
        "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
        "DiffWalk", "Sex", "Age", "Education", "Income",
    )),
    "unsw": _dataset("unsw", "UNSW_NB15", "unsw", "attack_cat",
                     ("proto", "service", "state"), "test_official", ("label",)),
    "nf_toniot": _dataset("nf_toniot", "NF_TONIOT", "nf_toniot", "Attack", FLOW_CATEGORICAL),
    "nf_toniot_sanitized": _dataset("nf_toniot_sanitized", "NF_TONIOT_SANITIZED", "nf_toniot", "Attack", FLOW_CATEGORICAL),
    "nf_unsw": _dataset("nf_unsw", "NF_UNSW", "nf_unsw", "Attack", FLOW_CATEGORICAL),
}


def load_partitions(config: DatasetConfig):
    missing = [path for path in (config.train_file, config.calibration_file, config.test_file) if not path.exists()]
    if missing:
        locations = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(f"Missing prepared files for {config.name}:\n{locations}")
    return tuple(pd.read_csv(path, low_memory=False) for path in (
        config.train_file, config.calibration_file, config.test_file,
    ))


def split_xy(frame: pd.DataFrame, config: DatasetConfig):
    x = frame.drop(columns=[config.target, *config.drop_columns]).copy()
    for column in config.categorical:
        x[column] = x[column].fillna("__MISSING__").astype(str)
    return x, frame[config.target].astype(str).to_numpy()


class TabularEncoder:
    def __init__(self, categorical: tuple[str, ...], leaf_components: int, seed: int):
        self.categorical = list(categorical)
        self.leaf_components = leaf_components
        self.seed = seed
        self.numeric: list[str] = []
        self.preprocessor = None
        self.leaf_encoder = None
        self.leaf_svd = None
        self.category_maps: dict[str, dict[str, int]] = {}

    def fit_raw(self, frame: pd.DataFrame):
        self.numeric = [column for column in frame if column not in self.categorical]
        self.preprocessor = ColumnTransformer([
            ("numeric", StandardScaler(), self.numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32), self.categorical),
        ], remainder="drop", verbose_feature_names_out=False)
        self.preprocessor.fit(frame)
        for column in self.categorical:
            values = frame[column].fillna("__MISSING__").astype(str)
            self.category_maps[column] = {value: index + 1 for index, value in enumerate(sorted(values.unique()))}
        return self

    def fit_leaves(self, leaves: np.ndarray):
        self.leaf_encoder = OneHotEncoder(handle_unknown="ignore", dtype=np.float32)
        encoded = self.leaf_encoder.fit_transform(leaves)
        components = min(self.leaf_components, max(2, encoded.shape[1] - 1))
        self.leaf_svd = TruncatedSVD(n_components=components, random_state=self.seed).fit(encoded)
        return self

    def raw_matrix(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.preprocessor.transform(frame), dtype=np.float32)

    def leaf_matrix(self, leaves: np.ndarray) -> np.ndarray:
        return np.asarray(self.leaf_svd.transform(self.leaf_encoder.transform(leaves)), dtype=np.float32)

    def transformer_arrays(self, frame: pd.DataFrame):
        numeric = np.zeros((len(frame), len(self.numeric)), dtype=np.float32)
        if self.numeric:
            numeric = self.preprocessor.named_transformers_["numeric"].transform(frame[self.numeric]).astype(np.float32)
        categorical = np.zeros((len(frame), len(self.categorical)), dtype=np.int64)
        for index, column in enumerate(self.categorical):
            categorical[:, index] = frame[column].fillna("__MISSING__").astype(str).map(
                self.category_maps[column]
            ).fillna(0).astype(np.int64)
        cardinalities = [len(self.category_maps[column]) + 1 for column in self.categorical]
        return numeric, categorical, cardinalities

    def save(self, path: Path):
        with path.open("wb") as handle:
            pickle.dump(self, handle)
