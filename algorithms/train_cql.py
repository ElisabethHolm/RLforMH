from pathlib import Path

import pandas as pd
from d3rlpy.algos import DiscreteCQLConfig
from d3rlpy.dataset import MDPDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "daily_studentlife.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "studentlife_discrete_cql.d3"

STATE_COLS = [
    "mood",
    "sleep_z",
    "activity_z",
    "social_z",
    "mood_lag1",
    "sleep_z_lag1",
    "activity_z_lag1",
    "social_z_lag1",
    "mood_lag2",
    "sleep_z_lag2",
    "activity_z_lag2",
    "social_z_lag2",
    "mood_lag3",
    "sleep_z_lag3",
    "activity_z_lag3",
    "social_z_lag3",
    "mood_observed",
]


def load_dataset(path: Path = DATA_PATH) -> MDPDataset:
    df = pd.read_csv(path)

    return MDPDataset(
        observations=df[STATE_COLS].to_numpy("float32"), #conversions
        actions=df["action"].to_numpy("int64"),
        rewards=df["reward"].to_numpy("float32"),
        terminals=df["done"].to_numpy("bool"),
    )

#Clean up and load dataset
def main():
    dataset = load_dataset()
    cql = DiscreteCQLConfig().create(device="cpu")
    cql.fit(dataset, n_steps=10_000)
    cql.save_model(str(MODEL_PATH))
    print(f"Saved fitted model to {MODEL_PATH}")

if __name__ == "__main__":
    main()
