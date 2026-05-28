import argparse
import os
import pandas as pd

STATE_FEATURES = [
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
    "episode_start",
    "timestamp_diff_days",
]

META_COLUMNS = [
    "student_id",
    "date",
    "episode_id",
    "action",
    "action_name",
    "reward_sparse",
    "reward_dense",
    "reward_observed_only",
    "done",
]

OUTPUT_COLS = META_COLUMNS + STATE_FEATURES


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare RL transition dataset from aggregated StudentLife CSV."
    )

    parser.add_argument(
        "--input",
        type=str,
        default="daily_studentlife_no_transitions.csv",
        help="Input aggregated CSV produced by build_aggregated_data.py."
    )

    parser.add_argument(
        "--output",
        type=str,
        default="daily_studentlife.csv",
        help="Output transition-level CSV for RL training."
    )

    parser.add_argument(
        "--split",
        type=str,
        default="chronological",
        choices=["none", "student", "chronological"],
        help="Optional split strategy."
    )

    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.8,
        help="Train split fraction."
    )

    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="Validation split fraction."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for student split."
    )

    return parser.parse_args()


def build_transitions(df: pd.DataFrame):

    required_cols = [
        "student_id",
        "date",
        "episode_id",
        "action",
        "action_name",
        "done",
        "timestamp_diff_days",
    ] + STATE_FEATURES

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns in input CSV: {missing}")

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(
        ["student_id", "episode_id", "date"]
    ).reset_index(drop=True)

    # shift only within each episode
    shifted = df.groupby(
        ["student_id", "episode_id"]
    )[STATE_FEATURES + ["date"]].shift(-1)

    shifted.columns = [f"next_{col}" for col in shifted.columns]

    transitions = pd.concat([df, shifted], axis=1)

    # remove terminal rows with no next state
    transitions = transitions[
        transitions["next_date"].notna()
    ].copy()

    transitions["transition_id"] = range(len(transitions))

    transitions = transitions[
        ["transition_id"]
        + OUTPUT_COLS
        + ["next_date"]
        + [f"next_{col}" for col in STATE_FEATURES]
    ]

    return transitions


def split_by_student(
    transitions: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    seed: int,
):

    students = transitions["student_id"].unique()

    students = (
        pd.Series(students)
        .sample(frac=1.0, random_state=seed)
        .tolist()
    )

    n_train = int(len(students) * train_frac)
    n_val = int(len(students) * val_frac)

    train_students = set(students[:n_train])
    val_students = set(students[n_train:n_train + n_val])
    test_students = set(students[n_train + n_val:])

    train = transitions[
        transitions["student_id"].isin(train_students)
    ].reset_index(drop=True)

    val = transitions[
        transitions["student_id"].isin(val_students)
    ].reset_index(drop=True)

    test = transitions[
        transitions["student_id"].isin(test_students)
    ].reset_index(drop=True)

    return train, val, test


def split_chronologically(
    transitions: pd.DataFrame,
    train_frac: float,
    val_frac: float,
):

    transitions = transitions.copy()

    transitions["date"] = pd.to_datetime(transitions["date"])

    transitions = transitions.sort_values(
        ["date", "student_id", "episode_id"]
    ).reset_index(drop=True)

    n = len(transitions)

    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train = transitions.iloc[:train_end].reset_index(drop=True)
    val = transitions.iloc[train_end:val_end].reset_index(drop=True)
    test = transitions.iloc[val_end:].reset_index(drop=True)

    return train, val, test


def main():

    args = parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input)

    print(f"Loaded {len(df)} rows from {args.input}")

    transitions = build_transitions(df)

    print(f"Built {len(transitions)} transition rows")

    if args.split == "none":

        transitions.to_csv(args.output, index=False)

        print(f"Saved transitions to {args.output}")

        return

    if args.split == "student":

        train, val, test = split_by_student(
            transitions,
            args.train_frac,
            args.val_frac,
            args.seed,
        )

    elif args.split == "chronological":

        train, val, test = split_chronologically(
            transitions,
            args.train_frac,
            args.val_frac,
        )

    else:
        raise ValueError(f"Unknown split type: {args.split}")

    base, ext = os.path.splitext(args.output)

    train_path = f"{base}.train{ext}"
    val_path = f"{base}.val{ext}"
    test_path = f"{base}.test{ext}"

    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    test.to_csv(test_path, index=False)

    print(f"Saved {len(train)} train rows to {train_path}")
    print(f"Saved {len(val)} val rows to {val_path}")
    print(f"Saved {len(test)} test rows to {test_path}")


if __name__ == "__main__":
    main()