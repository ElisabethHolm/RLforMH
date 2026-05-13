import os
import json
import pandas as pd
import numpy as np

# =========================================================
# CONFIG
# =========================================================

DATASET_ROOT = "studentLifeDataset"
OUTPUT_PATH = "daily_studentlife_no_transitions.csv"
HISTORY_DAYS = 3
ACTION_THRESHOLD = 0.6
REWARD_LONG_WINDOW = 7
REWARD_LONG_WEIGHT = 0.25

# =========================================================
# HELPERS
# =========================================================

def unix_to_date(ts):
    return pd.to_datetime(ts, unit="s").date()


def extract_uid(filename):
    base = os.path.basename(filename)
    if "_u" in base:
        return "u" + base.split("_u")[1].split(".")[0]
    return None


def normalize_per_student(df, cols):
    return df.groupby("student_id")[cols].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )


def build_action_labels(df):
    df["activity_z_delta"] = df.groupby("student_id")["activity_z"].diff()
    df["sleep_z_delta"] = df.groupby("student_id")["sleep_z"].diff()
    df["social_z_delta"] = df.groupby("student_id")["social_z"].diff()

    def choose_action(row):
        deltas = {
            "activity": row["activity_z_delta"],
            "sleep": row["sleep_z_delta"],
            "social": row["social_z_delta"],
        }
        best_feature = max(deltas, key=lambda k: abs(deltas[k]) if pd.notna(deltas[k]) else -1)
        best_delta = deltas[best_feature]

        if pd.isna(best_delta) or abs(best_delta) < ACTION_THRESHOLD:
            return 0

        if best_feature == "activity":
            return 1 if best_delta > 0 else 2
        if best_feature == "sleep":
            return 3 if best_delta > 0 else 4
        return 5 if best_delta > 0 else 6

    df["action"] = df.apply(choose_action, axis=1)
    df["action_name"] = df["action"].map({
        0: "none",
        1: "increase_activity",
        2: "decrease_activity",
        3: "increase_sleep",
        4: "decrease_sleep",
        5: "increase_social",
        6: "decrease_social",
    })
    return df


def add_history_features(df, cols, n):
    for lag in range(1, n + 1):
        for col in cols:
            df[f"{col}_lag{lag}"] = df.groupby("student_id")[col].shift(lag)
    df.fillna(0, inplace=True)
    return df


def compute_reward(df):
    df["mood_centered"] = df.groupby("student_id")["mood"].transform(
        lambda x: x - x.mean()
    )

    df["mood_trend"] = df.groupby("student_id")["mood_centered"].transform(
        lambda x: x.rolling(window=REWARD_LONG_WINDOW, min_periods=1).mean()
    )

    df["mood_short_term"] = df.groupby("student_id")["mood_centered"].diff()
    df["mood_long_term"] = df.groupby("student_id")["mood_trend"].diff()

    df["reward"] = df["mood_short_term"] + REWARD_LONG_WEIGHT * df["mood_long_term"]
    return df

# =========================================================
# MOOD
# =========================================================

def process_mood():
    mood_dir = os.path.join(DATASET_ROOT, "EMA/response/Mood")
    rows = []

    def score(entry):
        try:
            return float(entry.get("happy", 0)) - float(entry.get("sad", 0))
        except Exception:
            return None

    for file in os.listdir(mood_dir):
        if not file.endswith(".json"):
            continue

        uid = extract_uid(file)
        path = os.path.join(mood_dir, file)

        try:
            with open(path, "r") as f:
                data = json.load(f)

            for e in data:
                if "resp_time" not in e:
                    continue
                s = score(e)
                if s is None:
                    continue
                rows.append({
                    "student_id": uid,
                    "date": unix_to_date(e["resp_time"]),
                    "timestamp": e["resp_time"],
                    "mood": s,
                })
        except Exception as e:
            print(f"Mood error {file}: {e}")

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["student_id", "date", "timestamp", "mood"])

    return (
        df.groupby(["student_id", "date"])
        .agg({"timestamp": "mean", "mood": "mean"})
        .reset_index()
    )

# =========================================================
# SLEEP
# =========================================================

def process_sleep():
    sleep_dir = os.path.join(DATASET_ROOT, "EMA/response/Sleep")
    rows = []

    for file in os.listdir(sleep_dir):
        if not file.endswith(".json"):
            continue

        uid = extract_uid(file)
        path = os.path.join(sleep_dir, file)

        try:
            with open(path, "r") as f:
                data = json.load(f)

            for e in data:
                if "resp_time" not in e or "rate" not in e:
                    continue
                try:
                    rate = float(e["rate"])
                except Exception:
                    continue
                rows.append({
                    "student_id": uid,
                    "date": unix_to_date(e["resp_time"]),
                    "sleep": rate,
                })
        except Exception as e:
            print(f"Sleep error {file}: {e}")

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["student_id", "date", "sleep"])
    return df.groupby(["student_id", "date"]).agg({"sleep": "mean"}).reset_index()

# =========================================================
# ACTIVITY
# =========================================================

def process_activity():
    activity_dir = os.path.join(DATASET_ROOT, "sensing/activity")
    rows = []

    for file in os.listdir(activity_dir):
        if not file.endswith(".csv"):
            continue

        uid = extract_uid(file)
        path = os.path.join(activity_dir, file)

        try:
            df = pd.read_csv(path)
            df.columns = [c.strip() for c in df.columns]
            if "timestamp" not in df.columns or "activity inference" not in df.columns:
                continue

            df["date"] = pd.to_datetime(df["timestamp"], unit="s").dt.date
            daily = df.groupby("date").agg({"activity inference": "mean"}).reset_index()
            daily["student_id"] = uid
            daily.rename(columns={"activity inference": "activity"}, inplace=True)
            rows.append(daily)
        except Exception as e:
            print(f"Activity error {file}: {e}")

    if not rows:
        return pd.DataFrame(columns=["student_id", "date", "activity"])
    return pd.concat(rows, ignore_index=True)

# =========================================================
# SOCIAL
# =========================================================

def process_social():
    def load_simple(dir_path, col):
        rows = []
        if not os.path.isdir(dir_path):
            return pd.DataFrame(columns=["student_id", "date", col])

        for file in os.listdir(dir_path):
            if not file.endswith(".csv"):
                continue
            uid = extract_uid(file)
            path = os.path.join(dir_path, file)
            try:
                df = pd.read_csv(path)
                df.columns = [c.strip() for c in df.columns]
                if "timestamp" not in df.columns:
                    continue
                df["date"] = pd.to_datetime(df["timestamp"], unit="s").dt.date
                daily = df.groupby("date").size().reset_index(name=col)
                daily["student_id"] = uid
                rows.append(daily)
            except Exception as e:
                print(f"{col} error {file}: {e}")

        if not rows:
            return pd.DataFrame(columns=["student_id", "date", col])
        return pd.concat(rows, ignore_index=True)

    conv_rows = []
    conv_dir = os.path.join(DATASET_ROOT, "sensing/conversation")
    if os.path.isdir(conv_dir):
        for file in os.listdir(conv_dir):
            if not file.endswith(".csv"):
                continue
            uid = extract_uid(file)
            path = os.path.join(conv_dir, file)
            try:
                df = pd.read_csv(path)
                df.columns = [c.strip() for c in df.columns]
                if "start_timestamp" not in df.columns:
                    continue
                df["date"] = pd.to_datetime(df["start_timestamp"], unit="s").dt.date
                daily = df.groupby("date").size().reset_index(name="conversation_count")
                daily["student_id"] = uid
                conv_rows.append(daily)
            except Exception as e:
                print(f"Conversation error {file}: {e}")
    if conv_rows:
        conv = pd.concat(conv_rows, ignore_index=True)
    else:
        conv = pd.DataFrame(columns=["student_id", "date", "conversation_count"])

    call = load_simple(os.path.join(DATASET_ROOT, "call_log"), "call_count")
    sms = load_simple(os.path.join(DATASET_ROOT, "sms"), "sms_count")
    social = conv.merge(call, on=["student_id", "date"], how="outer")
    social = social.merge(sms, on=["student_id", "date"], how="outer")
    social = social.fillna(0)
    social["social"] = (
        social.get("conversation_count", 0)
        + social.get("call_count", 0)
        + social.get("sms_count", 0)
    )
    return social[["student_id", "date", "social"]]

# =========================================================
# BUILD DATASET
# =========================================================

print("Processing mood...")
mood_df = process_mood()

print("Processing sleep...")
sleep_df = process_sleep()

print("Processing activity...")
activity_df = process_activity()

print("Processing social...")
social_df = process_social()

print("Merging...")

df = mood_df.merge(sleep_df, on=["student_id", "date"], how="outer")
df = df.merge(activity_df, on=["student_id", "date"], how="outer")
df = df.merge(social_df, on=["student_id", "date"], how="outer")

df = df.sort_values(["student_id", "date"]).reset_index(drop=True)

df["mood_observed"] = df["mood"].notna().astype(int)

# fill sensor values so the state can be defined every day,
# but keep raw mood sparse to avoid inventing labels
sensor_features = ["sleep", "activity", "social"]
df[sensor_features] = df.groupby("student_id")[sensor_features].ffill()
df[sensor_features] = df.groupby("student_id")[sensor_features].bfill()
df[sensor_features] = df[sensor_features].fillna(0)

# timestamp fallback for days without mood EMA
if "timestamp" not in df.columns:
    df["timestamp"] = np.nan

midnights = pd.to_datetime(df["date"]).astype("int64") // 10**9
df["timestamp"] = df["timestamp"].fillna(midnights)

# episodic reset for large time gaps (> 2 days)
df["timestamp_diff"] = df.groupby("student_id")["timestamp"].diff()
df["timestamp_diff_days"] = df["timestamp_diff"] / 86400.0
df["episode_start"] = df["timestamp_diff_days"].isna() | (df["timestamp_diff_days"] > 2)
df["episode_start"] = df["episode_start"].astype(int)
df["episode_id"] = df.groupby("student_id")["episode_start"].cumsum()
df["next_episode_start"] = df.groupby("student_id")["episode_start"].shift(-1).fillna(True).astype(int)
df["done"] = df["next_episode_start"]
df["timestamp_diff_days"] = df["timestamp_diff_days"].fillna(0)

gaps = int((df["timestamp_diff_days"] > 2).sum())
print(f"Large gaps (>2 days) between steps: {gaps}")

# use heavy-tail stabilization for social before normalization
if "social" in df.columns:
    social_upper = df["social"].quantile(0.99)
    df["social"] = df["social"].clip(lower=0, upper=social_upper)
    df["social"] = np.log1p(df["social"])

# normalized sensor features relative to each student baseline
z_cols = ["sleep_z", "activity_z", "social_z"]
df[z_cols] = normalize_per_student(df, sensor_features)

# action labels from meaningful baseline shifts
print("Building action labels...")
df = build_action_labels(df)
# reset invalid cross-episode deltas to none
df.loc[df["episode_start"] == 1, ["activity_z_delta", "sleep_z_delta", "social_z_delta", "action"]] = 0
df.loc[df["episode_start"] == 1, "action_name"] = "none"

# use mood observations for reward, with long-term smoothing
print("Computing reward...")
df = compute_reward(df)
# clear reward for new episodes to prevent invalid cross-episode transitions
df.loc[df["episode_start"] == 1, ["mood_short_term", "mood_long_term", "reward"]] = 0

df["mood_centered"] = df["mood_centered"]

# state history features approximate POMDP memory
print(f"Adding history features (last {HISTORY_DAYS} days)...")
state_cols = ["mood", "sleep_z", "activity_z", "social_z"]
df = add_history_features(df, state_cols, HISTORY_DAYS)

# drop invalid transitions where mood improvement cannot be computed
print("Dropping rows with invalid reward...")
df = df.dropna(subset=["reward"]).reset_index(drop=True)

# convert action to categorical labels for easier downstream training
df["action"] = df["action"].astype(int)

print("Saving output...")
df.to_csv(OUTPUT_PATH, index=False)

print("\nDONE")
print(df.head())
print("\nShape:", df.shape)
print("\nMissing:")
print(df.isna().sum())