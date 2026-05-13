import os
import json
import pandas as pd
import numpy as np

# =========================================================
# CONFIG
# =========================================================

DATASET_ROOT = "studentLifeDataset"
OUTPUT_PATH = "daily_studentlife.csv"

# =========================================================
# HELPERS
# =========================================================

def unix_to_date(ts):
    return pd.to_datetime(ts, unit="s").date()


def extract_uid(filename):
    """
    Example:
        Mood_u01.json -> u01
        activity_u14.csv -> u14
    """
    base = os.path.basename(filename)

    if "_u" in base:
        return "u" + base.split("_u")[1].split(".")[0]

    return None


# =========================================================
# MOOD PROCESSING (OPTION 1: ONLY OBSERVED SIGNALS)
# =========================================================

def process_mood():
    mood_dir = os.path.join(
        DATASET_ROOT,
        "EMA",
        "response",
        "Mood"
    )

    rows = []

    def extract_mood_score(entry):
        """
        Compute mood only from valid observed EMA response.
        No filling, no smoothing.
        """

        try:
            happy = float(entry.get("happy", 0))
            sad = float(entry.get("sad", 0))
            return happy - sad
        except:
            return None

    for file in os.listdir(mood_dir):

        if not file.endswith(".json"):
            continue

        uid = extract_uid(file)
        path = os.path.join(mood_dir, file)

        try:
            with open(path, "r") as f:
                data = json.load(f)

            for entry in data:

                if "resp_time" not in entry:
                    continue

                mood_score = extract_mood_score(entry)

                # -------------------------------------------------
                # ONLY KEEP REAL OBSERVATIONS
                # -------------------------------------------------
                if mood_score is None:
                    continue

                rows.append({
                    "student_id": uid,
                    "timestamp": entry["resp_time"],
                    "date": unix_to_date(entry["resp_time"]),
                    "mood": mood_score
                })

        except Exception as e:
            print(f"Error processing mood file {file}: {e}")

    df = pd.DataFrame(rows)

    # ---------------------------------------------------------
    # DO NOT forward fill or mean fill across time
    # only aggregate repeated same-day EMA responses
    # ---------------------------------------------------------
    df = (
        df.groupby(["student_id", "date"])
        .agg({
            "mood": "mean",
            "timestamp": "mean"
        })
        .reset_index()
    )

    return df


# =========================================================
# SLEEP PROCESSING
# =========================================================

def process_sleep():
    sleep_dir = os.path.join(
        DATASET_ROOT,
        "EMA",
        "response",
        "Sleep"
    )

    rows = []

    def extract_rate(entry):
        """
        Extract ONLY sleep rate signal.
        This is the chosen sleep proxy.
        """

        if "rate" in entry:
            try:
                return float(entry["rate"])
            except:
                return None

        # ignore everything else (gps, null, social, etc.)
        return None

    for file in os.listdir(sleep_dir):

        if not file.endswith(".json"):
            continue

        uid = extract_uid(file)
        path = os.path.join(sleep_dir, file)

        try:
            with open(path, "r") as f:
                data = json.load(f)

            for entry in data:

                if "resp_time" not in entry:
                    continue

                sleep_score = extract_rate(entry)

                # skip invalid / noisy entries
                if sleep_score is None:
                    continue

                rows.append({
                    "student_id": uid,
                    "date": unix_to_date(entry["resp_time"]),
                    "sleep": sleep_score
                })

        except Exception as e:
            print(f"Error processing sleep file {file}: {e}")

    df = pd.DataFrame(rows)

    # -----------------------------------------------------
    # Daily aggregation (IMPORTANT: mean over day)
    # -----------------------------------------------------
    df = (
        df.groupby(["student_id", "date"])
        .agg({"sleep": "mean"})
        .reset_index()
    )

    return df


# =========================================================
# ACTIVITY PROCESSING
# =========================================================

def process_activity():

    activity_dir = os.path.join(
        DATASET_ROOT,
        "sensing",
        "activity"
    )

    rows = []

    for file in os.listdir(activity_dir):

        if not file.endswith(".csv"):
            continue

        uid = extract_uid(file)

        path = os.path.join(activity_dir, file)

        try:
            df = pd.read_csv(path)

            # clean weird column spacing
            df.columns = [c.strip() for c in df.columns]

            df["date"] = pd.to_datetime(
                df["timestamp"],
                unit="s"
            ).dt.date

            daily = (
                df.groupby("date")
                .agg({
                    "activity inference": "mean"
                })
                .reset_index()
            )

            daily["student_id"] = uid

            daily.rename(
                columns={
                    "activity inference": "activity"
                },
                inplace=True
            )

            rows.append(daily)

        except Exception as e:
            print(f"Error processing activity file {file}: {e}")

    return pd.concat(rows, ignore_index=True)


# =========================================================
# SOCIAL PROCESSING
# =========================================================

def process_social():

    conversation_dir = os.path.join(
        DATASET_ROOT,
        "sensing",
        "conversation"
    )

    call_dir = os.path.join(
        DATASET_ROOT,
        "call_log"
    )

    sms_dir = os.path.join(
        DATASET_ROOT,
        "sms"
    )

    social_rows = []

    # -----------------------------------------------------
    # CONVERSATIONS
    # -----------------------------------------------------

    for file in os.listdir(conversation_dir):

        if not file.endswith(".csv"):
            continue

        uid = extract_uid(file)

        path = os.path.join(conversation_dir, file)

        try:
            df = pd.read_csv(path)

            df.columns = [c.strip() for c in df.columns]

            df["date"] = pd.to_datetime(
                df["start_timestamp"],
                unit="s"
            ).dt.date

            daily = (
                df.groupby("date")
                .size()
                .reset_index(name="conversation_count")
            )

            daily["student_id"] = uid

            social_rows.append(daily)

        except Exception as e:
            print(f"Conversation error {file}: {e}")

    conversation_df = pd.concat(social_rows, ignore_index=True)

    # -----------------------------------------------------
    # CALLS
    # -----------------------------------------------------

    call_rows = []

    for file in os.listdir(call_dir):

        if not file.endswith(".csv"):
            continue

        uid = extract_uid(file)

        path = os.path.join(call_dir, file)

        try:
            df = pd.read_csv(path)

            df["date"] = pd.to_datetime(
                df["timestamp"],
                unit="s"
            ).dt.date

            daily = (
                df.groupby("date")
                .size()
                .reset_index(name="call_count")
            )

            daily["student_id"] = uid

            call_rows.append(daily)

        except Exception as e:
            print(f"Call error {file}: {e}")

    call_df = pd.concat(call_rows, ignore_index=True)

    # -----------------------------------------------------
    # SMS
    # -----------------------------------------------------

    sms_rows = []

    for file in os.listdir(sms_dir):

        if not file.endswith(".csv"):
            continue

        uid = extract_uid(file)

        path = os.path.join(sms_dir, file)

        try:
            df = pd.read_csv(path)

            df["date"] = pd.to_datetime(
                df["timestamp"],
                unit="s"
            ).dt.date

            daily = (
                df.groupby("date")
                .size()
                .reset_index(name="sms_count")
            )

            daily["student_id"] = uid

            sms_rows.append(daily)

        except Exception as e:
            print(f"SMS error {file}: {e}")

    sms_df = pd.concat(sms_rows, ignore_index=True)

    # -----------------------------------------------------
    # MERGE SOCIAL FEATURES
    # -----------------------------------------------------

    social_df = conversation_df.merge(
        call_df,
        on=["student_id", "date"],
        how="outer"
    )

    social_df = social_df.merge(
        sms_df,
        on=["student_id", "date"],
        how="outer"
    )

    social_df = social_df.fillna(0)

    social_df["social"] = (
        social_df["conversation_count"] +
        social_df["call_count"] +
        social_df["sms_count"]
    )

    social_df = social_df[
        ["student_id", "date", "social"]
    ]

    return social_df


# =========================================================
# MAIN
# =========================================================

print("Processing mood...")
mood_df = process_mood()

print("Processing sleep...")
sleep_df = process_sleep()

print("Processing activity...")
activity_df = process_activity()

print("Processing social...")
social_df = process_social()

print("Merging all modalities...")

# Start from mood as base
final_df = mood_df.merge(
    sleep_df,
    on=["student_id", "date"],
    how="outer"
)

final_df = final_df.merge(
    activity_df,
    on=["student_id", "date"],
    how="outer"
)

final_df = final_df.merge(
    social_df,
    on=["student_id", "date"],
    how="outer"
)

# =========================================================
# SORT
# =========================================================

final_df = final_df.sort_values(
    ["student_id", "date"]
)

# =========================================================
# MISSING VALUES (less filling)
# =========================================================

features = ["mood", "sleep", "activity", "social"]

# forward fill ONLY within student (no cross contamination)
final_df[features] = (
    final_df.groupby("student_id")[features]
    .transform(lambda x: x.ffill())
)

# fill remaining NaNs with student mean
for col in features:
    final_df[col] = (
        final_df.groupby("student_id")[col]
        .transform(lambda x: x.fillna(x.mean()))
    )

# final fallback (prevents RL crash)
final_df[features] = final_df[features].fillna(0)

# # =========================================================
# # MISSING VALUES
# # =========================================================

# # ---------------------------------------------------------
# # Behavioral features:
# # forward fill is reasonable because behavior is continuous
# # over time
# # ---------------------------------------------------------

# behavior_cols = [
#     "sleep",
#     "activity",
#     "social"
# ]

# final_df[behavior_cols] = (
#     final_df.groupby("student_id")[behavior_cols]
#     .transform(lambda x: x.ffill())
# )

# # ---------------------------------------------------------
# # Fill remaining behavioral NaNs with student mean
# # ---------------------------------------------------------

# for col in behavior_cols:

#     final_df[col] = (
#         final_df.groupby("student_id")[col]
#         .transform(
#             lambda x: x.fillna(x.mean())
#         )
#     )

# # ---------------------------------------------------------
# # Mood should NOT be forward filled
# # because it creates fake emotional persistence
# # ---------------------------------------------------------

# final_df["mood"] = (
#     final_df.groupby("student_id")["mood"]
#     .transform(
#         lambda x: x.fillna(x.mean())
#     )
# )

# # ---------------------------------------------------------
# # Final fallback
# # ---------------------------------------------------------

# all_feature_cols = [
#     "mood",
#     "sleep",
#     "activity",
#     "social"
# ]

# final_df[all_feature_cols] = (
#     final_df[all_feature_cols]
#     .fillna(0)
# )

# =========================================================
# FEATURE NORMALIZATION (PER-STUDENT)
# =========================================================
for col in ["mood", "sleep", "activity", "social"]:
    final_df[col] = (
        final_df.groupby("student_id")[col]
        .transform(lambda x: (x - x.mean()) / (x.std() + 1e-8))
    )

# =========================================================
# SOCIAL FEATURE STABILIZATION (heavy tail end skew)
# =========================================================

# clip extreme outliers first (prevents log issues)
upper = final_df["social"].quantile(0.99)
final_df["social"] = final_df["social"].clip(lower=0, upper=upper)

# safe log transform
final_df["social"] = np.log1p(final_df["social"])

# fill any resulting NaNs (VERY IMPORTANT)
final_df["social"] = final_df["social"].fillna(final_df["social"].mean())

# =========================================================
# SANITY CHECKS
# =========================================================

print(final_df.describe())
for col in ["mood", "sleep", "activity", "social"]:
    print("\n", col)
    print(final_df[col].value_counts().head(20))

# =========================================================
# REWARD DEFINITION (STABLE + NON-COLLAPSING)
# =========================================================

final_df = final_df.sort_values(["student_id", "date"])

# centered mood per student (prevents flat-line users)
final_df["mood_centered"] = final_df["mood"] - final_df.groupby("student_id")["mood"].transform("mean")

# reward = change in centered mood
final_df["reward"] = final_df.groupby("student_id")["mood_centered"].diff()

# drop invalid transitions only
final_df = final_df.dropna(subset=["reward"])

# =========================================================
# SAVE
# =========================================================

final_df.to_csv(OUTPUT_PATH, index=False)

print("\nDONE")
print(final_df.head())

print(f"\nSaved to: {OUTPUT_PATH}")

print("Final shape:", final_df.shape)

print("\nMissing values:")
print(final_df.isna().sum())

print("\nPer-student counts:")
print(final_df["student_id"].value_counts().head(10))