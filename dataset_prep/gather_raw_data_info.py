'''
DIRECTORY STRUCTURE
================================================================================
studentLifeDataset/
    .DS_Store
    EMA/
        EMA_definition.json
        response/
            Behavior/
                Behavior_u09.json
                Behavior_u25.json
                ...
                Behavior_u10.json
                Behavior_u51.json
                Behavior_u30.json
            Dimensions protestors/
                Dimensions protestors_u30.json
                Dimensions protestors_u51.json
                ..
                Dimensions protestors_u33.json
                Dimensions protestors_u25.json
            Cancelled Classes/
                Cancelled Classes_u30.json
                Cancelled Classes_u51.json
               ...
                Cancelled Classes_u33.json
                Cancelled Classes_u25.json
            Green Key 2/
                Green Key 2_u17.json
                Green Key 2_u56.json
                ...
                Green Key 2_u18.json
                Green Key 2_u59.json
            Sleep/
                Sleep_u01.json
                Sleep_u56.json
                ...
                Sleep_u59.json
                Sleep_u18.json
            Boston Bombing/
                Boston Bombing_u36.json
                Boston Bombing_u20.json
                ...
                Boston Bombing_u35.json
                Boston Bombing_u23.json
            Activity/
                Activity_u56.json
                Activity_u01.json
               ...
                Activity_u14.json
                Activity_u43.json
            Comment/
                Comment_u30.json
                Comment_u51.json
                ...
                Comment_u09.json
            Class 2/
                Class 2_u45.json
                ...
                Class 2_u50.json
            Lab/
                Lab_u57.json
                ...
                Lab_u42.json
                Lab_u39.json
            Mood 1/
                Mood 1_u45.json
                Mood 1_u12.json
               ...
                Mood 1_u50.json
            PAM/
                PAM_u24.json
                PAM_u32.json
               ...
                PAM_u27.json
                PAM_u31.json
            Social/
                Social_u27.json
                Social_u31.json
              ...
                Social_u24.json
                Social_u32.json
            Administration response/
                Administration response_u08.json
                Administration response_u49.json
               ...
                Administration response_u31.json
                Administration response_u27.json
            Exercise/
                Exercise_u00.json
                Exercise_u57.json
                ...
                Exercise_u42.json
                Exercise_u15.json
            Class/
                Class_u30.json
                Class_u51.json
              ...
                Class_u25.json
                Class_u09.json
            Green Key 1/
                Green Key 1_u12.json
                Green Key 1_u45.json
                ...
                Green Key 1_u07.json
            Mood 2/
                Mood 2_u17.json
                Mood 2_u01.json
                ...
                Mood 2_u22.json
                Mood 2_u34.json
            Events/
                Events_u46.json
                Events_u50.json
                ...
                Events_u53.json
                Events_u04.json
            Dimensions/
                Dimensions_u36.json
                Dimensions_u20.json
               ...
                Dimensions_u58.json
                Dimensions_u19.json
            Mood/
                Mood_u56.json
                Mood_u01.json
               ...
                Mood_u14.json
                Mood_u43.json
            QR_Code/
                QR_u46.json
                QR_u50.json
                ...
                QR_u53.json
                QR_u04.json
            Dining Halls/
                Dining Halls_u32.json
                Dining Halls_u24.json
              ...
                Dining Halls_u27.json
            Study Spaces/
                Study Spaces_u56.json
                Study Spaces_u01.json
              ...
                Study Spaces_u22.json
            Dartmouth now/
                Dartmouth now_u46.json
                Dartmouth now_u50.json
                ...
                Dartmouth now_u04.json
            Stress/
                Stress_u44.json
                Stress_u13.json
               ...
                Stress_u51.json
    dinning/
        u14.txt
        u01.txt
   ...
        u25.txt
    calendar/
        calendar_u50.csv
      ...
        calendar_u49.csv
    sms/
        sms_u58.csv
    ...
        sms_u41.csv
    education/
        class_info.json
        class.csv
        piazza.csv
        grades.csv
        deadlines.csv
    call_log/
        call_log_u57.csv
        call_log_u43.csv
...
        call_log_u58.csv
        call_log_u59.csv
    app_usage/
        running_app_u44.csv
        running_app_u50.csv
       ...
        running_app_u58.csv
        running_app_u49.csv
    sensing/
        .DS_Store
        wifi/
            wifi_u03.csv
            wifi_u17.csv
           ...
            wifi_u33.csv
            wifi_u32.csv
        gps/
            gps_u45.csv
            gps_u51.csv
            ...
            gps_u49.csv
        activity/
            activity_u44.csv
          ...
            activity_u49.csv
        phonelock/
            phonelock_u09.csv
           ...
            phonelock_u10.csv
        wifi_location/
            wifi_location_u59.csv
           ...
            wifi_location_u57.csv
        audio/
            audio_u59.csv
      ...
            audio_u54.csv
        bluetooth/
            bt_u45.csv
            ...
            bt_u49.csv
        dark/
            dark_u53.csv
      ...
            dark_u49.csv
        phonecharge/
            phonecharge_u58.csv
           ...
            phonecharge_u41.csv
        conversation/
            conversation_u10.csv
            ...
            conversation_u09.csv
    survey/
        PerceivedStressScale.csv
        panas.csv
        BigFive.csv
        psqi.csv
        FlourishingScale.csv
        LonelinessScale.csv
        vr_12.csv
        PHQ-9.csv
'''


import os
import json
import pandas as pd

# =========================================================
# CONFIG
# =========================================================

DATASET_ROOT = "studentLifeDataset"

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def inspect_csv(path, nrows=5):
    print(f"\nFILE: {path}")

    try:
        df = pd.read_csv(path)

        print(f"Shape: {df.shape}")

        print("\nColumns:")
        print(df.columns.tolist())

        print("\nDtypes:")
        print(df.dtypes)

        print("\nMissing Values:")
        print(df.isnull().sum())

        print("\nFirst Rows:")
        print(df.head(nrows))

    except Exception as e:
        print(f"ERROR: {e}")


def inspect_json(path, nrows=3):
    print(f"\nFILE: {path}")

    try:
        with open(path, "r") as f:
            data = json.load(f)

        print(f"Top-level type: {type(data)}")

        if isinstance(data, list):
            print(f"List length: {len(data)}")

            if len(data) > 0:
                print("\nFirst item:")
                print(json.dumps(data[0], indent=2)[:2000])

        elif isinstance(data, dict):
            print(f"Keys: {list(data.keys())[:20]}")

            print("\nPreview:")
            print(json.dumps(data, indent=2)[:2000])

    except Exception as e:
        print(f"ERROR: {e}")

# =========================================================
# DIRECTORY OVERVIEW (HIGH LEVEL ONLY)
# =========================================================

print_section("TOP-LEVEL DIRECTORY STRUCTURE")

for item in sorted(os.listdir(DATASET_ROOT)):
    path = os.path.join(DATASET_ROOT, item)

    if os.path.isdir(path):
        subitems = os.listdir(path)

        print(f"\n{item}/")
        print(f"  Contains {len(subitems)} items")

        # Print only first few
        preview = sorted(subitems)[:10]

        for p in preview:
            print(f"    {p}")

        if len(subitems) > 10:
            print("    ...")

    else:
        print(item)

# =========================================================
# TARGET FILES TO INSPECT
# =========================================================

print_section("INSPECT IMPORTANT DATA SOURCES")

# ---------------------------------------------------------
# MOOD EMA
# ---------------------------------------------------------

mood_json = os.path.join(
    DATASET_ROOT,
    "EMA",
    "response",
    "Mood",
    "Mood_u01.json"
)

inspect_json(mood_json)

# ---------------------------------------------------------
# SLEEP EMA
# ---------------------------------------------------------

sleep_json = os.path.join(
    DATASET_ROOT,
    "EMA",
    "response",
    "Sleep",
    "Sleep_u01.json"
)

inspect_json(sleep_json)

# ---------------------------------------------------------
# ACTIVITY SENSOR
# ---------------------------------------------------------

activity_csv = os.path.join(
    DATASET_ROOT,
    "sensing",
    "activity",
    "activity_u01.csv"
)

inspect_csv(activity_csv)

# ---------------------------------------------------------
# CONVERSATION SENSOR
# ---------------------------------------------------------

conversation_csv = os.path.join(
    DATASET_ROOT,
    "sensing",
    "conversation",
    "conversation_u01.csv"
)

inspect_csv(conversation_csv)

# ---------------------------------------------------------
# CALL LOG
# ---------------------------------------------------------

call_csv = os.path.join(
    DATASET_ROOT,
    "call_log",
    "call_log_u01.csv"
)

inspect_csv(call_csv)

# ---------------------------------------------------------
# SMS LOG
# ---------------------------------------------------------

sms_csv = os.path.join(
    DATASET_ROOT,
    "sms",
    "sms_u01.csv"
)

inspect_csv(sms_csv)

# ---------------------------------------------------------
# SURVEY DATA
# ---------------------------------------------------------

survey_csv = os.path.join(
    DATASET_ROOT,
    "survey",
    "PHQ-9.csv"
)

inspect_csv(survey_csv)

# =========================================================
# COUNT USERS PER MODALITY
# =========================================================

print_section("USER COVERAGE PER DATA SOURCE")

sources = {
    "Mood EMA": os.path.join(DATASET_ROOT, "EMA", "response", "Mood"),
    "Sleep EMA": os.path.join(DATASET_ROOT, "EMA", "response", "Sleep"),
    "Activity": os.path.join(DATASET_ROOT, "sensing", "activity"),
    "Conversation": os.path.join(DATASET_ROOT, "sensing", "conversation"),
    "Call Log": os.path.join(DATASET_ROOT, "call_log"),
    "SMS": os.path.join(DATASET_ROOT, "sms"),
}

for name, folder in sources.items():

    try:
        files = [
            f for f in os.listdir(folder)
            if not f.startswith(".")
        ]

        print(f"\n{name}")
        print(f"  Num files/users: {len(files)}")

        print("  Example files:")
        for f in sorted(files)[:5]:
            print(f"    {f}")

    except Exception as e:
        print(f"{name}: ERROR {e}")

# =========================================================
# RECOMMENDED FEATURES SUMMARY
# =========================================================

print_section("RECOMMENDED FEATURES FOR RL PIPELINE")

print("""
LIKELY FEATURE SOURCES

Mood:
    EMA/response/Mood/

Sleep:
    EMA/response/Sleep/

Activity:
    sensing/activity/

Social Interaction:
    sensing/conversation/
    call_log/
    sms/

OPTIONAL:
    sensing/audio/
    sensing/gps/
    app_usage/

IGNORE FOR NOW:
    education/
    calendar/
    dining/
    most event-specific EMA folders
""")

print_section("DONE")