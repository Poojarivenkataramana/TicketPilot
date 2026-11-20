"""
TicketPilot - Sprint 1, Day 2: Data Cleaning & Preprocessing Pipeline
-------------------------------------------------------------------
Steps:
1. Load df1, df4, df5 (files containing support answers).
2. Filter for English tickets ('language' == 'en').
3. Handle missing values:
   - Handle missing subject by filling empty string (or dropping if strict).
   - Drop corrupt rows missing 'body' or 'answer'.
4. Combine 'subject' and 'body' into a single 'text' field.
5. Deduplicate identical tickets (prevent train/test leakage).
6. Verify and document final label sets ('queue' and 'priority').
7. Export 'cleaned_tickets.csv'.
"""

import pandas as pd
import os

def run_day2_cleaning(strict_subject_drop: bool = False):
    print("=" * 60)
    print("TicketPilot - Sprint 1, Day 2: Data Cleaning")
    print("=" * 60)

    # 1. Load candidate datasets
    files = {
        "df1": "aa_dataset-tickets-multi-lang-5-2-50-version.csv",
        "df4": "dataset-tickets-multi-lang3-4k.csv",
        "df5": "dataset-tickets-multi-lang-4-20k.csv"
    }

    core_cols = ["subject", "body", "answer", "type", "queue", "priority", "language"]
    dfs = []
    for name, fpath in files.items():
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing {fpath}")
        df = pd.read_csv(fpath)
        print(f"Loaded {name} ({fpath}): {len(df):,} rows")
        dfs.append(df[core_cols])

    combined = pd.concat(dfs, ignore_index=True)
    initial_count = len(combined)
    print(f"\nStep 1: Total combined raw rows: {initial_count:,}")

    # 2. Filter English only
    en_df = combined[combined["language"] == "en"].copy()
    en_count = len(en_df)
    print(f"Step 2: English tickets ('language' == 'en'): {en_count:,} rows (dropped {initial_count - en_count:,} non-English)")

    # 3. Missing values inspection & handling
    print("\nMissing values in English records:")
    print(en_df.isnull().sum())

    if strict_subject_drop:
        # Drop rows missing subject, body, or answer
        cleaned_nulls = en_df.dropna(subset=["subject", "body", "answer", "queue", "priority"]).copy()
        nulls_dropped = en_count - len(cleaned_nulls)
        print(f"\nStep 3 (Strict): Dropped {nulls_dropped:,} rows with null in subject/body/answer.")
    else:
        # Fill missing subject with empty string so valid ticket bodies aren't discarded
        en_df["subject"] = en_df["subject"].fillna("").astype(str).str.strip()
        cleaned_nulls = en_df.dropna(subset=["body", "answer", "queue", "priority"]).copy()
        nulls_dropped = en_count - len(cleaned_nulls)
        print(f"\nStep 3 (Preserve bodies): Dropped {nulls_dropped:,} rows missing body/answer; filled missing subjects.")

    after_nulls_count = len(cleaned_nulls)
    print(f"Rows after missing value handling: {after_nulls_count:,}")

    # 4. Combine subject and body into one normalized text field
    cleaned_nulls["subject_clean"] = cleaned_nulls["subject"].astype(str).str.strip()
    cleaned_nulls["body_clean"] = cleaned_nulls["body"].astype(str).str.strip()

    # If subject exists, combine 'subject + \n\n + body', else body alone
    cleaned_nulls["text"] = cleaned_nulls.apply(
        lambda r: f"{r['subject_clean']} {r['body_clean']}".strip() if r['subject_clean'] else r['body_clean'],
        axis=1
    )

    # Normalize whitespace
    cleaned_nulls["text"] = cleaned_nulls["text"].str.replace(r"\s+", " ", regex=True).str.strip()

    # 5. Deduplicate based on cleaned ticket text to prevent data leakage
    cleaned_nulls["text_normalized"] = cleaned_nulls["text"].str.lower()
    dedup_df = cleaned_nulls.drop_duplicates(subset=["text_normalized"], keep="first").copy()
    dedup_df = dedup_df.drop(columns=["text_normalized", "subject_clean", "body_clean"])

    final_count = len(dedup_df)
    duplicates_removed = after_nulls_count - final_count
    print(f"\nStep 4 & 5: Combined text & Deduplicated.")
    print(f"Duplicates removed: {duplicates_removed:,}")
    print(f"Final Cleaned Row Count: {final_count:,}")

    # 6. Final Label Sets
    print("\n" + "=" * 40)
    print("FINAL LABEL SETS & DISTRIBUTION")
    print("=" * 40)
    print("\nQueue Distribution (10 classes):")
    queue_counts = dedup_df["queue"].value_counts()
    for q, count in queue_counts.items():
        pct = (count / final_count) * 100
        print(f"  - {q:<35}: {count:5,} ({pct:5.2f}%)")

    print("\nPriority Distribution (3 classes):")
    priority_counts = dedup_df["priority"].value_counts()
    for p, count in priority_counts.items():
        pct = (count / final_count) * 100
        print(f"  - {p:<10}: {count:5,} ({pct:5.2f}%)")

    # 7. Export clean file
    output_filename = "cleaned_tickets.csv"
    dedup_df.to_csv(output_filename, index=False)
    print(f"\nExported clean dataset to: {output_filename} ({os.path.getsize(output_filename):,} bytes)")

    return dedup_df

if __name__ == "__main__":
    # Run with standard preservation of valid ticket bodies
    df = run_day2_cleaning(strict_subject_drop=False)
