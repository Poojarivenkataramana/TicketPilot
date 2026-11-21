"""
TicketPilot - Sprint 3, Day 11: Knowledge Base Construction & Cleaning
----------------------------------------------------------------------
1. Source: Strictly training-set answers only (data/train.csv - 17,591 records).
2. Clean Placeholders: Replace synthetic template tags (<name>, <tel_num>, <email>, etc.)
   with professional, reusable knowledge base tokens.
3. Deduplicate: Remove duplicate resolution templates.
4. Add Metadata: Assign unique citation IDs (KB-00001), queue, and priority for RAG retrieval.
5. Strict Anti-Leakage Check: Formally verify 0 test tickets exist in the Knowledge Base.
6. Export: data/knowledge_base_clean.csv.
"""

import os
import re
import pandas as pd
import numpy as np

def clean_placeholders(text: str) -> str:
    """Standardizes synthetic tags into clean, professional KB documentation."""
    if not isinstance(text, str):
        return ""
    
    # Mapping synthetic/template tags to professional placeholders
    replacements = {
        r"<name>": "[Customer Name]",
        r"<name of agent>": "[Support Agent]",
        r"<Customer Support Agent>": "[Support Agent]",
        r"<Service Desk Agent Name>": "[Support Agent]",
        r"<Service Desk Agent>": "[Support Agent]",
        r"<service_desk_agent>": "[Support Agent]",
        r"<Your Name>": "[Support Agent]",
        r"<Customer Support>": "[Customer Support]",
        r"<Customer Support Team>": "[Customer Support Team]",
        r"<Customer Support Team Name>": "[Customer Support Team]",
        r"<Service Desk Team>": "[Service Desk Team]",
        r"<Company Name>": "[Company]",
        r"<company>": "[Company]",
        r"<Service Company>": "[Company]",
        r"<IT Services Company>": "[IT Services]",
        r"<email>": "support@company.com",
        r"<tel_num>": "[Support Hotline]",
        r"<contact_number>": "[Support Hotline]",
        r"<your_contact_info>": "[Contact Info]",
        r"<support_contact>": "[Support Contact]",
        r"<website_url>": "https://support.company.com",
        r"<service_link>": "https://support.company.com/portal",
        r"<order_num>": "[Order #]",
        r"<case_num>": "[Case #]",
        r"<acc_num>": "[Account #]",
        r"<cust_id>": "[Customer ID]",
        r"<date>": "[Date]",
        r"<invoice_date>": "[Invoice Date]",
        r"<purchase_date>": "[Purchase Date]",
        r"<time>": "[Time]",
        r"<price>": "[Amount]",
        r"<new_address>": "[Address]",
        r"<tool_name>": "[Tool Name]",
        r"<username>": "[Username]",
        r"<last_four_digits>": "[Last 4 Digits]",
        r"<br\s*/?>": "\n"
    }

    cleaned = text
    for pattern, repl in replacements.items():
        cleaned = re.sub(pattern, repl, cleaned, flags=re.IGNORECASE)

    # Clean any remaining rogue <...> tags
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    # Normalize whitespaces
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n\s*\n+", "\n\n", cleaned)
    return cleaned.strip()

def run_day11_build_kb():
    print("=" * 70)
    print("TicketPilot - Sprint 3, Day 11: Building Clean Knowledge Base")
    print("=" * 70)

    data_dir = "data"
    train_file = os.path.join(data_dir, "train.csv")
    test_file = os.path.join(data_dir, "test.csv")

    if not os.path.exists(train_file) or not os.path.exists(test_file):
        raise FileNotFoundError("Missing train.csv or test.csv in data/. Run earlier sprints first.")

    # 1. Load Training Data (Strictly Train Only)
    train_df = pd.read_csv(train_file)
    print(f"Step 1: Loaded {len(train_df):,} records strictly from '{train_file}'")

    # 2. Extract and Clean Solutions
    print("\nStep 2: Cleaning placeholders and standardizing resolutions...")
    train_df["clean_solution"] = train_df["answer"].apply(clean_placeholders)
    train_df["clean_problem"] = train_df["text"].apply(lambda x: re.sub(r"\s+", " ", str(x)).strip())

    # Drop empty or corrupted resolutions
    valid_mask = (train_df["clean_solution"].str.len() > 15) & (train_df["clean_problem"].str.len() > 10)
    valid_df = train_df[valid_mask].copy()
    print(f"Valid problem-solution pairs: {len(valid_df):,} (dropped {len(train_df) - len(valid_df)} invalid)")

    # 3. Deduplicate on Normalized Solution Text
    print("\nStep 3: Deduplicating identical resolution templates...")
    valid_df["solution_norm"] = valid_df["clean_solution"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    
    # Keep the first occurrence of each distinct resolution
    kb_df = valid_df.drop_duplicates(subset=["solution_norm"], keep="first").copy().reset_index(drop=True)
    duplicates_removed = len(valid_df) - len(kb_df)
    print(f"Duplicate resolutions removed: {duplicates_removed:,}")
    print(f"Distinct Knowledge Base articles: {len(kb_df):,}")

    # 4. Assign Citation IDs and Metadata
    print("\nStep 4: Assigning citation IDs and indexing metadata...")
    kb_df["kb_id"] = [f"KB-{i+1:05d}" for i in range(len(kb_df))]

    final_cols = ["kb_id", "queue", "priority", "clean_problem", "clean_solution"]
    kb_clean = kb_df[final_cols].rename(columns={
        "clean_problem": "problem",
        "clean_solution": "solution"
    })

    # 5. STRICT ANTI-LEAKAGE VERIFICATION & FILTERING
    print("\n" + "=" * 55)
    print("STEP 5: STRICT TEST SET LEAKAGE VERIFICATION")
    print("=" * 55)
    test_df = pd.read_csv(test_file)
    print(f"Checking against {len(test_df):,} held-out test tickets...")

    test_problems = set(test_df["text"].astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip())
    test_solutions = set(test_df["answer"].dropna().astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip())

    # Filter out any generic solution strings that also appear in the test set
    clean_mask = ~kb_clean["solution"].astype(str).str.lower().str.strip().isin(test_solutions)
    kb_clean = kb_clean[clean_mask].copy().reset_index(drop=True)
    
    # Re-assign sequential citation IDs
    kb_clean["kb_id"] = [f"KB-{i+1:05d}" for i in range(len(kb_clean))]

    kb_problems = set(kb_clean["problem"].astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip())
    kb_solutions = set(kb_clean["solution"].astype(str).str.lower().str.strip())
    
    problem_overlap = test_problems.intersection(kb_problems)
    solution_overlap = test_solutions.intersection(kb_solutions)

    print(f"  Test Question Overlap in KB: {len(problem_overlap)} tickets")
    print(f"  Test Solution Overlap in KB: {len(solution_overlap)} tickets")

    if len(problem_overlap) == 0 and len(solution_overlap) == 0:
        print("  VERIFICATION RESULT: 100% PASSED! Zero test questions or solutions exist in Knowledge Base.")
    else:
        raise ValueError(f"CRITICAL LEAKAGE DETECTED: {len(problem_overlap)} problems, {len(solution_overlap)} solutions!")

    # 6. Save Knowledge Base
    output_kb_path = os.path.join(data_dir, "knowledge_base_clean.csv")
    kb_clean.to_csv(output_kb_path, index=False)
    file_size_mb = os.path.getsize(output_kb_path) / (1024 * 1024)
    print(f"\nExported Clean Knowledge Base to: '{output_kb_path}' ({file_size_mb:.2f} MB)")

    # 7. Distribution by Department in Knowledge Base
    print("\nKnowledge Base Articles by Department:")
    queue_counts = kb_clean["queue"].value_counts()
    for q, count in queue_counts.items():
        pct = (count / len(kb_clean)) * 100
        print(f"  - {q:<35}: {count:5,} articles ({pct:5.2f}%)")

    # Sample Preview
    print("\n--- Sample Knowledge Base Article Preview ---")
    sample = kb_clean.iloc[0]
    print(f"Citation ID: {sample['kb_id']}")
    print(f"Department:  {sample['queue']} | Priority: {sample['priority']}")
    print(f"Problem:     {sample['problem'][:110]}...")
    print(f"Solution:    {sample['solution'][:140]}...")

    return kb_clean

if __name__ == "__main__":
    run_day11_build_kb()
