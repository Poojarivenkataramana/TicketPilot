# Day 3: Exploratory Data Analysis (EDA) Insights

**Project**: TicketPilot  
**Sprint**: 1 (Understand Data & Build Baseline)  
**Deliverable**: 5 Written Data-Backed Insights, each paired with an interactive Plotly chart in `charts/`.

---

### Insight 1: Heavy Class Imbalance Follows a Power-Law Distribution
* **Backed by**: [chart1_queue_distribution.html](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/charts/chart1_queue_distribution.html)
* **Data Observation**:
  - `Technical Support` dominates the volume with **7,325 tickets (29.15%)**, followed by `Product Support` with **4,659 (18.54%)** and `Customer Service` with **3,783 (15.05%)**.
  - Together, the top 3 queues account for **62.74%** of the entire dataset.
  - The bottom 3 queues (`Sales and Pre-Sales` at 3.05%, `Human Resources` at 1.89%, and `General Inquiry` at 1.42%) comprise only **6.36%** combined.
* **Engineering Impact**:
  - A naive classifier that blindly predicts the majority class would achieve ~29% accuracy while completely failing on customer inquiries, HR, and sales requests.
  - Stratified sampling is strictly required for the train/val/test split in Day 4.
  - Model evaluation must prioritize **Macro-F1** over raw accuracy so that rare queues carry equal evaluation weight.

---

### Insight 2: Departmental Identity Strongly Governs Priority Urgency
* **Backed by**: [chart2_priority_by_queue.html](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/charts/chart2_priority_by_queue.html)
* **Data Observation**:
  - Urgency is highly skewed by operational domain:
    - **`Service Outages and Maintenance`**: **71.15%** of tickets are `High` priority, with only 11.94% `Low`.
    - **`Technical Support`**: **58.31%** `High` priority vs. 11.44% `Low`.
    - **`IT Support`**: **47.03%** `High` priority vs. 9.87% `Low`.
  - In stark contrast, general business inquiry queues are skewed toward low urgency:
    - **`General Inquiry`**: **58.54%** `Low` priority, only 15.13% `High`.
    - **`Human Resources`**: **47.90%** `Low` priority, only 10.50% `High`.
* **Engineering Impact**:
  - The queue category serves as a powerful prior for predicting ticket priority.
  - In Sprint 5 (API & Routing), priority routing rules can leverage predicted queue confidence to auto-escalate outage/infrastructure tickets immediately.

---

### Insight 3: Uniform Conciseness Enables Lossless Transformer Context Windows
* **Backed by**: [chart3_word_count_by_queue.html](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/charts/chart3_word_count_by_queue.html)
* **Data Observation**:
  - Mean ticket length is **63.89 words** ($\approx 85$ subword tokens), with a median of **61.0 words** and standard deviation of **36.89**.
  - The middle 50% of tickets (IQR) fall between 36 and 85 words. The maximum length recorded is 374 words.
  - Length distributions are remarkably consistent across all 10 departments (averaging between 59.1 words for HR and 65.3 words for Billing).
* **Engineering Impact**:
  - 100% of customer tickets fit comfortably within the 512-token context limit of **DistilBERT** (Sprint 2) and modern sentence embedding models (Sprint 3) without requiring text truncation or chunking.

---

### Insight 4: High Lexical Separation for Vertical Queues vs. Semantic Ambiguity in Tech Support
* **Backed by**: [chart4_top_keywords_per_queue.html](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/charts/chart4_top_keywords_per_queue.html)
* **Data Observation**:
  - Class-specific TF-IDF reveals highly distinct discriminative vocabulary for specialized departments:
    - `Billing and Payments`: dominated by *billing* (0.1066), *payment* (0.0652), *details*, *invoice*.
    - `Service Outages`: dominated by *service* (0.0765), *outage* (0.0455), *outages* (0.0352), *impacting* (0.0295).
    - `Human Resources`: characterized by *employee* (0.0393) and *access* (0.0255).
  - However, `Technical Support` and `IT Support` share almost identical generic technical terminology: *data*, *issue*, *problem*, *software*, *support*.
* **Engineering Impact**:
  - Linear bag-of-words baselines (TF-IDF + Logistic Regression, Day 4) will achieve high precision on Billing and Outages, but will likely exhibit significant cross-confusion between `Technical Support` and `IT Support`.
  - Contextual representations (DistilBERT in Sprint 2) will be required to disambiguate enterprise IT problems from consumer product technical issues.

---

### Insight 5: Ticket Length Is a Zero-Information Feature for Priority Classification
* **Backed by**: [chart5_length_vs_priority.html](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/charts/chart5_length_vs_priority.html)
* **Data Observation**:
  - Counter to the hypothesis that urgent outages or high-priority incidents contain longer logs or more verbose descriptions, word counts are virtually identical across all priority tiers:
    - **High Priority**: Mean = **63.81 words**, Median = **61.0**
    - **Medium Priority**: Mean = **64.16 words**, Median = **62.0**
    - **Low Priority**: Mean = **63.50 words**, Median = **61.0**
* **Engineering Impact**:
  - Priority cannot be inferred from text volume or superficial length features.
  - The Day 5 priority classifier must rely strictly on semantic urgency indicators (e.g., severity adjectives, action verbs, outage nouns, and SLA impact keywords).

---

## "Done When" Verification Check
- [x] Label counts analyzed across all queues and priority tiers.
- [x] Text lengths and summary statistics quantified.
- [x] Class-specific distinctive keywords extracted via TF-IDF n-grams.
- [x] 5 standalone Plotly visualizations generated in `charts/`.
- [x] 5 comprehensive written insights recorded, each directly referencing its visual chart.
