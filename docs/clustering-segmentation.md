# Clustering & Customer Segmentation Documentation

This document explains the behavioral segmentation system — how K-Means clustering groups customers into actionable segments, the methods used to determine optimal clusters, and how to interpret results for business decisions.

---

## Overview

While the classification models predict **who** will churn, clustering answers a different question: **what types of customers do we have?** Segmentation groups customers by behavioral patterns, enabling targeted marketing and retention strategies.

```
Raw Customer Data (6 behavioral features)
    │
    ▼ StandardScaler
Scaled Features
    │
    ├──► Elbow Method (find optimal K)
    ├──► Silhouette Analysis (validate K)
    │
    ▼ K-Means (K=4)
Cluster Labels (0, 1, 2, 3)
    │
    ▼ interpret_segments()
Business Labels + Profiles
    │
    ├──► segment_profiles.csv
    ├──► elbow_method.png
    └──► silhouette_analysis.png
```

---

## Clustering Features

**File:** `src/models/clustering.py`

The clustering uses 6 behavioral features (not the full 36-column encoded dataset):

| Feature | Why It's Used | Business Meaning |
|---------|--------------|-----------------|
| `tenure_months` | Customer lifecycle stage | New vs. established relationship |
| `monthly_charges` | Revenue contribution | High-value vs. budget customer |
| `total_charges` | Lifetime value | Overall revenue generated |
| `num_support_tickets` | Service satisfaction | Frustrated vs. satisfied |
| `monthly_minutes_used` | Product engagement | Active vs. inactive usage |
| `data_usage_gb` | Digital engagement | Power user vs. light user |

**Why only 6 features?**
- Clustering works best with meaningful, interpretable features
- One-hot encoded columns (e.g., `contract_type_One year`) are binary and distort distance calculations
- These 6 features capture the most important behavioral dimensions

### Feature Scaling for Clustering

Features are re-scaled independently from the classification pipeline using a separate `StandardScaler`. This is critical because K-Means uses **Euclidean distance** — without scaling, features with large ranges (like `total_charges`: 0-10,000) would dominate over small-range features (like `num_support_tickets`: 0-10).

---

## Finding the Optimal Number of Clusters

### Elbow Method

**What it does:** Tests K-Means with K=2 through K=10, plotting the **inertia** (within-cluster sum of squared distances) for each K.

**How to read the chart:**
```
Inertia
  │
  │ X
  │  \
  │   \
  │    \___        ← "elbow" = optimal K
  │        ----____
  │                ------
  └───────────────────────── K
  2   3   4   5   6   7   8   9  10
```

- As K increases, inertia always decreases (more clusters = tighter groups)
- The "elbow" is where adding more clusters gives **diminishing returns**
- The elbow suggests where cluster quality stops improving significantly

**Saved at:** `outputs/visualizations/elbow_method.png`

### Silhouette Analysis

**What it does:** For each K, computes the **silhouette score** — a measure of how similar each point is to its own cluster vs. the nearest other cluster.

**Silhouette score interpretation:**

| Score | Meaning |
|-------|---------|
| +1.0 | Perfect — point is far from other clusters |
| 0.0 | On the boundary between two clusters |
| -1.0 | Misclassified — closer to another cluster |

**Average silhouette scores from our data:**

| K | Silhouette Score | Interpretation |
|---|-----------------|----------------|
| 2 | 0.2566 | Best score, but too few segments for business use |
| 3 | 0.1801 | Moderate |
| **4** | **0.1887** | Good balance of separation and business utility |
| 5 | 0.1906 | Marginally better but more complex |
| 6+ | Declining | Clusters become too fragmented |

**Why K=4?**
- K=2 has the best silhouette but only gives "high value" vs. "low value" — not actionable enough
- K=4 provides 4 distinct, business-meaningful segments
- Silhouette of 0.19 indicates moderate but real structure in the data
- K=4 is a common business-friendly number for customer segmentation

**Saved at:** `outputs/visualizations/silhouette_analysis.png`

---

## K-Means Algorithm

### How K-Means Works

1. **Initialize:** Randomly place K=4 centroids in the feature space
2. **Assign:** Each customer is assigned to the nearest centroid (by Euclidean distance)
3. **Update:** Move each centroid to the mean of its assigned customers
4. **Repeat:** Steps 2-3 until centroids stop moving (convergence)

**Settings:**
- `n_clusters=4`: Four customer segments
- `n_init=10`: Run the algorithm 10 times with different random starts, keep the best
- `random_state=42`: Reproducible results

### Why `n_init=10`?

K-Means is sensitive to initial centroid placement. Running it 10 times with different starts and keeping the best result (lowest inertia) provides more robust clusters.

---

## Segment Profiles

After clustering, each segment is profiled by computing average feature values:

### Example Output

| Segment | Count | Avg Tenure | Avg Monthly Charges | Avg Total Charges | Avg Support Tickets | Avg Minutes | Avg Data GB | Churn Rate | Business Label |
|---------|-------|------------|--------------------|--------------------|-----------------------|-------------|-------------|------------|---------------|
| 0 | 29,297 | 13.5 mo | $92.70 | $1,240 | 1.48 | 451 | 5.6 | 69.7% | At-Risk / High Spenders |
| 1 | 9,722 | 21.9 mo | $65.95 | $1,350 | 1.49 | 446 | 25.1 | 51.5% | New Power Users |
| 2 | 32,736 | 21.4 mo | $40.33 | $815 | 1.50 | 450 | 5.6 | 42.9% | Budget-Conscious |
| 3 | 19,382 | 59.0 mo | $82.87 | $4,794 | 1.50 | 451 | 7.0 | 31.2% | Loyal High-Value |

---

## Business Segment Interpretation

### How Labels Are Assigned

The labeling algorithm analyzes each segment's profile relative to the overall population:

```python
if avg_tenure > 36 and churn_rate < median_churn:
    → "Loyal High-Value"
elif churn_rate > 75th percentile and high charges:
    → "At-Risk / High Spenders"
elif churn_rate > 75th percentile and low charges:
    → "At-Risk / Frustrated"
elif avg_monthly_charges < 30th percentile:
    → "Budget-Conscious"
elif avg_data_gb > 1.5x median:
    → "New Power Users"
else:
    → "Mid-Tier Stable"
```

### Segment Deep Dives

#### Segment: Loyal High-Value

**Profile:**
- Long tenure (avg ~59 months / ~5 years)
- High monthly charges (~$83)
- Very high total charges (~$4,800 lifetime)
- Low churn rate (~31%)

**Business interpretation:** These are your best customers. They've been with you for years, pay well, and are unlikely to leave. They represent the gold standard of customer relationships.

**Recommended actions:**
- Loyalty rewards programs
- Early access to new features
- Referral incentive programs
- Premium support channels
- Avoid price increases on this segment

---

#### Segment: At-Risk / High Spenders

**Profile:**
- Short tenure (avg ~13 months)
- Highest monthly charges (~$93)
- Highest churn rate (~70%)

**Business interpretation:** These customers are paying a lot but haven't committed long-term. They're the most likely to leave, and each one lost represents significant revenue. They may feel they're not getting value for their money.

**Recommended actions:**
- Immediate retention outreach
- Personalized discount offers
- Contract upgrade incentives (lock in at lower rate)
- Satisfaction surveys to identify pain points
- Assign dedicated account managers

---

#### Segment: Budget-Conscious

**Profile:**
- Medium tenure (avg ~21 months)
- Lowest monthly charges (~$40)
- Moderate churn rate (~43%)

**Business interpretation:** Price-sensitive customers who use basic services. They're stable but won't tolerate price increases. Low revenue per customer but large segment size means significant total revenue.

**Recommended actions:**
- Value-add bundles at current price point
- Usage-based upgrade suggestions
- Avoid aggressive upselling
- Highlight value of existing services

---

#### Segment: New Power Users

**Profile:**
- Medium tenure (avg ~22 months)
- Moderate charges (~$66)
- Very high data usage (~25 GB, 4x other segments)
- Moderate churn rate (~52%)

**Business interpretation:** Heavy data users who may outgrow their current plan or find better deals. Their high engagement is a positive sign but needs to be converted into long-term loyalty.

**Recommended actions:**
- Data-focused plans and upgrades
- Streaming bundles
- Tech-forward marketing
- Network quality assurance (they'll notice degradation first)

---

## Using Clustering Results

### Loading the Clustering Model

```python
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load models
kmeans = joblib.load("models/saved/kmeans_model.joblib")
scaler = joblib.load("models/saved/clustering_scaler.joblib")

# Prepare new customer data
new_customer = pd.DataFrame([{
    "tenure_months": 8,
    "monthly_charges": 95.0,
    "total_charges": 760.0,
    "num_support_tickets": 4,
    "monthly_minutes_used": 300,
    "data_usage_gb": 12.0,
}])

# Scale and predict segment
scaled = scaler.transform(new_customer)
segment = kmeans.predict(scaled)
print(f"Customer belongs to segment: {segment[0]}")
```

### Segment Assignment for Batch Data

```python
# Assign segments to entire customer base
clustering_features = [
    "tenure_months", "monthly_charges", "total_charges",
    "num_support_tickets", "monthly_minutes_used", "data_usage_gb"
]
scaled_data = scaler.transform(df[clustering_features].dropna())
df["segment"] = kmeans.predict(scaled_data)
```

---

## Saved Artifacts

| File | Location | Description |
|------|----------|-------------|
| `kmeans_model.joblib` | `models/saved/` | Trained K-Means model (4 clusters) |
| `clustering_scaler.joblib` | `models/saved/` | StandardScaler fitted on clustering features |
| `segment_profiles.csv` | `outputs/reports/` | Segment profiles with business labels |
| `elbow_method.png` | `outputs/visualizations/` | Elbow method chart |
| `silhouette_analysis.png` | `outputs/visualizations/` | Silhouette score chart |

---

## Limitations and Considerations

1. **K-Means assumes spherical clusters** — if customer groups have non-spherical shapes, DBSCAN or Gaussian Mixture Models may perform better
2. **Feature selection matters** — different features produce different segments. The 6 chosen features were selected for business interpretability
3. **Segments are not static** — customers move between segments over time. Re-clustering periodically (monthly/quarterly) is recommended
4. **Silhouette scores of ~0.19 indicate moderate cluster separation** — the segments are real but have some overlap. This is normal for behavioral data
5. **Cluster labels (0, 1, 2, 3) are arbitrary** — they may change between runs. Always use the business labels for communication
