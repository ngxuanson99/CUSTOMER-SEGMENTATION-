# Customer Segmentation using Clustering Techniques
> Identifying distinct travel customer segments using KMeans++ and Agglomerative clustering to develop targeted marketing strategies — built with Python (scikit-learn, Pandas, Matplotlib, Seaborn).

---

## 📌 Project Overview

In the travel industry, understanding who your customers are is essential for designing effective marketing campaigns. This project applies unsupervised machine learning clustering techniques to segment 2,000 travel customers into distinct groups based on demographic and socio-economic attributes — and translates those segments into actionable marketing strategies.

**Course:** BUSA8001 – Applied Predictive Analytics · Macquarie University (Oct 2024)

---

## 🎯 Business Problem

Travel agencies often apply one-size-fits-all marketing approaches, missing opportunities to personalise campaigns by customer profile. By identifying distinct customer segments, marketing teams can:

- Design targeted promotions that resonate with each group
- Allocate marketing budget more efficiently
- Improve customer engagement and conversion rates

---

## 📂 Dataset

| Detail | Description |
|---|---|
| Records | 2,000 customers |
| Variables | 7 (Gender, Marital Status, Age, Education, Income, Occupation, Settlement Size) |
| Missing values | None |
| Source | Travel agency customer database |

### Variable types

| Type | Features |
|---|---|
| Nominal (categorical) | Gender, Marital Status, Occupation, Settlement Size |
| Ordinal | Education |
| Numeric (continuous) | Age, Income |

---

## 🛠️ Tools & Techniques

| Tool / Library | Usage |
|---|---|
| **Python** | Core analysis language |
| **Pandas** | Data loading, cleaning, manipulation |
| **Matplotlib / Seaborn** | EDA visualisations, correlation matrix, cluster plots |
| **scikit-learn** | StandardScaler, KMeans++, AgglomerativeClustering, silhouette_score |
| **NumPy** | Numerical operations |

---

## 🔍 Methodology

### 1. Exploratory Data Analysis (EDA)

**Univariate analysis** — examined distributions of all 7 variables:
- 60.4% female, 39.6% male
- Majority high school (43.8%) or university educated (37.9%)
- 56.5% live in small cities; 39.9% in big cities
- Near-equal split between single (50.1%) and non-single (50.0%)
- ~50% unemployed or unskilled; 39.6% skilled workers

**Bivariate analysis** — key correlations from the correlation matrix:
- Age & Income (r=1.00): perfect positive correlation — income increases with age
- Marital Status & Occupation (r=0.90): non-single individuals tend to hold better career prospects
- Marital Status & Education (r=0.83): higher education associated with non-single status
- Education & Occupation (r=0.75): higher education linked to more advanced occupations
- Settlement Size & Occupation (r=0.72): larger city residents more likely to hold skilled jobs

---

### 2. Pre-processing

Numeric variables (Age and Income) were standardised using `StandardScaler` to prevent disproportionate influence on distance-based clustering algorithms.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scaled[['Age', 'Income']] = scaler.fit_transform(df[['Age', 'Income']])
```

---

### 3. Determining Optimal Number of Clusters

**Elbow Method:** Cluster inertia (Within-Cluster SSE) was plotted for k=1 to k=10. The curve flattened at k=2, indicating the optimal balance between variance reduction and model simplicity.

**Silhouette Analysis:** Silhouette plots were generated for k=2, k=3, and k=4 to validate the choice:

| k | Silhouette Score |
|---|---|
| k=2 | **0.54** ✅ best |
| k=3 | 0.43 |
| k=4 | 0.45 |

k=2 produced the most defined bands and highest average score — selected as optimal.

---

### 4. Clustering

Both algorithms were applied at k=2 and results compared:

#### KMeans++ (k=2)

| Cluster | Avg Age | Avg Income | Female % | Married % | Count |
|---|---|---|---|---|---|
| Cluster 1 | 48.2 | $173,443 | 86% | 99% | 976 |
| Cluster 2 | 33.8 | $103,341 | 36% | 3% | 1,024 |

#### Agglomerative Clustering (k=2)

| Cluster | Avg Age | Avg Income | Female % | Married % | Count |
|---|---|---|---|---|---|
| Cluster 1 | 47.1 | $168,085 | 77% | 86% | 1,163 |
| Cluster 2 | 32.1 | $95,041 | 37% | 1% | 837 |

Both techniques produced consistent, well-defined segments with minor differences in cluster boundaries — validating the robustness of the k=2 solution.

---

## 👥 Customer Segments

### Cluster 1 — Middle-aged, Wealthy, Female, Married, Big City
- Average age ~48 years, average income ~$170K
- Predominantly female (77–86%), almost all married
- Higher education and skilled/management occupations
- Concentrated in large cities

### Cluster 2 — Young, Lower-Income, Male, Single, Small City
- Average age ~33 years, average income ~$100K
- Predominantly male (63–64%), almost all single
- Lower education levels, unemployed or unskilled occupations
- Concentrated in small cities

---

## 💡 Marketing Recommendations

### Cluster 1 — Luxury & Family Focus
| Strategy | Description |
|---|---|
| Exclusive loyalty programme | VIP membership with personalised deals, early booking access, and points exchange for upgrades |
| Family travel packages | Bundled family vacation deals with kid-friendly and adult activities |

### Cluster 2 — Budget & Adventure Focus
| Strategy | Description |
|---|---|
| Group & solo travel discounts | Referral programmes rewarding customers who bring friends or book solo adventures |
| Budget-friendly adventures | Affordable bundles combining low-cost airlines, economy hotels, and activity packages |

---

## 📁 Repository Structure

```
├── data/
│   └── customer_data.csv              # Source dataset
├── notebooks/
│   └── customer_segmentation.ipynb    # Full analysis notebook
├── images/
│   ├── eda_categorical.png            # Univariate EDA charts
│   ├── eda_numeric.png                # Age and income distributions
│   ├── correlation_matrix.png         # Correlation heatmap
│   ├── elbow_method.png               # Elbow curve
│   ├── silhouette_k2.png              # Silhouette plot k=2
│   └── cluster_profiles.png          # Cluster summary charts
└── README.md
```

---

## ▶️ How to Run

```bash
# Clone the repository
git clone https://github.com/yourname/customer-segmentation.git
cd customer-segmentation

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# Launch notebook
jupyter notebook notebooks/customer_segmentation.ipynb
```

---

## 📈 Key Takeaways

- Both KMeans++ and Agglomerative clustering converged on **the same two-segment solution**, confirming the robustness of the k=2 choice
- The silhouette score of **0.54 at k=2** indicates well-separated, meaningful clusters
- Segment profiles align with real-world travel behaviour patterns — high-income married women favour family and luxury travel; young single males favour budget and adventure travel
- The segmentation provides a **directly actionable framework** for marketing campaign design

---

## 👤 Author

**Son Nguyen**
Master of Business Analytics — Macquarie University (WAM 84)
[LinkedIn](#) · [Portfolio](#) · Sydney, NSW

---

*This project was completed as part of BUSA8001 – Applied Predictive Analytics at Macquarie University (October 2024).*
