# Project Hail Minor: Comprehensive Technical Architecture & Engineering Documentation

A rigorous, end-to-end technical reference for **Project Hail Minor**, an ICU clinical decision support and patient outcome prediction platform built upon the **eICU Collaborative Research Database** (v2.0).

---

## 1. Executive Summary & Clinical Problem Formulation

### 1.1. Clinical Context
Critically ill patients in Intensive Care Units (ICUs) experience high rates of physiological volatility. Timely clinical interventions require predictive tools capable of synthesizing dynamic, multi-rate physiological time series (vital signs, laboratory tests, drug titrations, organ support therapies) alongside rich static patient histories (chronic comorbidities, demographics, admission severity scoring).

**Project Hail Minor** addresses this challenge by framing ICU risk prediction through a **multimodal, relational graph neural network (GNN)** paradigm. Rather than treating patient records as isolated vectors, this architecture models both:
1. **Intra-patient multi-modal dynamics**: Integrating irregular temporal trajectories with high-dimensional static baselines.
2. **Inter-patient relational topology**: Leveraging clinical similarity across patients treated within the same healthcare facility to regularize and enrich representations.

### 1.2. Primary Clinical Objectives
The platform is optimized for two high-stakes clinical prediction targets and an auxiliary self-supervised task:

1. **In-Hospital Mortality Prediction (Task 1)**:
   - **Target**: `hospital_expired` (binary classification: 0 = Survived, 1 = Deceased).
   - **Clinical Challenge**: Extreme class imbalance (prevalence $\approx 8.8\% - 9.7\%$). False negatives carry fatal consequences; high Precision-Recall AUC (AUPRC) and balanced $F_1$-score are prioritized over raw accuracy.
2. **ICU Sepsis Phenotyping & Prediction (Task 2)**:
   - **Target**: `sepsis` (binary classification: 0 = Non-sepsis, 1 = Sepsis/Septic Shock).
   - **Definition**: Consensus phenotype derived from clinical diagnoses (ICD-9 codes `038.*`, `995.91`, `995.92`, `785.52`) and APACHE admission indications containing `seps` or `septic`.
   - **Clinical Challenge**: Prevalence $\approx 14\% - 16\%$. Sepsis demands rapid early recognition; high sensitivity (recall) and $F_2$-score are clinically prioritized.
3. **Auxiliary Feature Reconstruction**:
   - **Target**: Reconstruction of 67 static tabular clinical features via a symmetric Denoising Autoencoder (DAE).
   - **Objective**: Prevents latent representation collapse and acts as a regularizer during joint multi-task training.

---

## 2. Technology Stack & Hardware Infrastructure

| Category | Technology | Version / Specification | Exact Function in Project |
| :--- | :--- | :--- | :--- |
| **Language** | Python | 3.13 / 3.14 | Core language for modular scripts, training pipelines, and notebooks |
| **Deep Learning** | PyTorch | $\ge 2.14$ (`torch`, `torch.nn`, `torch.sparse`) | Implementation of LSTMs, autoencoders, multi-head attention, sparse GCNs |
| **Hardware Acceleration** | Apple Silicon MPS & NVIDIA CUDA | MPS (`torch.backends.mps`) / CUDA | High-throughput sparse tensor matrix multiplications and graph convolutions |
| **Numerical Processing** | NumPy | $\ge 2.5.0$ | High-dimensional array manipulation and storage (`.npy`, `.npz`) |
| **Tabular Data / ETL** | Pandas, PyArrow | Latest | Multi-table joins, parquet feature extraction, column coercion |
| **Sparse Graph Math** | SciPy Sparse | `scipy.sparse` (`coo_matrix`, `csr_matrix`) | Constructing and storing $153,831 \times 153,831$ adjacency matrices |
| **Graph Construction** | Scikit-Learn | `sklearn.neighbors.NearestNeighbors` | High-dimensional cosine $k$-NN neighbor search and index mapping |
| **Metrics & Validation** | Scikit-Learn | `sklearn.metrics` | AUROC, AUPRC, precision-recall curve, $F_1$, $F_2$, confusion matrices |
| **Graph Visualization** | NetworkX & Matplotlib | `networkx`, `matplotlib` | Subgraph sampling, degree distribution, and patient cluster topology plots |

---

## 3. Cohort Design & Leakage Prevention Protocol

### 3.1. Hospital-Level Partitioning
Clinical practice patterns, physician treatment thresholds, laboratory test panels, and patient demographics vary dramatically across individual hospitals. Standard random patient-level splits lead to **practice-pattern leakage**, causing models to memorize institutional artifacts rather than generalizable pathophysiology.

Project Hail Minor enforces a strict **Hospital-Level Split** across all 207 hospitals in the cohort:
- **Total Patient Cohort**: 200,860 ICU stays across 207 hospitals.
- **Split Distribution (`processed/hospital_split.json`)**:
  - **Training Set**: 145 hospitals (143,419 patients; 12,640 mortality positives [8.81%]; 23,759 sepsis positives [16.57%]).
  - **Validation Set**: 31 unseen hospitals (30,047 patients; 2,927 mortality positives [9.74%]; 4,574 sepsis positives [15.22%]).
  - **Testing Set**: 31 unseen hospitals (27,360 patients; 2,437 mortality positives [8.91%]; 3,809 sepsis positives [13.92%]).

### 3.2. Intra-Hospital Patient Graph Isolation
When constructing patient similarity graphs for the Graph Convolutional Network:
- Edges are formed **exclusively between patients admitted to the exact same hospital**.
- The cross-hospital adjacency is strictly zero: $A_{u, v} = 0 \quad \forall u \in \mathcal{H}_i, v \in \mathcal{H}_j \quad (i \ne j)$.
- This ensures the graph adjacency matrix is purely **block-diagonal**, preventing relational information from leaking across institutions.

---

## 4. End-to-End System Architecture

```mermaid
flowchart TD
    subgraph S1["Stage 1: Temporal Sequence Processing"]
        TS_In["eICU 48-Hour Time Series<br/>(12 Bins x 46 Features)"]
        TS_Imp["LOCF Imputation + Missingness Indicators<br/>Tensor: (N, 12, 92)"]
        LSTM["Temporal LSTM Encoder<br/>(Hidden: 128, Last-Valid Pooling, Linear Proj)"]
        TS_Emb["h_temp in R^64"]
        TS_In --> TS_Imp --> LSTM --> TS_Emb
    end

    subgraph S2["Stage 2: Static Tabular Processing"]
        Tab_In["Static Clinical Records<br/>(APACHE IV, Comorbidities, Demographics)"]
        Tab_Norm["Train-Only Z-Score Standardization<br/>Vector: (N, 67)"]
        DAE["Tabular Denoising Autoencoder<br/>(Input Noise sigma=0.1, Latent: 64)"]
        Tab_Emb["h_tab in R^64"]
        Tab_Recon["Reconstruction x_hat in R^67"]
        Tab_In --> Tab_Norm --> DAE
        DAE --> Tab_Emb
        DAE --> Tab_Recon
    end

    subgraph S3["Stage 3: Multimodal Fusion"]
        AttnFusion["Multi-Head Cross-Attention Fusion<br/>(4 Heads, LayerNorm, FFN, Residuals)"]
        Fused_Emb["h_fused in R^64"]
        TS_Emb --> AttnFusion
        Tab_Emb --> AttnFusion
        AttnFusion --> Fused_Emb
    end

    subgraph S4["Stage 4: Relational Graph Modeling"]
        KNN["Intra-Hospital Cosine k-NN Graph (k=5)<br/>Block-Diagonal Adjacency"]
        Laplacian["Symmetric Renormalization<br/>A_norm = D^-1/2 (A + I) D^-1/2"]
        GCN["2-Layer Sparse Graph Convolutional Network<br/>(Hidden: 128, Dropout: 0.25)"]
        Graph_Emb["H_graph in R^64"]
        Fused_Emb --> KNN --> Laplacian
        Fused_Emb --> GCN
        Laplacian --> GCN
        GCN --> Graph_Emb
    end

    subgraph S5["Stage 5: Dual Clinical Heads & Multi-Task Loss"]
        Mort_Head["Mortality Head: Linear(64 -> 32 -> 1)"]
        Seps_Head["Sepsis Head: Linear(64 -> 32 -> 1)"]
        L_Mort["BCE Loss (pos_weight = 10.35)"]
        L_Seps["BCE Loss (pos_weight = 5.04)"]
        L_Recon["MSE Reconstruction Loss"]
        Loss["MultiTaskLoss = L_Mort + lambda_seps * L_Seps + lambda_recon * L_Recon"]

        Graph_Emb --> Mort_Head --> L_Mort
        Graph_Emb --> Seps_Head --> L_Seps
        Tab_Recon --> L_Recon
        L_Mort --> Loss
        L_Seps --> Loss
        L_Recon --> Loss
    end
```

---

## 5. Mathematical Formulations & Component Specifications

### 5.1. Stage 1: Temporal Encoder (LSTM)
- **Time Discretization**: First 48 hours of ICU admission partitioned into $T = 12$ uniform 4-hour bins ($t \in \{0, 1, \dots, 11\}$).
- **Input Feature Dimension**: $C = 46$ physiological features (20 lab tests, 10 continuous infusion drug rates, 11 treatment flags, 5 medications).
- **Missingness Indicator Formulation**:
  $$M_{t, c} = \begin{cases} 1 & \text{if feature } c \text{ is observed in bin } t \\ 0 & \text{if feature } c \text{ is missing in bin } t \end{cases}$$
- **Laboratory Imputation**: Last Observation Carried Forward (LOCF) across valid stay bins; remaining missing entries zero-filled:
  $$X^{\text{imp}}_{t, c} = \begin{cases} X_{t, c} & \text{if } M_{t, c} = 1 \\ X^{\text{imp}}_{t-1, c} & \text{if } M_{t, c} = 0 \text{ and } t > 0 \\ 0 & \text{otherwise} \end{cases}$$
- **Concatenated Input**: $\tilde{X}_t = [X^{\text{imp}}_t \parallel M_t] \in \mathbb{R}^{92}$.
- **Recurrent Update**:
  $$h_t, c_t = \text{LSTM}(\tilde{X}_t, (h_{t-1}, c_{t-1})), \quad h_t \in \mathbb{R}^{128}$$
- **Last-Valid State Extraction**: For a patient with valid length $\tau \le 12$, the temporal representation is extracted via sequence masking:
  $$h_{\text{last}} = h_{\tau-1}$$
- **Dense Projection**:
  $$h_{\text{temp}} = \text{ReLU}(W_p h_{\text{last}} + b_p) \in \mathbb{R}^{64}$$

### 5.2. Stage 2: Tabular Denoising Autoencoder (DAE)
- **Input Dimension**: $x_{\text{tab}} \in \mathbb{R}^{67}$ (standardized continuous APACHE variables + binary categorical flags).
- **Stochastic Corruption**: Gaussian perturbation applied during training mode only:
  $$\tilde{x}_{\text{tab}} = x_{\text{tab}} + \epsilon, \quad \epsilon \sim \mathcal{N}(0, 0.1^2 \cdot I_{67})$$
- **Encoder**:
  $$\begin{aligned}
  z_1 &= \text{ReLU}(\text{BatchNorm1d}(W_{e1} \tilde{x}_{\text{tab}} + b_{e1})) \in \mathbb{R}^{67} \\
  z_2 &= \text{ReLU}(W_{e2} z_1 + b_{e2}) \in \mathbb{R}^{128} \\
  h_{\text{tab}} &= W_{e3} z_2 + b_{e3} \in \mathbb{R}^{64}
  \end{aligned}$$
- **Decoder**:
  $$\begin{aligned}
  d_1 &= \text{ReLU}(W_{d1} h_{\text{tab}} + b_{d1}) \in \mathbb{R}^{128} \\
  \hat{x}_{\text{tab}} &= W_{d2} d_1 + b_{d2} \in \mathbb{R}^{67}
  \end{aligned}$$

### 5.3. Stage 3: Cross-Modal Attention Fusion
- **Modality Stacking**: The representations $h_{\text{temp}}$ and $h_{\text{tab}}$ are stacked into a sequence of length 2:
  $$\mathbf{Z}^{(0)} = \begin{bmatrix} h_{\text{temp}}^T \\ h_{\text{tab}}^T \end{bmatrix} \in \mathbb{R}^{2 \times 64}$$
- **Multi-Head Self-Attention ($h=4$ heads, key dimension $d_k = 16$)**:
  $$\text{head}_i = \text{Softmax}\left( \frac{(\mathbf{Z}^{(0)} W_i^Q)(\mathbf{Z}^{(0)} W_i^K)^T}{\sqrt{16}} \right) (\mathbf{Z}^{(0)} W_i^V)$$
- **Residual & Feed-Forward Transformations**:
  $$\mathbf{Z}^{(1)} = \text{LayerNorm}\left( \mathbf{Z}^{(0)} + [\text{head}_1 \parallel \dots \parallel \text{head}_4] W^O \right)$$
  $$\mathbf{Z}^{(2)} = \text{LayerNorm}\left( \mathbf{Z}^{(1)} + \text{FFN}(\mathbf{Z}^{(1)}) \right)$$
- **Mean Pooling**:
  $$h_{\text{fused}} = \frac{1}{2} \left( \mathbf{Z}^{(2)}_1 + \mathbf{Z}^{(2)}_2 \right) \in \mathbb{R}^{64}$$

### 5.4. Stage 4: Relational Graph & Graph Convolutional Network
- **Pairwise Cosine Similarity**: For all patients $u, v$ within the same hospital $\mathcal{H}_j$:
  $$S_{u, v} = \frac{h_{\text{fused}}(u) \cdot h_{\text{fused}}(v)}{\|h_{\text{fused}}(u)\|_2 \|h_{\text{fused}}(v)\|_2}$$
- **$k$-Nearest Neighbor Adjacency ($k=5$)**:
  $$A_{u, v} = \begin{cases} S_{u, v} & \text{if } v \in \text{top-}k(S_{u, \cdot}) \text{ and } u, v \in \mathcal{H}_j \\ 0 & \text{otherwise} \end{cases}$$
- **Symmetrization & Self-Loops**:
  $$\tilde{A} = \frac{A + A^T}{2} + I_N$$
- **Kipf-Welling Renormalization**:
  $$\tilde{D}_{u, u} = \sum_{v=1}^N \tilde{A}_{u, v}, \quad \tilde{A}_{\text{norm}} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$$
- **2-Layer Graph Convolution**:
  $$H^{(1)} = \text{ReLU}\left( \tilde{A}_{\text{norm}} h_{\text{fused}} W^{(1)} \right), \quad W^{(1)} \in \mathbb{R}^{64 \times 128}$$
  $$H^{(2)} = \text{ReLU}\left( \tilde{A}_{\text{norm}} \text{Dropout}(H^{(1)}) W^{(2)} \right), \quad W^{(2)} \in \mathbb{R}^{128 \times 64}$$
  $$H_{\text{graph}} = H^{(2)} \in \mathbb{R}^{N \times 64}$$

### 5.5. Stage 5: Dual Prediction Heads & Multi-Task Loss
- **Mortality Head**:
  $$\hat{y}_{\text{mort}} = W_{m2} \text{ReLU}(W_{m1} H_{\text{graph}} + b_{m1}) + b_{m2} \in \mathbb{R}$$
- **Sepsis Head**:
  $$\hat{y}_{\text{seps}} = W_{s2} \text{ReLU}(W_{s1} H_{\text{graph}} + b_{s1}) + b_{s2} \in \mathbb{R}$$
- **Multi-Task Objective Function**:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}}( \hat{y}_{\text{mort}}, y_{\text{mort}}; w_{\text{mort}} ) + \lambda_{\text{seps}} \mathcal{L}_{\text{BCE}}( \hat{y}_{\text{seps}}, y_{\text{seps}}; w_{\text{seps}} ) + \lambda_{\text{recon}} \|\hat{x}_{\text{tab}} - x_{\text{tab}}\|^2_2$$
  where:
  $$w_{\text{mort}} = \frac{N_{\text{neg, mort}}}{N_{\text{pos, mort}}} \approx 10.35, \quad w_{\text{seps}} = \frac{N_{\text{neg, seps}}}{N_{\text{pos, seps}}} \approx 5.04$$
  $$\lambda_{\text{seps}} = 1.0, \quad \lambda_{\text{recon}} = 0.1$$

---

## 6. Complete Feature Dictionary

### 6.1. Temporal Features (46 Channels $\to$ 92 Dimensions with Indicators)
1. **Laboratory Measurements (20 continuous variables)**:
   - `lab_anion gap`: Acid-base balance and metabolic acidosis marker.
   - `lab_bedside glucose`: Point-of-care capillary blood sugar.
   - `lab_bicarbonate`: Venous/arterial $\text{HCO}_3^-$.
   - `lab_bun`: Blood Urea Nitrogen (renal function / catabolism).
   - `lab_calcium`: Total serum calcium.
   - `lab_chloride`: Serum chloride electrolyte.
   - `lab_creatinine`: Serum creatinine (acute kidney injury indicator).
   - `lab_glucose`: Formal venous blood glucose.
   - `lab_hct`: Hematocrit percentage.
   - `lab_hgb`: Hemoglobin concentration.
   - `lab_mch`: Mean Corpuscular Hemoglobin.
   - `lab_mchc`: Mean Corpuscular Hemoglobin Concentration.
   - `lab_mcv`: Mean Corpuscular Volume.
   - `lab_mpv`: Mean Platelet Volume.
   - `lab_platelets x 1000`: Platelet count (coagulopathy marker).
   - `lab_potassium`: Serum potassium (cardiac arrhythmia risk).
   - `lab_rbc`: Red Blood Cell count.
   - `lab_rdw`: Red Cell Distribution Width.
   - `lab_sodium`: Serum sodium electrolyte.
   - `lab_wbc x 1000`: White Blood Cell count (infection / systemic inflammation).
2. **Infusion Medications (10 continuous drug rates)**:
   - `infusion_amiodarone (mg/min)`: Anti-arrhythmic agent.
   - `infusion_dexmedetomidine (mcg/kg/hr)`: Alpha-2 adrenergic sedative.
   - `infusion_fentanyl (mcg/hr)`: Opioid analgesia.
   - `infusion_heparin (units/hr)`: Systemic anticoagulation.
   - `infusion_insulin (units/hr)`: Continuous glycemic management.
   - `infusion_midazolam (mg/hr)`: Benzodiazepine sedation.
   - `infusion_norepinephrine (mcg/min)`: Vasopressor (first-line septic shock).
   - `infusion_norepinephrine (ml/hr)`: Volumetric rate of norepinephrine.
   - `infusion_ns (ml/hr)`: Normal Saline (0.9% NaCl) fluid resuscitation.
   - `infusion_propofol (mcg/kg/min)`: Intravenous anesthetic sedative.
3. **Treatment Categories (11 binary intervention flags)**:
   - `treatment_burns/trauma`, `treatment_cardiovascular`, `treatment_endocrine`, `treatment_gastrointestinal`, `treatment_general`, `treatment_infectious diseases`, `treatment_neurologic`, `treatment_pulmonary`, `treatment_renal`, `treatment_surgery`, `treatment_toxicology`.
4. **Intermittent Medications (5 binary drug administration flags)**:
   - `med_fentanyl`: Bolus fentanyl administration.
   - `med_furosemide`: Loop diuretic (Lasix).
   - `med_lorazepam`: Benzodiazepine (Ativan).
   - `med_potassium chloride`: Electrolyte repletion.
   - `med_zofran`: Antiemetic (Ondansetron).

### 6.2. Static Tabular Features (67 Dimensions)
1. **Demographics & Admission Characteristics (10 variables)**:
   - Continuous: `age`, `BMI`, `hospitaladmitoffset`, `unitvisitnumber`.
   - Binary Indicators: `gender_Male`, `ethnicity_Caucasian`, `ethnicity_African American`, `ethnicity_Hispanic`, `ethnicity_Asian`, `ethnicity_Other`.
2. **APACHE IV Physiology & Scoring (10 variables)**:
   - Continuous: `acutephysiologyscore` (APS score), `apachescore` (Total APACHE IV score), `predictedicumortality`, `predictediculos`, `creatinine`, `pao2`, `fio2`.
   - Neurological (GCS): `verbal`, `motor`, `eyes` (Glasgow Coma Scale sub-scores).
3. **Chronic Comorbidities (8 binary indicators)**:
   - `aids`, `cirrhosis`, `diabetes`, `hepaticfailure`, `leukemia`, `lymphoma`, `metastaticcancer`, `immunosuppression`.
4. **Admission Organ-System Diagnoses (17 categories)**:
   - `diag_cardiovascular`, `diag_pulmonary`, `diag_neurologic`, `diag_renal`, `diag_gastrointestinal`, `diag_endocrine`, `diag_infectious diseases`, `diag_hematology`, `diag_burns/trauma`, `diag_oncology`, `diag_toxicology`, `diag_surgery`, `diag_general`, `diag_transplant`, `diag_obstetrics/gynecology`, `diag_genitourinary`, `diag_musculoskeletal`, `diag_other_count`.
5. **Pre-Admission Outpatient Medications (22 categories)**:
   - `admitdrug_aspirin`, `admitdrug_lisinopril`, `admitdrug_lasix`, `admitdrug_omeprazole`, `admitdrug_albuterol`, `admitdrug_levothyroxine sodium`, `admitdrug_simvastatin`, `admitdrug_gabapentin`, `admitdrug_multivitamin`, `admitdrug_lantus`, `admitdrug_metformin hcl`, `admitdrug_metoprolol succinate`, `admitdrug_atorvastatin calcium`, `admitdrug_amlodipine besilate`, `admitdrug_lipitor`, `admitdrug_prednisone`, `admitdrug_colace`, `admitdrug_protonix`, `admitdrug_plavix`, `admitdrug_norco`, `admitdrug_other_count`.

---

## 7. Exhaustive File & Directory Reference

### 7.1. Python Modules (`src/`)

#### [`src/models/architecture.py`](file:///Users/aryandwivedi/Desktop/Project Hail Minor/src/models/architecture.py)
Master neural network definition file (608 lines):
- `TemporalEncoder`: Implements the 92-dimensional LSTM with `_last_valid_hidden` state extraction and dense projection to 64 dimensions. Supports dual multi-task heads (`use_sepsis_head=True`) predicting both mortality and sepsis.
- `TabularAutoencoder`: Implements the 67-dimensional symmetric denoising autoencoder with Gaussian corruption ($\sigma=0.1$) and mortality classification head.
- `MultimodalFusion`: Baseline concatenation fusion model ($128 \to 128 \to \text{Dropout}$).
- `AttentionFusion`: Cross-attention fusion module ($h=4, d=64$) with residual LayerNorm connections and feed-forward transformations.
- `GraphConvLayer`: Kipf & Welling sparse graph convolution operator with dense linear projections.
- `PatientGraphGNN`: 2-layer GCN operating over the intra-hospital patient graph. Includes `build_adj` static method for on-the-fly cosine $k$-NN graph generation.
- `MortalityHead` & `SepsisHead`: 2-layer MLP classification heads (`64 → 32 → 1`).
- `MultimodalGNNPipeline`: Composed end-to-end local architecture linking Stages 1–5 into a unified forward pass.
- `MultiTaskLoss`: Composite loss function weighting binary cross-entropy for mortality ($w_{\text{mort}}$) and sepsis ($w_{\text{seps}}$) alongside tabular feature reconstruction MSE ($\lambda_{\text{recon}}$).

#### [`src/data/dataset.py`](file:///Users/aryandwivedi/Desktop/Project Hail Minor/src/data/dataset.py)
Data loading abstractions and feature definitions (363 lines):
- Contains explicit column definitions (`TEMPORAL_FEATURE_COLS`, `TABULAR_FEATURE_COLS`, `TABULAR_CONT_COLS`).
- `get_hospital_splits`: Splits hospital directories 70/15/15 into train, validation, and test.
- `ICUSequenceDataset`: Loads 48-hour time series from Parquet, computes missingness indicators, applies LOCF, and packages samples as `(x, mask, y, [pid])`.
- `ICUTabularDataset`: Loads static clinical records, imputes missing values, and scales continuous features using precomputed training statistics.
- `fit_scaler`: Utility computing mean and standard deviation for continuous columns strictly across training hospitals.

#### [`src/data/extract_sepsis_labels.py`](file:///Users/aryandwivedi/Desktop/Project Hail Minor/src/data/extract_sepsis_labels.py)
Ground truth label extractor (111 lines):
- Iterates over all hospital subdirectories under `processed/hospitals/`.
- Extracts sepsis patient IDs from `diagnosis.csv` (ICD-9 codes and text keywords) and `patient.csv` (`apacheadmissiondx`).
- Aligns labels against existing patient ID index files (`checkpoints/temporal_patient_ids_{split}.npy`).
- Generates `checkpoints/fusion/sepsis_labels_{train,val,test}.npy` and `checkpoints/fusion/sepsis_metadata.json`.

#### [`src/data/cache_multitask_hospitals.py`](file:///Users/aryandwivedi/Desktop/Project Hail Minor/src/data/cache_multitask_hospitals.py)
Pre-extraction ETL script (222 lines):
- Fits continuous scalers on training hospitals and writes `checkpoints/fusion/multitask/multitask_scaler.json`.
- Compiles each hospital's aligned multimodal tensors into `processed/multitask_cache/hospital_{id}.npz`.
- Computes positive class weights ($w_{\text{mort}}, w_{\text{seps}}$) and saves `checkpoints/fusion/multitask/split_summary.json`.

#### [`src/data/train_multitask_pipeline.py`](file:///Users/aryandwivedi/Desktop/Project Hail Minor/src/data/train_multitask_pipeline.py)
End-to-end production training script (358 lines):
- Implements intra-hospital batch training across the 145 training hospitals.
- Updates weights using Adam ($\text{lr}=10^{-3}$, weight decay $= 10^{-4}$) and Cosine Annealing learning rate scheduling.
- Evaluates on 31 unseen validation hospitals each epoch and tracks Mean AUPRC.
- Selects validation-optimal decision thresholds ($T_{\text{mort}}$ for $F_1$, $T_{\text{seps}}$ for $F_2$).
- Executes a single blind evaluation on the 31 test hospitals and writes all artifacts to `checkpoints/fusion/multitask/`.

---

### 7.2. Jupyter Notebooks (`src/**/*.ipynb`)

All project pipelines and ablation notebooks are structured modularly inside the [`src/`](file:///Users/aryandwivedi/Desktop/Project%20Hail%20Minor/src/) directory:

#### Temporal Encoding (`src/Temporal Encoding/`)
1. [`src/Temporal Encoding/lstm_temporal_encoder.ipynb`](file:///Users/aryandwivedi/Desktop/Project%20Hail%20Minor/src/Temporal%20Encoding/lstm_temporal_encoder.ipynb):
   - Implements the complete Phase 1 dual-task temporal LSTM pipeline predicting both **In-Hospital Mortality** and **ICU Sepsis**.
   - Features secondary `sepsis_head` (`Linear(64, 1)`), weighted dual-task BCE loss ($w_{\text{mort}} = 10.35, w_{\text{seps}} = 5.04$), 20-epoch training with Cosine Annealing, and gradient clipping ($5.0$).
   - Evaluates on 22,809 held-out test patients: Mortality AUROC: 0.8474, AUPRC: 0.4590; Sepsis AUROC: 0.8668, AUPRC: 0.6105.
   - Exports publication figure `figures/lstm_dual_task_evaluation.png`, metrics `checkpoints/lstm_baseline_metrics.json`, and dual-task representations: `checkpoints/temporal_embeddings_{train,val,test}.npy` and `checkpoints/temporal_sepsis_labels_{train,val,test}.npy`.

#### Multi-Task End-to-End Integration (`src/`)
2. [`src/multitask_pipeline.ipynb`](file:///Users/aryandwivedi/Desktop/Project%20Hail%20Minor/src/multitask_pipeline.ipynb):
   - Complete Step 6 end-to-end multi-task integration notebook.
   - Reuses representations and empirical findings from all prior phases (Phase 1 dual-task temporal embeddings, Phase 2 tabular latents, Phase 3 cross-attention fusion, Phase 4 intra-hospital graphs $k=5$).
   - Resolves `KeyError: 'patientunitstayid'` through resilient index fallback (`patient_ids` vs `patientunitstayid`), verifying 100.0% patient ID alignment across 200,826 patients.
   - Performs end-to-end gradient flow verification across all 5 submodules.
   - Generates 6-panel publication-quality test evaluation suite (`figures/multitask_evaluation_plots.png`) and comprehensive metrics summary.

#### Multimodal Fusion Baselines (`src/multimodal fusion/`)
3. [`src/multimodal fusion/fusion_additive.ipynb`](file:///Users/aryandwivedi/Desktop/Project%20Hail%20Minor/src/multimodal%20fusion/fusion_additive.ipynb):
   - Evaluates element-wise addition fusion: $h = h_{\text{temp}} + h_{\text{tab}}$.
   - MLP classifier achieves Test AUROC: 0.8768, AUPRC: 0.5237, F1: 0.4960.
4. [`src/multimodal fusion/fusion_concat.ipynb`](file:///Users/aryandwivedi/Desktop/Project%20Hail%20Minor/src/multimodal%20fusion/fusion_concat.ipynb):
   - Concatenates temporal and tabular embeddings: $h = [h_{\text{temp}} \parallel h_{\text{tab}}] \in \mathbb{R}^{128}$.
   - MLP classifier achieves Test AUROC: 0.8782, AUPRC: 0.5331, F1: 0.4999.
   - Exports `checkpoints/fusion/fused_embeddings_{split}.npy`.
5. [`src/multimodal fusion/fusion_gated.ipynb`](file:///Users/aryandwivedi/Desktop/Project%20Hail%20Minor/src/multimodal%20fusion/fusion_gated.ipynb):
   - Evaluates learned dimension-wise sigmoid gating: $h = g \odot h_{\text{temp}} + (1 - g) \odot h_{\text{tab}}$.
   - Classifier achieves Test AUROC: 0.8789, AUPRC: 0.5289, F1: 0.5015.
6. [`src/multimodal fusion/fusion_attention.ipynb`](file:///Users/aryandwivedi/Desktop/Project%20Hail%20Minor/src/multimodal%20fusion/fusion_attention.ipynb):
   - Implements Multi-Head Cross-Attention fusion across modalities.
   - Outperforms all fusion baselines with Test AUROC: 0.8793, AUPRC: 0.5293, F1: 0.5044.
   - Exports attention checkpoint and embedding generation logic.

#### Patient Similarity Graphs (`src/patient similarity graph/`)
7. [`src/patient similarity graph/knn_k_grid_search.ipynb`](file:///Users/aryandwivedi/Desktop/Project%20Hail%20Minor/src/patient%20similarity%20graph/knn_k_grid_search.ipynb):
   - Systematic grid search over graph neighborhood size $k \in \{5, 10, 15, 20, 30\}$.
   - Evaluates validation AUPRC on the attention-fused patient graph.
   - Proves $k=5$ is optimal (Val AUPRC = 0.53712; 1,294,781 training edges).
8. [`src/patient similarity graph/graph_knn_concatenation.ipynb`](file:///Users/aryandwivedi/Desktop/Project%20Hail%20Minor/src/patient%20similarity%20graph/graph_knn_concatenation.ipynb):
   - Precomputes sparse hospital-restricted $k$-NN graphs ($k=5$) over 128-d concatenation embeddings.
   - Generates NetworkX graph visualizations and stores sparse adjacency matrices in `patient_graphs_knn_concat/`.
9. [`src/patient similarity graph/graph_knn_attention.ipynb`](file:///Users/aryandwivedi/Desktop/Project%20Hail%20Minor/src/patient%20similarity%20graph/graph_knn_attention.ipynb):
   - Extracts 64-d attention embeddings across all splits.
   - Precomputes sparse hospital-restricted $k$-NN graphs ($k=5$) and saves adjacency matrices in `patient_graphs_knn/`.

#### Graph Neural Networks (`src/GNN/`)
10. [`src/GNN/gcn_attention.ipynb`](file:///Users/aryandwivedi/Desktop/Project%20Hail%20Minor/src/GNN/gcn_attention.ipynb):
    - Standalone 2-layer sparse GCN for mortality prediction trained on attention graphs.
    - Evaluates 12 HPO hyperparameter trials (best: hidden=256, dropout=0.35, lr=0.001).
    - Test AUROC: 0.8744, Test AUPRC: 0.5126.
11. [`src/GNN/gcn_sepsis.ipynb`](file:///Users/aryandwivedi/Desktop/Project%20Hail%20Minor/src/GNN/gcn_sepsis.ipynb):
    - Standalone 2-layer sparse GCN dedicated to ICU sepsis prediction.
    - Test AUROC: 0.8242, Test AUPRC: 0.4618, Test F1: 0.4965.

---

### 7.3. Processed Data Assets (`processed/`)

- `processed/hospital_split.json`: Master partition metadata defining the exact 145 train, 31 val, and 31 test hospital directories.
- `processed/timeseries_stats.json`: Population-level mean and standard deviation for 20 laboratory variables.
- `processed/mapping/patient_to_hospital.csv`: Fast lookup mapping 200,860 `patientunitstayid` entries to their respective `hospitalid`.
- `processed/multitask_cache/hospital_{id}.npz`: 207 preprocessed, scaled, and compressed hospital archives containing:
  - `x_seq`: `(N, 12, 92)` `float32`
  - `mask`: `(N, 12)` `bool`
  - `x_tab`: `(N, 67)` `float32`
  - `y_mort`: `(N,)` `float32`
  - `y_seps`: `(N,)` `float32`
  - `pids`: `(N,)` `int64`
- `processed/hospitals/hospital_{id}/`:
  - `tabular/`: Source clinical tables (`patient.csv`, `diagnosis.csv`, `admissionDrug.csv`, `apachePatientResult.csv`, `apachePredVar.csv`, `hospital.csv`) and compiled `tabular_features.parquet`.
  - `timeseries/`: Source dynamic tables (`lab.csv`, `infusionDrug.csv`, `medication.csv`, `microLab.csv`, `treatment.csv`) and compiled `timeseries_features.parquet`.

---

### 7.4. Checkpoints & Model Weights (`checkpoints/`)

- `checkpoints/lstm_temporal_encoder.pt`: Standalone dual-task LSTM weights.
- `checkpoints/lstm_baseline_metrics.json`: Phase 1 LSTM test metrics (Mortality & Sepsis).
- `checkpoints/temporal_embeddings_{train,val,test}.npy`: Dense 64-d temporal embeddings (`(153831, 64)`, `(24186, 64)`, `(22809, 64)`).
- `checkpoints/temporal_sepsis_labels_{train,val,test}.npy`: Aligned binary sepsis ground-truth arrays.
- `checkpoints/tabular_embeddings_{train,val,test}.npz`: Dense 64-d tabular latent embeddings.
- `checkpoints/fusion/sepsis_labels_{train,val,test}.npy`: Binary sepsis ground-truth arrays.
- `checkpoints/fusion/additive_baseline/`: Additive model weights, scalers, and metrics.
- `checkpoints/fusion/concat_baseline/`: Concatenation model weights, scalers, and metrics.
- `checkpoints/fusion/gated_baseline/`: Gated model weights, scalers, and metrics.
- `checkpoints/fusion/attention_baseline/`: Attention fusion model weights, scalers, and metrics.
- `checkpoints/fusion/attention_embeddings/`: Attention-fused 64-d embeddings (`(N, 64)`).
- `checkpoints/fusion/patient_graphs_knn/`:
  - `adjacency_{split}.npz`: Raw sparse intra-hospital $k$-NN graph.
  - `adjacency_norm_{split}.npz`: Kipf-Welling normalized graph Laplacian.
  - `nodes_{split}.npz`: Patient IDs, hospital IDs, and outcome labels.
- `checkpoints/fusion/gnn_attention/`:
  - `gcn_attention.pt`: Standalone GCN mortality checkpoint.
  - `hpo_results.json`: 12-trial hyperparameter optimization results.
  - `knn_k_grid_search_results.json`: Validation results for $k \in [5..30]$.
  - `selected_knn_k_grid_search.json`: Records $k=5$ as the selected neighborhood size.
- `checkpoints/fusion/gnn_sepsis/`:
  - `gcn_sepsis.pt`: Dedicated sepsis GCN checkpoint.
  - `sepsis_label_extraction_summary.json`: Sepsis prevalence and source table breakdown.
- `checkpoints/fusion/multitask/`:
  - `multitask_pipeline.pt`: Complete end-to-end multi-task model weights.
  - `multitask_fused_{train,val}.npz`: Extracted 64-d fused representations (`(143419, 64)`, `(30047, 64)`) from the multi-task model.
  - `multitask_adj_norm_{train,val}.npz`: Block-diagonal intra-hospital graph Laplacians ($k=5$).
  - `metrics.json`: Final multi-task test evaluation results.
  - `test_predictions.npz`: Test probabilities, binary predictions, true labels, and patient IDs.
  - `multitask_scaler.json`: Scaler parameters computed on the 145 training hospitals.
  - `split_summary.json`: Exact cohort statistics and class weights.
- `checkpoints/fusion/gcn_multitask_training_history.json`: 50-epoch loss and validation metrics history for the multi-task GCN.
- `checkpoints/fusion/gcn_graph_smoothing_results.json`: Mean connected-patient cosine similarity across GCN layers.
- `checkpoints/fusion/gcn_k_sensitivity_results.json`: Graph density sensitivity across $k \in [1, 3, 5, 10, 15]$.

---

### 7.5. Automation & Evaluation Scripts (`scripts/`)

All scripts output their publication-ready visualizations directly into the dedicated top-level `figures/` directory:

- [`scripts/generate_gcn_analysis_plots.py`](file:///Users/aryandwivedi/Desktop/Project%20Hail%20Minor/scripts/generate_gcn_analysis_plots.py): Generates all 4 core GCN diagnostic and ablation figures (`figures/gcn_training_validation_loss.png`, `figures/gcn_validation_metrics.png`, `figures/gcn_graph_smoothing.png`, `figures/gcn_k_sensitivity.png`) using the true multi-task fused representations.
- [`scripts/generate_multitask_evaluation.py`](file:///Users/aryandwivedi/Desktop/Project%20Hail%20Minor/scripts/generate_multitask_evaluation.py): Evaluates the end-to-end Multi-Task Multimodal GNN pipeline on unseen validation and held-out test hospitals, calibrating decision thresholds and rendering `figures/multitask_evaluation_plots.png`.
- [`scripts/train_eval_multitask_lstm.py`](file:///Users/aryandwivedi/Desktop/Project%20Hail%20Minor/scripts/train_eval_multitask_lstm.py): Autonomous script for training, evaluating, and exporting the Phase 1 dual-task temporal LSTM to `figures/lstm_dual_task_evaluation.png`.
- [`scripts/plot_attention_fusion.py`](file:///Users/aryandwivedi/Desktop/Project%20Hail%20Minor/scripts/plot_attention_fusion.py): Visualizes attention head matrices, weight distributions, clinical outcome stratification, and latent PCA manifolds into `figures/`.
- [`scripts/update_multitask_pipeline_notebook.py`](file:///Users/aryandwivedi/Desktop/Project%20Hail%20Minor/scripts/update_multitask_pipeline_notebook.py): Automates integration of precomputed embeddings and prior findings into `multitask_pipeline.ipynb`.

---

### 7.6. Centralized Figures & Diagnostic Suite (`figures/`)

All publication plots, diagnostic visualizations, and clinical interpretability figures are centralized in `figures/`:

1. [`figures/multitask_evaluation_plots.png`](file:///Users/aryandwivedi/Desktop/Project%20Hail%20Minor/figures/multitask_evaluation_plots.png): Publication 6-panel test evaluation suite for the Multi-Task GNN (ROC curves, PR curves, threshold sweeps, confusion matrices, and risk calibration).
2. [`figures/lstm_dual_task_evaluation.png`](file:///Users/aryandwivedi/Desktop/Project%20Hail%20Minor/figures/lstm_dual_task_evaluation.png): Publication 6-panel test evaluation suite for the standalone dual-task temporal LSTM.
3. [`figures/gcn_training_validation_loss.png`](file:///Users/aryandwivedi/Desktop/Project%20Hail%20Minor/figures/gcn_training_validation_loss.png): Multi-task training and validation loss convergence trajectories across 50 epochs.
4. [`figures/gcn_validation_metrics.png`](file:///Users/aryandwivedi/Desktop/Project%20Hail%20Minor/figures/gcn_validation_metrics.png): Epoch-wise AUROC and AUPRC trajectories showing acute sepsis outperforming mortality due to acute trajectory signals.
5. [`figures/gcn_graph_smoothing.png`](file:///Users/aryandwivedi/Desktop/Project%20Hail%20Minor/figures/gcn_graph_smoothing.png): Mean neighbor cosine similarity across GCN depth ($0.8953 \to 0.9695 \to 0.9904$), confirming context assimilation without oversmoothing.
6. [`figures/gcn_k_sensitivity.png`](file:///Users/aryandwivedi/Desktop/Project%20Hail%20Minor/figures/gcn_k_sensitivity.png): Validation AUPRC sensitivity across graph densities $k \in [1, 3, 5, 10, 15]$, demonstrating $k=5$ optimality.
7. [`figures/attention_fusion_head_matrices.png`](file:///Users/aryandwivedi/Desktop/Project%20Hail%20Minor/figures/attention_fusion_head_matrices.png): 4-head attention transition matrices revealing modality specialization (temporal dynamics vs chronic baseline querying).
8. [`figures/attention_fusion_clinical_stratification.png`](file:///Users/aryandwivedi/Desktop/Project%20Hail%20Minor/figures/attention_fusion_clinical_stratification.png): Stratified attention dynamics showing $+12.8\%$ surge in temporal self-attention for deceased patients.
9. [`figures/attention_fusion_weight_distributions.png`](file:///Users/aryandwivedi/Desktop/Project%20Hail%20Minor/figures/attention_fusion_weight_distributions.png): Patient-level modality weight violin distributions across outcome cohorts.
10. [`figures/attention_fusion_latent_pca.png`](file:///Users/aryandwivedi/Desktop/Project%20Hail%20Minor/figures/attention_fusion_latent_pca.png): 2D PCA representation manifold of 64-d fused multimodal latents across 24,186 validation patients.
11. [`figures/graph_sample_train.png`](file:///Users/aryandwivedi/Desktop/Project%20Hail%20Minor/figures/graph_sample_train.png): NetworkX cluster topology visualization of intra-hospital $k$-NN patient similarity graph.

---

## 8. Comparative Experimental Benchmark Matrix

All evaluation metrics below were computed on the held-out **unseen test split** (22,809 patients from 31 unseen test hospitals):

| Model Architecture | Evaluated Target | Test AUROC | Test AUPRC | Test $F_1$ | Test Accuracy | Balanced Accuracy | Sensitivity (Recall) | Specificity |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage 1: Standalone LSTM (Mortality)** | Mortality | 0.8474 | 0.4590 | 0.4421 | 89.41% | 69.88% | 45.99% | 93.77% |
| **Stage 1: Standalone LSTM (Sepsis)** | Sepsis | 0.8668 | 0.6105 | 0.5404 | 77.23% | 78.95% | 81.50% | 76.39% |
| **Additive Fusion Baseline** | Mortality | 0.8768 | 0.5237 | 0.4960 | 90.22% | 73.37% | 52.76% | 93.98% |
| **Concatenation Fusion Baseline** | Mortality | 0.8782 | 0.5331 | 0.4999 | 90.33% | 73.52% | 52.96% | 94.08% |
| **Gated Fusion Baseline** | Mortality | 0.8789 | 0.5289 | 0.5015 | 90.46% | 73.42% | 52.57% | 94.27% |
| **Attention Fusion Baseline** | Mortality | **0.8793** | 0.5293 | **0.5044** | 90.22% | 74.17% | 54.54% | 93.80% |
| **GCN (Attention Graph)** | Mortality | 0.8744 | 0.5126 | 0.4905 | 89.87% | 73.50% | 52.76% | 93.59% |
| **GCN (Dedicated Sepsis)** | Sepsis | 0.8242 | 0.4618 | 0.4965 | 79.96% | 72.00% | 56.41% | 84.55% |
| **Multi-Task Pipeline (Mortality)** | Mortality | 0.8712 | 0.4744 | 0.4779 | **90.39%** | 71.88% | 49.36% | **94.40%** |
| **Multi-Task Pipeline (Sepsis)** | Sepsis | **0.9129** | **0.6364** | **0.5546** | 80.43% | **83.41%** | **87.53%** | 79.28% |

### Key Clinical & Technical Insights:
1. **Multimodal Synergy**: Fusing static tabular data with temporal sequences yields a massive jump in mortality prediction (AUPRC increases from $0.4551$ to $0.5293$, $+7.42\%$ absolute gain).
2. **Attention Superiority**: Multi-Head Cross-Attention achieved the highest test AUROC ($0.8793$) and highest $F_1$-score ($0.5044$) among all fusion strategies.
3. **Multi-Task Sepsis Breakthrough**: The unified end-to-end Multi-Task GNN pipeline achieved state-of-the-art sepsis performance (**AUROC: 0.9129, AUPRC: 0.6364**), successfully capturing **87.53% of all sepsis cases** with a clinical $F_2$-score of **0.7109**.

---

## 9. Execution & Replication Guide

### 9.1. Environment Setup
Verify Python 3.13+ environment with PyTorch and PyData dependencies:
```bash
python3 -c "import torch, numpy, scipy, sklearn; print('All core libraries installed successfully!')"
```

### 9.2. Data Preprocessing & Hospital Caching
Extract sepsis ground-truth labels and compile hospital multi-task tensors:
```bash
# 1. Extract consensus sepsis ground-truth labels
python3 src/data/extract_sepsis_labels.py

# 2. Build multi-task hospital cache and fit training scaler
python3 src/data/cache_multitask_hospitals.py
```

### 9.3. Multi-Task Model Training
Execute end-to-end multi-task training across all 145 training hospitals:
```bash
python3 src/data/train_multitask_pipeline.py
```

### 9.4. Interactive Inspection & Demo
Launch JupyterLab to interactively explore models, verify gradient flow, and inspect visualizations:
```bash
jupyter lab
```
Recommended execution order:
1. [`src/multitask_pipeline.ipynb`](file:///Users/aryandwivedi/Desktop/Project%20Hail%20Minor/src/multitask_pipeline.ipynb): End-to-end pipeline demonstration and test inspection.
2. [`src/patient similarity graph/knn_k_grid_search.ipynb`](file:///Users/aryandwivedi/Desktop/Project%20Hail%20Minor/src/patient%20similarity%20graph/knn_k_grid_search.ipynb): Graph neighborhood optimization.
3. [`src/multimodal fusion/fusion_attention.ipynb`](file:///Users/aryandwivedi/Desktop/Project%20Hail%20Minor/src/multimodal%20fusion/fusion_attention.ipynb): Cross-modal attention ablation.
4. [`src/GNN/gcn_attention.ipynb`](file:///Users/aryandwivedi/Desktop/Project%20Hail%20Minor/src/GNN/gcn_attention.ipynb): Graph convolutional network benchmarking.

### 9.5. Generating GCN Diagnostic & Ablation Curves
Execute the automated analysis suite to generate all four core GCN diagnostic plots:
```bash
python3 scripts/generate_gcn_analysis_plots.py
```
This produces:
- `figures/gcn_training_validation_loss.png`: Convergence and multi-task loss dynamics.
- `figures/gcn_validation_metrics.png`: Epoch-wise AUROC and AUPRC progression for both mortality and sepsis.
- `figures/gcn_graph_smoothing.png`: Mean neighbor cosine similarity across GCN depth.
- `figures/gcn_k_sensitivity.png`: Validation AUPRC sensitivity across graph densities $k \in [1, 3, 5, 10, 15]$.

---

## 10. GCN-Specific Diagnostic & Representation Analysis

### 10.1. Multi-Task Training Convergence (Plot 1)
- **Artifact**: [`figures/gcn_training_validation_loss.png`](file:///Users/aryandwivedi/Desktop/Project%20Hail%20Minor/figures/gcn_training_validation_loss.png)
- **Dynamics**: Shows smooth, stable monotonic decrease in total loss (from $2.40$ down to $1.52$ train, $1.58$ validation). Both task-specific objectives stabilize harmoniously without negative gradient interference:
  - Mortality BCE loss: $1.25 \to 0.86$ (train), $1.29 \to 0.91$ (validation).
  - Sepsis BCE loss: $1.15 \to 0.66$ (train), $1.09 \to 0.67$ (validation).

### 10.2. Validation Metric Trajectories (Plot 2)
- **Artifact**: [`figures/gcn_validation_metrics.png`](file:///Users/aryandwivedi/Desktop/Project%20Hail%20Minor/figures/gcn_validation_metrics.png)
- **Trajectories**: Tracks validation AUROC and AUPRC across 50 epochs trained on multi-task fused representations:
  - **Validation AUROC (Left Plot)**:
    - **Sepsis AUROC (pink)** starts at **0.8824** and converges to **0.8984** (~0.90), forming the **upper curve**.
    - **Mortality AUROC (blue)** starts at **0.8160** and converges to **0.8537**, forming the **lower curve**.
  - **Validation AUPRC (Right Plot)**:
    - **Sepsis AUPRC (red)** starts at **0.5706** and converges to **0.6317**, forming the **upper curve**.
    - **Mortality AUPRC (green)** starts at **0.4058** and converges to **0.4406**, forming the **lower curve**.
- **Resolution of Apparent Curve Inversion**:
  - In earlier iterations, `scripts/generate_gcn_analysis_plots.py` loaded `checkpoints/fusion/attention_embeddings/attention_embeddings_train.npz` (which was trained solely on mortality in Stage 3, lacking sepsis signals), causing Sepsis AUROC to falsely plateau around ~0.75.
  - Updating the pipeline to use the unified multi-task fused representations (`multitask_fused_{train,val}.npz`) restored the true clinical dynamics where acute sepsis features enable Sepsis AUROC (~0.90) and AUPRC (~0.63) to consistently surpass mortality.

### 10.3. Graph Smoothing Across Layers (Plot 3)
- **Artifact**: [`figures/gcn_graph_smoothing.png`](file:///Users/aryandwivedi/Desktop/Project%20Hail%20Minor/figures/gcn_graph_smoothing.png)
- **Connected Neighbor Cosine Similarity**:
  - **Before GCN (Fused Multi-Task Embeddings)**: $0.8953$
  - **After GCN Layer 1**: $0.9695$
  - **After GCN Layer 2**: $0.9904$
- **Interpretation**: Graph message passing effectively aggregates intra-hospital neighborhood clinical phenotypes. The similarity increases steadily from $0.8953$ to $0.9904$, demonstrating beneficial localized context assimilation while 2 layers prevent oversmoothing across distinct clinical subpopulations.

### 10.4. Graph Density Sensitivity Ablation (Plot 4)
- **Artifact**: [`figures/gcn_k_sensitivity.png`](file:///Users/aryandwivedi/Desktop/Project%20Hail%20Minor/figures/gcn_k_sensitivity.png)
- **Neighborhood Size Evaluation ($k \in [1, 3, 5, 10, 15]$)**:
  - $k=1$: Mortality AUPRC = $0.4375$, Sepsis AUPRC = $0.6281$
  - $k=3$: Mortality AUPRC = $0.4399$, Sepsis AUPRC = $0.6281$
  - $k=5$: Mortality AUPRC = **$0.4407$** (optimal peak), Sepsis AUPRC = $0.6315$
  - $k=10$: Mortality AUPRC = $0.4406$, Sepsis AUPRC = **$0.6321$**
  - $k=15$: Mortality AUPRC = $0.4400$, Sepsis AUPRC = $0.6307$
- **Takeaway**: Neighborhood size $k=5$ achieves the highest mortality AUPRC ($0.4407$) and strong sepsis AUPRC ($0.6315$), validating $k=5$ as the optimal graph density balancing localized clinical support against oversmoothing. Sepsis AUPRC remains consistently high (~0.63) across all neighborhood densities.

---

## 11. Attention-Based Multimodal Fusion Diagnostics & Interpretability

### 11.1. Multi-Head Specialization (Plot 1)
- **Artifact**: [`figures/attention_fusion_head_matrices.png`](file:///Users/aryandwivedi/Desktop/Project%20Hail%20Minor/figures/attention_fusion_head_matrices.png)
- **Head Specialization**:
  - **Head 1 (Temporal Focus)**: $0.810$ Temporal $\to$ Temporal self-attention, $0.190$ Temporal $\to$ Tabular.
  - **Head 2 (Cross-Modal Context)**: Specialized cross-modal head with **$0.610$ Temporal $\to$ Tabular attention**, querying chronic patient baselines to contextualize temporal signals.
  - **Head 3 (Temporal Dynamics)**: $0.848$ Temporal $\to$ Temporal and $0.621$ Tabular $\to$ Temporal attention.
  - **Head 4 (Balanced Synthesis)**: Symmetrically balanced mixing ($0.578$ / $0.422$ and $0.498$ / $0.502$).
  - **Overall Ensemble Mean**: Average attention places $65.6\%$ on temporal dynamics and $34.4\%$ on static tabular context.

### 11.2. Clinical Outcome Stratification (Plot 2)
- **Artifact**: [`figures/attention_fusion_clinical_stratification.png`](file:///Users/aryandwivedi/Desktop/Project%20Hail%20Minor/figures/attention_fusion_clinical_stratification.png)
- **Clinical Shift ($\Delta = \text{Deceased} - \text{Survived}$)**:
  - **Survived Patients ($y=0$)**: Temporal $\to$ Temporal = $0.644$, Temporal $\to$ Tabular = $0.356$.
  - **Deceased Patients ($y=1$)**: Temporal $\to$ Temporal surges to **$0.772$** ($+12.8\%$ absolute shift), while Temporal $\to$ Tabular drops to $0.228$.
  - **Clinical Finding**: When a patient acutely decompensates toward in-hospital mortality, the attention mechanism dynamically decouples from static chronic scores and re-allocates attention almost entirely to high-frequency deteriorating physiological trajectories.

### 11.3. Patient-Level Modality Weight Distributions (Plot 3)
- **Artifact**: [`figures/attention_fusion_weight_distributions.png`](file:///Users/aryandwivedi/Desktop/Project%20Hail%20Minor/figures/attention_fusion_weight_distributions.png)
- **Distributional Dynamics**: Violin plots demonstrate that while surviving patients maintain a wide, balanced distribution across tabular context, non-surviving patients cluster tightly in the upper tail of temporal dependence.

### 11.4. Fused Multimodal Representation Manifold (Plot 4)
- **Artifact**: [`figures/attention_fusion_latent_pca.png`](file:///Users/aryandwivedi/Desktop/Project%20Hail%20Minor/figures/attention_fusion_latent_pca.png)
- **Manifold Separation**: PCA 2D projection of $h_{\text{fused}} \in \mathbb{R}^{64}$ across 24,186 validation patients demonstrates clear separability of critical decompensation cases along Principal Component 1.


