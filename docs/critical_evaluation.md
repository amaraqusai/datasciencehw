# Critical Evaluation — Belarbi et al. (2022) on CICIDS2017

**Evidence sources:** `Final_Project_Notebook.ipynb` metrics, `report/Final_Project_Report.tex`.

---

## Claim evaluation matrix

| Claim | Evidence | Verdict | Notes |
|-------|----------|---------|-------|
| **C1** — DBN achieves state-of-the-art multi-class performance | Author baseline: DBN achieves high F1 across multiple classes. Our pipeline: MLP (deep baseline) F1 0.9987, AUC 0.9999. | **Supported** | Deep architectures are highly capable of isolating complex, non-linear attack features in the CICIDS2017 dataset. |
| **C2** — Deep architectures significantly outperform traditional machine learning on network flows | Our pipeline: MLP F1 0.9987 vs Random Forest F1 1.0000. | **Partial** | While DBN/MLP perform excellently, our ensemble tree-based baseline (Random Forest) matched or slightly exceeded performance with significantly lower training cost on tabular data. |
| **C3** — Feature scaling and redundancy removal are critical | Scaling flow duration vs packet length enables rapid convergence. Removing constant features dropped irrelevant dimensionality. | **Supported** | Unscaled inputs with massive magnitude differences (e.g. bytes/sec vs count) completely distort neural network gradients. |

---

## Methodology strengths
1. **Public replication package** — Authors provided PyTorch code and clear model hyperparameter configurations for the Deep Belief Network.
2. **Standard Evaluation Metrics** — Reporting of Precision, Recall, and F1 to deal with severe class imbalances inherent in cybersecurity.

## Methodology weaknesses
1. **Dataset footprint** — The 18GB raw CICIDS2017 data makes direct, end-to-end continuous reproduction burdensome.
2. **Over-complexity** — Employing stacked RBMs (DBN) for a structured tabular dataset introduces heavy computational overhead when decision trees (Random Forest) yield competitive metrics. 
