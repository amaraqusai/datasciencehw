# Third-Party Source Code Documentation

The deep learning architectures implemented in this repository are adapted from the original research paper's repository. In the interest of academic integrity and transparency, this document explicitly outlines the origin, licensing, and modifications applied to these files.

## Copied & Adapted Files
The following files were originally authored by Belarbi et al. (2022) and have been integrated into this repository:
1.  `models/RBM.py` (Restricted Boltzmann Machine)
2.  `models/DBN.py` (Deep Belief Network)
3.  `models/MLP.py` (Multi-Layer Perceptron)

## Original Source & License
*   **Original Author:** Othmane Belarbi
*   **Source Repository:** [othmbela/dbn-based-nids](https://github.com/othmbela/dbn-based-nids)
*   **Associated Paper:** "An Intrusion Detection System based on Deep Belief Networks" (Science of Cyber Security, 2022).
*   **License:** Apache 2.0 (Retained for all downstream modifications).

## Modifications Made
While the fundamental forward-pass and unsupervised mathematical properties of the architectures were strictly preserved to guarantee a fair reproduction, critical engineering modifications were necessary to execute the pipeline locally:

1. **Scikit-Learn Pipeline Wrapping (`src/model_utils.py`)**
   * **Change:** The raw PyTorch `nn.Module` classes were encapsulated inside a `PyTorchDBNWrapper` and `MappedXGBClassifier`.
   * **Why it was necessary:** To adhere to the strict data-leakage constraints of our reproduction, the models had to be compatible with `sklearn.pipeline.Pipeline` and `StratifiedKFold`. Sklearn requires custom estimators to explicitly implement `self.classes_` and `self.fitted_` attributes.
   * **Behavior Preserved:** Yes. The underlying gradient descent, contrastive divergence pre-training, and epoch looping logic remain completely identical to the original paper.

2. **Dtype Conversions & Tensor Safety**
   * **Change:** Explicitly forcing `.float()` and `.long()` type-casting when feeding Pandas DataFrame matrices into the PyTorch `DataLoader`.
   * **Why it was necessary:** The original codebase relied on outdated NumPy/Pandas array mappings that cause fatal exceptions (`RuntimeError: expected scalar type Long but found Float`) on modern PyTorch 2.x versions.
   * **Behavior Preserved:** Yes. It only affects memory typing, not mathematical computations.

3. **Label Encoding Isolation**
   * **Change:** The target labels are now encoded *outside* the PyTorch class via a `LabelEncoder` managed by the broader pipeline, rather than hardcoded string maps inside the model.
   * **Why it was necessary:** CICIDS2017 contains 15 distinct classes. Hardcoding labels causes catastrophic failure when subsets of data (e.g., Stratified Cross-Validation folds) are missing a specific minority attack class. 

## Summary
These files **are not newly developed code by the current student**. They are structurally identical to the original authors' work, mathematically preserving the Contrastive Divergence (CD-k) algorithms. All engineering modifications were solely aimed at upgrading the codebase to run on modern frameworks and to seamlessly integrate into our mathematically rigorous, zero-leakage cross-validation evaluation loop.
