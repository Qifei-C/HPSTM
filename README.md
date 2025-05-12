# Human Pose Smoothing with Transformer and Manifold Constraints (HPSTM)

## Overview

This repository contains a PyTorch implementation of a pose refinement model designed to take a sequence of noisy 3D human joint positions and output a denoised, temporally smooth, and physically plausible 3D pose sequence. The model leverages a Transformer-based temporal encoder to capture long-range dependencies within a sliding window of frames and employs manifold constraints by representing poses as joint rotations with fixed bone lengths. A differentiable Forward Kinematics (FK) decoder reconstructs joint positions from this latent representation, ensuring anatomically consistent outputs. This approach is inspired by the concepts presented in SmoothNet and ManiPose.

The core goal is to refine noisy 3D pose data (e.g., from 2D-to-3D pose estimators) by learning a low-dimensional manifold of valid human poses and applying temporal smoothing.

## Relation to the Genuine-ESFP Project

This manifold-based pose smoothing model is an integral component of the **Genuine-ESFP (Expressive Speech-driven Full-body Pose)** project. 

The Genuine-ESFP pipeline focuses on generating expressive and coherent full-body human animations directly from speech inputs. This repository provides a vital post-processing or refinement stage within ESFP, ensuring that the generated poses are not only temporally smooth but also adhere to human kinematic constraints and physical plausibility by leveraging manifold learning.

For more information on the complete Genuine-ESFP system and how this module fits into the larger architecture, please refer to the main project repository:
* **Genuine-ESFP:** [https://github.com/Qifei-C/Genuine-ESFP](https://github.com/Qifei-C/Genuine-ESFP)


## Key Features & Concepts

* **Sliding Window Temporal Encoder:** Utilizes a Transformer encoder to process a sliding window of input frames, enabling the model to learn temporal smoothness and long-range dependencies, effectively reducing jitter while preserving motion dynamics. [cite: 1, 5, 6, 7]
* **Manifold-Constrained Pose Representation:** Predicts joint rotations and bone lengths, defining poses on a human skeleton manifold. This ensures anatomically consistent bone lengths and joint connectivity, avoiding physically impossible poses. [cite: 2, 3, 4]
* **Differentiable Forward Kinematics (FK) Decoder:** A differentiable FK module reconstructs 3D joint positions from the predicted rotations and bone lengths. The human skeleton is modeled as a kinematic chain, and this process allows for end-to-end training. [cite: 3, 9, 10]
* **Modular PyTorch Implementation:** The codebase is structured модульно, with clear separation for dataset handling, skeleton definition, FK, the main pose model, and loss functions. [cite: 5]
* **Training Strategy:** Trained on noisy pose sequences with ground-truth targets, using losses such as per-joint position error and bone-length consistency error. [cite: 11, 12, 13]
* **Inference Pipeline:** Applies the trained model in a sliding-window fashion to new noisy sequences to produce smoothed outputs. [cite: 15, 16]

## Dataset Recommendations

* **AMASS (Archive of Motion Capture as Surface Shapes):** Recommended for its diversity and high-quality 3D poses. [cite: 18]
    * **CMU Mocap Subset:** Particularly well-suited due to its wide range of movements and numerous subjects, helping the model learn a broad pose manifold. [cite: 18, 19, 20]
* **Human3.6M:** Can be used as a validation set, though it is smaller in scope compared to the CMU subset of AMASS. [cite: 21, 22]
* **Data Preparation:**
    * Extract 3D joint coordinates. [cite: 23]
    * Ensure a common joint indexing and skeleton definition. [cite: 24]
    * Compute the skeleton's rest pose and bone connectivity for FK. [cite: 25]
    * Prepare noisy inputs by using outputs from 2D-to-3D estimators or by adding synthetic noise to ground-truth sequences. [cite: 26]

## Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/Qifei-C/HPSTM.git](https://github.com/Qifei-C/HPSTM.git)
    cd HPSTM
    ```
2.  Create and activate a Python virtual environment (e.g., using conda or venv).
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *(You will need to create a `requirements.txt` file listing packages like PyTorch, NumPy, etc.)*

## Usage

### Skeleton Definition

The model requires a skeleton definition, including parent joint indices and rest pose directions for each bone. An example 17-joint skeleton is provided in the documentation, but you should adapt this to your dataset's skeleton (e.g., from SMPL if using AMASS). [cite: 49, 50, 51, 52, 53, 58]

### Training

1.  **Prepare Data:**
    * Use the `src/datasets/preprocess_amass.py` script (or similar) to process your chosen dataset (e.g., AMASS CMU subset) into sequences of 3D joint coordinates. [cite: 154]
    * Ensure data is split into training and validation sets. [cite: 27]
    * Configure the `PoseDataset` in `src/datasets/pose_sequence_dataset.py` with your data paths, window size, and noise parameters (if using synthetic noise). [cite: 29, 41, 130]
2.  **Configure Model & Training:**
    * Adjust hyperparameters in the training script (e.g., `scripts/train.py`), such as `window_size`, `batch_size`, `learning_rate`, model dimensions (`d_model`, `nhead`, `num_layers`). [cite: 94, 130, 157, 158]
    * The training script initializes the `PoseModel`, optimizer (Adam with a learning rate scheduler like `ReduceLROnPlateau` is suggested), and loss functions (position MSE loss and bone length loss). [cite: 32, 34, 117, 118, 119, 127, 130]
3.  **Run Training:**
    ```bash
    python scripts/train.py
    ```
    * Monitor training and validation losses. Model checkpoints are typically saved based on validation performance. [cite: 132, 133, 135, 137, 138]

### Inference

1.  **Load Trained Model:**
    * Initialize the `PoseModel` with the same architecture used for training.
    * Load the saved model weights (e.g., `pose_model_best.pth`). [cite: 148]
2.  **Prepare Input Sequence:**
    * The input should be a NumPy array of shape `(T, J*3)` representing the noisy pose sequence (T frames, J joints with 3 coordinates each). [cite: 147]
3.  **Run Smoothing:**
    * Use the `smooth_sequence` function provided in the inference script (e.g., `scripts/infer.py` or as outlined in the documentation [cite: 146]) which applies the model in a sliding-window manner.
    ```python
    # Example usage within a script
    # model = PoseModel(...)
    # model.load_state_dict(torch.load("path/to/your/model.pth"))
    # model.eval()
    # noisy_pose_sequence = ... # Load your noisy sequence
    # refined_sequence = smooth_sequence(model, noisy_pose_sequence, window_size)
    ```
    * The output will be the refined 3D pose sequence. [cite: 148, 165]

## Implementation Details

The project is implemented in PyTorch with a modular structure:

* **`src/datasets/`**: Contains dataset loading (`PoseDataset` [cite: 29]) and preprocessing utilities.
* **`src/kinematics/`**: Includes Forward Kinematics (`ForwardKinematics` [cite: 31, 61]) and skeleton definitions.
* **`src/losses/`**: Defines loss functions like position MSE loss and bone length consistency loss. [cite: 34, 124, 125]
* **`src/models/`**: Contains the main `PoseModel` [cite: 32, 83, 94] integrating the Transformer encoder and manifold decoder.
* **`scripts/`**: Provides scripts for training, inference, and data preprocessing.

Key components include:
* **`PoseDataset`**: Handles loading sequences and yielding sliding window samples. [cite: 29, 41]
* **`ForwardKinematics`**: Converts predicted rotations and bone lengths to 3D joint positions. [cite: 31, 61, 68]
* **`PoseModel`**:
    * Input projection layer. [cite: 84, 95]
    * Learnable positional embeddings for the Transformer. [cite: 86, 96]
    * Transformer encoder. [cite: 85]
    * Output heads for rotations and bone lengths. [cite: 87, 88]
    * Uses the `ForwardKinematics` module for the final pose reconstruction. [cite: 89]

## References

* [1] Zeng et al. (2022). *SmoothNet: A Plug-and-Play Network for Refining Human Poses in Videos*. [cite: 171] ([Link to paper](https://www.researchgate.net/publication/365037480_SmoothNet_A_Plug-and-Play_Network_for_Refining_Human_Poses_in_Videos)[cite: 175], [ECCV version](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136650615.pdf) [cite: 176])
* [2] Rommel et al. (2024). *ManiPose: Manifold-Constrained Multi-Hypothesis 3D Human Pose Estimation*. [cite: 172] ([Link to paper](https://ar5iv.org/pdf/2312.06386) [cite: 175])
* [3] Various authors on Differentiable Forward Kinematics for human skeletons. [cite: 173]
* [4] AMASS Dataset: [https://amass.is.tue.mpg.de/](https://amass.is.tue.mpg.de/) [cite: 176]
* [5] Human3.6M Dataset: [http://vision.imar.ro/human3.6m/](http://vision.imar.ro/human3.6m/) [cite: 176]


