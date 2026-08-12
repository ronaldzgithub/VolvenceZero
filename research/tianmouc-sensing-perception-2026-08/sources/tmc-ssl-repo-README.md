# Tianmouc: Brain-Inspired Sensing-Perception Learning Framework

This repository contains the implementation of the brain-inspired two-stage learning framework presented in our paper: **"Brain-inspired Visual Sensing-Perception for Open-World Environments"**.

The framework leverages the **Tianmouc** sensor's complementary pathways to achieve robust visual perception in extreme open-world scenarios (High Dynamic Range, High-Speed motion, etc.) through a **Bottom-Up** representation learning stage and a **Top-Down** perception modulation stage.

***

## 🏗️ Framework Overview

> [!NOTE]
> **Author's Note**: Currently, the author is busy with work, and the environment setup for some sub-tasks might be non-trivial. However, the core projects primarily depend only on **PyTorch** and **tianmoucv**. If you encounter any issues, please feel free to leave a message/issue; I will get back to you when I have time.

The project is structured according to the hierarchical logic of the paper:

### 1. Bottom-Up Stage: Internal Model Building

In this stage, structured visual priors are embedded into **Intermediate Representations (IRs)**. We train an internal model, **IGFNet (Information-Guided Fusion Network)**, in a self-supervised manner to estimate **Visual Ground Truth (e-VGT)** by fusing complementary cues.

- **Core Implementation**: [reconstruction/](https://github.com/Tianmouc/TMC-SSL-Representation/blob/main/reconstruction)
  - **IRs**: $F\_I$ (Integral), $F\_O$ (Optical flow/warping), $F\_{HDR}$ (Poisson fusion).
  - **Model**: IGFNet with Integrated Gated Fusion and memory module.
  - **Tasks**: Self-supervised VGT estimation, high-speed reconstruction, and HDR synthesis.
- **Dataset (Tianmouc-R)**: [dataset\_raw/](https://github.com/Tianmouc/TMC-SSL-Representation/blob/main/dataset_raw) - Real-world extreme scenarios.

### 2. Top-Down Stage: Robust Perception Modulation

The visual knowledge distilled from the internal model adaptively modulates downstream perception algorithms, enhancing robustness and generalization.

- **Monocular Depth Estimation (MDE)**: [task\_depth/](https://github.com/Tianmouc/TMC-SSL-Representation/blob/main/task_depth)
  - **Model**: **IGF-BSNet** (IGFNet Encoder + BSNet Decoder).
  - **Dataset (Tianmouc-MDE)**: [dataset\_depth/](https://github.com/Tianmouc/TMC-SSL-Representation/blob/main/dataset_depth) - Pseudo-labels generated via DAM-V2-L on e-VGT.
- **Video Instance Segmentation (VIS)**: [task\_seg\_v5/](https://github.com/Tianmouc/TMC-SSL-Representation/blob/main/task_seg_v5)
  - **Model**: **YOLO-CVS** (Complementary Vision Sensor) & **EOLO** Baseline.
  - **Dataset (Tianmouc-VIS)**: [dataset\_vis/](https://github.com/Tianmouc/TMC-SSL-Representation/blob/main/dataset_vis) - High-frame-rate pixel-level annotations.

### 3. Simulation Tools

- **Modern Simulator**: [tianmoucv.sim](https://github.com/Tianmouc/tianmoucv_preview) (Recommended) - Install via `pip install tianmoucv`.
- **Legacy Simulators**: [tmcsim/](https://github.com/Tianmouc/TMC-SSL-Representation/blob/main/tmcsim) and [tmcsim\_gpu/](https://github.com/Tianmouc/TMC-SSL-Representation/blob/main/tmcsim_gpu) (First-generation versions).

***

## 🛠️ Installation & Environment

We recommend using the provided Anaconda environment backup for reproducing the results.

### Environment Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Tianmouc/TMC-SSL-Representation.git
   cd TMC-SSL-Representation
   ```
2. **Create Conda Environment**:
   We provide a reference environment file: [environment\_pytorch.yaml](https://github.com/Tianmouc/TMC-SSL-Representation/blob/main/environment_pytorch.yaml).
   ```bash
   conda env create -f environment_pytorch.yaml
   conda activate pytorch
   ```
3. **Install tianmoucv**:
   ```bash
   pip install tianmoucv
   ```

*Note: The environment for depth dataset generation is excluded as it involves specific heavy dependencies (Mask2Former, MobileSAM).*

***

## 📂 Project Structure Highlights

```text
.
├── reconstruction/       # Bottom-Up: IGFNet & VGT Estimation
├── task_depth/           # Top-Down: Monocular Depth Estimation
├── task_seg_v5/          # Top-Down: Video Instance Segmentation
├── dataset_raw/          # Raw Data Readers & Tianmouc-R
├── dataset_depth/        # Depth Annotation & Tianmouc-MDE
├── dataset_vis/          # VIS Annotation & Tianmouc-VIS
├── tmcsim/               # Legacy Simulator (CPU)
├── tmcsim_gpu/           # Legacy Simulator (GPU)
└── environment_pytorch.yaml # Conda Environment Backup
```

***

## 📜 Citation

If you find this work useful in your research, please cite:

```bibtex
@article{lin2026brain,
  title={Brain-inspired visual sensing--perception for open-world environments},
  author={Lin, Yihan and Chen, Yuguo and Meng, Yapeng and Chen, Xiangru and Wang, Taoyi and Zhao, Rong and Shi, Luping},
  journal={Nature Sensors},
  pages={1--17},
  year={2026},
  publisher={Nature Publishing Group UK London}
}
```

