# Top-Down Stage: Monocular Depth Estimation (IGF-BSNet)

This sub-project implements the **Top-Down Stage** for Monocular Depth Estimation (MDE) using **Tianmouc** sensor data. It leverages the visual knowledge distilled from the Bottom-Up pre-trained IGFNet encoder to achieve robust depth perception.

## 🚀 Training Scripts
All training and evaluation scripts are located in the [scripts](https://github.com/Tianmouc/TMC-SSL-Representation/blob/main/task_depth/scripts) directory.

### Core Hybrid Model
- **Proposed Hybrid (IGFNet Encoder + BSNet Decoder)**: [train_tm_bsnet.sh](https://github.com/Tianmouc/TMC-SSL-Representation/blob/main/task_depth/scripts/train_tm_bsnet.sh)  
  This experiment utilizes the encoder from our **IGFNet** (Integrated Gated Fusion Network) paired with the decoder from **BSNet** to leverage both high-temporal-resolution features and robust depth reconstruction.

### Baselines & Comparisons
The following scripts represent baseline models and comparative experiments:
- **Original BSNet**: `train_original_bsnet.sh` - The standard BSNet implementation.
- **TM UFormer**: `train_tm_uformer.sh` - A Transformer-based depth estimation baseline.
- **TM UNet**: `train_tm_unet.sh` - A standard UNet-based depth estimation baseline.
- **Ablation Studies**: `train_ablation.sh` - Experiments for component-wise analysis.
- **Variants**: `train_tm_bsnet_L.sh` - A larger variant of the hybrid model.

## 🛠 Configuration
Dataset paths and hyperparameters for depth tasks are managed via YAML files in the [config](https://github.com/Tianmouc/TMC-SSL-Representation/blob/main/task_depth/config) directory.
- Primary Config: [dataset_tianmouc_d_20241220_config.yaml](https://github.com/Tianmouc/TMC-SSL-Representation/blob/main/task_depth/config/dataset_tianmouc_d_20241220_config.yaml)

## 📦 Pre-trained Weights

You can download the pre-trained weights for the Monocular Depth Estimation (MDE) tasks from the following link:
- **Shared Content (MDE)**: [MDE Sharing link](https://ug.link/dh4300plus-0b13/filemgr/share-download/?id=458c92fe305642b790cc7773af75cc6b) (Password: `ambp`)

**Note**: After downloading the weights, please create a new directory named `ckpts_fixed` in this folder (`task_depth/`) and place all downloaded weight files inside it. Then, update the YAML configuration files in the [config](https://github.com/Tianmouc/TMC-SSL-Representation/blob/main/task_depth/config) directory with your actual `ckpts` storage location and `dataset` paths.

## 📂 Directory Structure
- **depth_model/**: Contains various depth estimation model implementations (BSNet, MiDaS, ZoeDepth, etc.).
- **bsnet_raw/**: Standard implementation of BSNet.
- **NeWCRFs-master/**: Implementation of the NewCRFs depth estimation method.
- **scripts/**: Shell scripts for training and evaluation.
- **utils/**: Utility functions for training, loss calculation, and depth-specific metrics.

---
For evaluation, use [evaluate.py](https://github.com/Tianmouc/TMC-SSL-Representation/blob/main/task_depth/evaluate.py) or the `evaluate.sh` script in the `scripts/` folder.
