# Bottom-Up Stage: Self-Supervised Reconstruction (IGFNet)

This sub-project implements the **Bottom-Up Stage** of our framework, focusing on the self-supervised reconstruction of high-frame-rate video from **Tianmouc** sensor data. It aims to build an internal model, **IGFNet**, to estimate Visual Ground Truth (e-VGT).

## 🚀 Training Scripts
All training configurations and execution scripts are located in the [scripts](https://github.com/Tianmouc/TMC-SSL-Representation/blob/main/reconstruction/scripts) directory.

### Core Models
- **IGFNet**: `train_unet_mem.sh`  
  The primary proposed model featuring an Integrated Gated Fusion and memory module for temporal consistency.
  **Note**: Training IGFNet requires pre-trained weights from **$F_O$** and **$F_I$**. Ensure you complete the training of RAFT ($F_O$) and TinyUNet ($F_I$) first.
- **Ablation Study (No mem)**: `train_unet_no_mem.sh`  
  A version of the model with the memory module removed to evaluate its impact on reconstruction quality.
- **$F_O$ & $F_I$ Components**:  
  - **RAFT**: `train_raft.sh` - Used for optical flow estimation ($F_O$) as described in the paper.
  - **TinyUNet**: `train_tinyunet.sh` - Used for efficient image reconstruction/interpolation ($F_I$).

### Method Visualization
- **TD2VID**: `train_TD2VID.sh`  
  An additional model used specifically for the visualizations and data flow demonstrations in **Method Figure 1**.

### Baselines & Comparisons
The following scripts are used for comparative experiments against existing state-of-the-art methods and baseline architectures:
- `train_baseline_cbmnet.sh`
- `train_baseline_lfnet.sh`
- `train_baseline_lfnet_mono.sh`
- `train_baseline_swinIR.sh`
- `train_baseline_uformer.sh`
- `train_baseline_unet_nature.sh`
- `train_spynet.sh`

## 🛠 Configuration
Training parameters, hyperparameters, and dataset paths are managed via YAML files in the [config](https://github.com/Tianmouc/TMC-SSL-Representation/blob/main/reconstruction/config) directory. 
- Example: [dataset_tianmouc_r_20241220_config.yaml](https://github.com/Tianmouc/TMC-SSL-Representation/blob/main/reconstruction/config/dataset_tianmouc_r_20241220_config.yaml)

## 📦 Pre-trained Weights & Evaluation Models

You can download the pre-trained weights from the following links:
- **Google Drive**: [eVGT Weights](https://drive.google.com/drive/folders/1gtnhQmP8_IS6VWQzVhWGYC62CP4yFkfU?usp=share_link)
- **Shared Content (eVGT)**: [eVGT Sharing link](https://ug.link/dh4300plus-0b13/filemgr/share-download/?id=e5bfdf2878f14916a54a9bbec1c1c6bc) (Password: `zJBx`)

For evaluation, you can use the evaluation detection model:
- **Shared Content (Evaluation)**: [Evaluation Sharing link](https://ug.link/dh4300plus-0b13/filemgr/share-download/?id=bd3e2e11073347d295c74a8358805f1e) (Password: `Xayg`)

**Note**: After downloading the weights, please create a new directory named `ckpts_fixed` in this folder (`reconstruction/`) and place all downloaded weight files inside it. Then, update the YAML configuration files in the [config](https://github.com/Tianmouc/TMC-SSL-Representation/blob/main/reconstruction/config) directory with your actual `ckpts` storage location and `dataset` paths.

## 📂 Directory Structure
- **model/**: Contains the implementation of various network architectures, including IGFNet, RAFT, SwinIR, and UFormer.
- **scripts/**: Shell scripts for batch training and evaluation tasks.
- **utils/**: Core helper functions for inference, loss functions, data augmentation, and distributed training.

---
For evaluation, please use [evaluate.py](https://github.com/Tianmouc/TMC-SSL-Representation/blob/main/reconstruction/evaluate.py) or the corresponding scripts in the `scripts/` folder.
