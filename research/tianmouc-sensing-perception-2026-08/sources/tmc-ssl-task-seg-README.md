# Top-Down Stage: Instance Segmentation Training & Evaluation

This sub-project implements the **Top-Down Stage** instance segmentation tasks, including our proposed **YOLO-CVS** model and the **EOLO** baseline, optimized for **Tianmouc** sensor data.

## 📦 Pre-trained Weights

You can download the pre-trained weights for the segmentation models from the following link:
- **Shared Content (SEGmodel)**: [SEGmodel Sharing link](https://ug.link/dh4300plus-0b13/filemgr/share-download/?id=f73587b2c8404dd7b2c11b3ec8bb37fc) (Password: `JYVR`)

**Note**: After downloading the weights, please create a new directory named `ckpts_fixed` in this folder (`task_seg_v5/`) and place all downloaded weight files inside it. Then, update the configuration files in the `config/` or `data/` directory with your actual `ckpts` storage location and `dataset` paths.

## 🚀 Training Scripts

All training and evaluation scripts are located in the [scripts](https://github.com/Tianmouc/TMC-SSL-Representation/blob/main/task_seg_v5/scripts) directory.

### Quick Start

#### Training with Simulation Data
```bash
bash scripts/sim_seg_coco_S.sh
```

#### Training with Mixed Data
```bash
bash scripts/mix_train_seg_S.sh
```

#### Model Evaluation
```bash
bash scripts/eval_real_all.sh
```

## 📂 Directory Structure

- **`scripts/`**: Root contains primary training scripts (S/M/L versions).
- **`scripts/v3/`**: Contains improved v3 version scripts and experimental configurations.
- **`segment/`**: Core implementation for segmentation tasks.
- **`detect/`**: Core implementation for detection tasks.
- **`models/`**: Model configuration and architecture definitions.
- **`utils/`**: Utility functions for training, loss calculation, and metrics.

---
For detailed naming conventions and utility scripts, please refer to the [scripts/README.md](https://github.com/Tianmouc/TMC-SSL-Representation/blob/main/task_seg_v5/scripts/README.md).
