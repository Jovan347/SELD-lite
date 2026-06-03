# SELD-lite: Lightweight Sound Event Localization and Detection

This repository contains a lightweight **Sound Event Localization and Detection (SELD)** system based on the DCASE SELD baseline code. The project focuses on reducing the size and computational cost of a SELD model while preserving as much performance as possible.

The main work is implemented in the **DCASE 2022 student model** folder. A larger teacher model is used as a reference, while the student model is trained as a compressed SELDnet-style CRNN using a combination of supervised learning, optional knowledge distillation, pruning, and dynamic quantization.

## Project Overview

Sound Event Localization and Detection is the task of detecting which sound events are active and estimating where they are located in space. A SELD system therefore combines two subtasks:

- **Sound Event Detection (SED):** identifies active sound classes over time.
- **Direction of Arrival (DOA) estimation:** predicts the spatial direction of the detected sounds.

This project investigates a smaller SELD model for the DCASE-style multi-ACCDOA task. The main goal is not to create the highest-scoring model, but to study how much the original SELDnet-style architecture can be reduced while still producing meaningful SELD predictions.

## Main Contributions

The updated project includes:

- a reduced **student CRNN** architecture,
- a larger **teacher CRNN** architecture for comparison and distillation,
- multi-ACCDOA output support,
- fine-tuning from an initialized student checkpoint,
- optional output-level knowledge distillation from the teacher model,
- optional global pruning,
- optional dynamic quantization,
- parameter and FLOP counting scripts,
- DCASE-style validation and test evaluation.

## Repository Structure

```text
.
├── repos/
│   ├── seld-dcase2021/             # Reference DCASE 2021 SELD code
│   ├── seld-dcase2022/             # Reference DCASE 2022 SELD code
│   ├── seld-dcase2022-teacher/     # Larger teacher model setup
│   └── seld-dcase2022-student/     # Main lightweight student implementation
│
├── data/                           # Not committed: dataset files
├── runs/                           # Not committed: extracted features / labels
├── results/                        # Not committed: DCASE-format predictions
├── models/                         # Not committed: trained checkpoints
└── README.md
```

The most important folder for the actual SELD-lite work is:

```text
repos/seld-dcase2022-student/
```

## Model Design

### Teacher Model

The teacher model is the larger SELDnet-style CRNN used as a reference for knowledge distillation. Its configuration uses larger hidden dimensions than the student model.

Typical teacher configuration:

```text
CNN filters: 64
GRU size:    128
FNN size:    128
Output:      multi-ACCDOA
```

The teacher checkpoint is used only to generate teacher predictions during distillation. Its weights are not directly transferred into the student model.

### Student Model

The student model is a reduced SELDnet-style CRNN. It keeps the general CNN-GRU-FNN structure of the original architecture, but uses smaller layer sizes.

Main student configuration:

```text
CNN filters: 32
GRU size:    64
FNN size:    64
Output:      multi-ACCDOA
```

In the current setup, the student model has approximately:

```text
Parameters: 156,332
FLOPs:      101,017,600
Output:     (batch_size, 50, 108)
```

The output dimension corresponds to the multi-ACCDOA format, where the model predicts activity and localization information for multiple tracks and sound classes.

## Knowledge Distillation

The student model can be trained with knowledge distillation using a larger pretrained teacher. During training, the student is optimized with two objectives:

1. the standard supervised SELD loss against the ground-truth labels,
2. an additional MSE distillation loss between the student output and teacher output.

The combined loss is:

```text
loss = (1 - kd_alpha) * hard_loss + kd_alpha * kd_loss
```

where:

- `hard_loss` is the normal supervised training loss,
- `kd_loss` is the MSE loss between student and teacher predictions,
- `kd_alpha` controls the balance between ground-truth learning and teacher imitation.

In the student parameter configuration, this is controlled by:

```python
distill = True
kd_alpha = 0.5
teacher_model_path = "models/3_1_dev_split0_multiaccdoa_foa_model.h5"
```

## Initialized Student Checkpoint

The initialized student checkpoint is used as the starting point for training the reduced student model. Since this checkpoint already matches the reduced student architecture, the model can continue training from learned student weights instead of starting completely randomly.

During the knowledge distillation experiment, this initialized student is then further optimized using the standard supervised objective together with the additional MSE distillation loss from the teacher predictions.

## Compression Methods

The student setup includes optional compression methods.

### Pruning

Global pruning can be applied to selected layers in the student model. In the current parameter setup, pruning is enabled with:

```python
do_prune = True
prune_amount = 0.3
```

This removes part of the model weights according to the pruning strategy used in the training script.

### Dynamic Quantization

Dynamic quantization can also be applied after training, mainly targeting layers where quantization is supported. In the current setup, it is controlled by:

```python
apply_dynamic_quant = True
dynamic_quant = True
quant_dtype = "qint8"
```

When dynamic quantization is enabled, evaluation is moved to CPU because the quantized model runs in the CPU-compatible inference path.

## Dataset

The project uses a DCASE-style SELD dataset with first-order Ambisonic features.

The dataset itself is not included in this repository because the raw audio, features, labels, checkpoints, and outputs are too large for GitHub. They should be placed locally in folders such as:

```text
data/
runs/
models/
results/
```

These folders are intentionally ignored by Git.

The student setup is configured for:

```text
Dataset format: FOA
Task format:    multi-ACCDOA
Classes:        12
```

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/SELD-lite.git
cd SELD-lite
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```

Install the required packages. The exact versions may depend on the local DCASE setup, but the main dependencies are:

```bash
pip install torch numpy scipy librosa matplotlib pandas scikit-learn h5py
```

If using CUDA, install the PyTorch version that matches the local CUDA setup.

## Running the Student Model

Move into the student implementation folder:

```bash
cd repos/seld-dcase2022-student
```

Run the student configuration:

```bash
python train_seldnet.py 8 1
```

Here:

- `8` selects the FOA multi-ACCDOA student model configuration from `parameters.py`,
- `1` is the job ID used in the output filenames.

The script trains the student model, evaluates it on the validation split, saves the best model, reloads it, optionally applies quantization, and evaluates on the unseen test split.

## Running the Teacher Model

Move into the teacher folder:

```bash
cd repos/seld-dcase2022-teacher
```

Run the teacher configuration:

```bash
python train_seldnet.py 3 1
```

The resulting teacher checkpoint can then be used by the student model for knowledge distillation.

## Utility Scripts

The student folder also includes small utility scripts for measuring model complexity.

Count parameters:

```bash
python count_params.py
```

Compute FLOPs:

```bash
python compute_flops.py
```

These scripts are useful for reporting the size and computational cost of the lightweight student model.

## Evaluation Metrics

The project follows the DCASE SELD evaluation style. The main reported metrics are:

- **SELD score**: combined early-stopping/evaluation metric; lower is better.
- **SED Error Rate (ER)**: lower is better.
- **SED F-score (F)**: higher is better.
- **DOA Localization Error (LE)**: lower is better.
- **DOA Localization Recall (LR)**: higher is better.

The evaluation script also reports classwise results for the macro setting.

## Example Results

In one student evaluation run, the lightweight model produced approximately:

```text
SELD score:             0.65
SED Error Rate:         0.77
SED F-score:            17.1
DOA Localization Error: 53.4°
DOA Localization Recall:29.5%
```

These results show that the reduced model can perform the full SELD task, but also that compression and distillation involve a trade-off between efficiency and accuracy.

## Important Notes

The repository contains several reference versions of the SELD baseline code. The main project work is in:

```text
repos/seld-dcase2022-student/
```

The local dataset paths inside `parameters.py` may need to be changed before running the code on another computer. Replace local absolute paths with paths that match the local machine, for example:

```python
params['dataset_dir'] = "../../data"
params['feat_label_dir'] = "../../runs/seld_feat_label_2022"
```

The trained models are not included in the repository. To reproduce the knowledge distillation setup, the teacher checkpoint and initialized student checkpoint must be placed in the expected `models/` folder or the paths must be updated in `parameters.py`.

## Acknowledgements

This project builds on the DCASE SELD baseline code and adapts it for a lightweight student-model experiment using multi-ACCDOA, pruning, quantization, and knowledge distillation.

## License

This repository contains code derived from external DCASE SELD baseline implementations. Check the original licenses and citation requirements of the included baseline repositories before reusing or redistributing the code.
