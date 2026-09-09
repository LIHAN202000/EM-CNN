# EM-CNN

This repository provides the code associated with the data-driven electron tomography framework described in the paper below. The workflow includes synthetic training-data generation, model training, sparse-view electron tomography reconstruction, defocus-corrected atomic electron tomography reconstruction, and iterative refinement. 
![image](https://github.com/LIHAN202000/EM-CNN/blob/main/Fig2_09.jpg)
## Reference

Li H, Cui W, Lei H, et al. A general data-driven framework for scalable electron tomography[J]. *National Science Review*, 2026, 13(14): nwag365.

## Requirements

- pyyaml
- numpy
- opencv-python
- torch
- scikit-image

## Generate a New Dataset

```bash
cd ./0_Dataset/SV
python run.py
```

When the projection angles, accelerating voltage, pixel size, convergence angle, or other relevant acquisition/imaging parameters differ from those used for the existing training data, it is recommended to regenerate the dataset with the corresponding parameters and retrain the model.

## Train a New Model

```bash
cd ./1_DL_model
python train.py --name "SV6" --config "config/SV6.yaml"
```

## Reconstruction

### 1. Sparse-View ET

```bash
cd ./1_DL_model
python IRM.py
python predict_SV.py
```

### 2. Defocus-Corrected AET

```bash
cd ./1_DL_model
python IRM.py
python predict_DF.py
```

## Atomic Structure Refinement

```bash
cd ./2_Iterative_refinement
python run.py
```
## The code for tomographic data processing:
[AET-AmorphousMaterials/Supplementary-Data-Codes](https://github.com/AET-AmorphousMaterials/Supplementary-Data-Codes)

## Inspired by


Lee, J., Jeong, C. & Yang, Y. Single-atom level determination of 3-dimensional surface atomic structure via neural network-assisted atomic electron tomography. *Nature Communications* **12**, 1962 (2021).

Mao L, Cui J, Yu R. Local-orbital tomography with depth-dependent interactions. *Physical Review B*, 2025, 111(6): 064116.


