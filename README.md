# CycPeptPPB

The official implementation of the **CycPeptPPB**.  
**CycPeptPPB** is a predictor of Plasma Protein Binding rate for cyclic peptide with high performance focusing on
residue-level features and circularity.

<img width="881" alt="スクリーンショット 2021-09-17 4 59 32" src="https://user-images.githubusercontent.com/44156441/133677917-80eda706-e5cd-462c-baea-150911556ede.png">

# Requirements
- Python:  3.7.9
- RDKit:  2020.03.1
- Chainer:  7.1.0

# Code
- **EXAMPLE.ipynb** 
  > Jupyter notebook with an example of prediction (trained weight is required).
- **cut_ring.py** 
  > Divide the main chain of the cyclic peptide into substructures.  
  > The target bonds of division are amide bonds and disulfide bonds.
- **generate_input.py** 
  > Generate prediction model input feature map.  
  > In the paper, we used descriptors computed from MOE software, but since MOE is a commercial software, CycPeptPPB implementation on this site used descriptors computed by RDKit.  
  > MOE descriptors used for the model in the paper: logP(o/w), PEOE_VSA-1, logS.   
  > RDKit descriptors used in the CycPeptPPB implementation on this site: MolLogP, PEOE_VSA6, EState_VSA3.  
- **generate_model.py**
  > Generate prediction model.  
  > You need to add the trained weights file of the model such as "model_weight/model.npz".
- **draw_saliency_2Dmol.py**
  > Draw a 2D molecular heatmap of Saliency Score.  
  > This function is only feasible when CyclicConv is not used (Baseline model & CycPeptPPB model 2).
- **get_output.py**
  > Make a prediction.  
  > You can change the variables ***use_augmentation***(=True) and ***use_CyclicConv***(=False) to specify the model to use.
  > + *use_augmentation*=False, *use_CyclicConv*=False → Baseline model (1DCNN)  
  > + *use_augmentation*=False, *use_CyclicConv*=True  → CycPeptPPB model 1 (CyclicConv)
  > + *use_augmentation*=True, *use_CyclicConv*=False  → CycPeptPPB model 2 (Augmentated 1DCNN)
  > + *use_augmentation*=True, *use_CyclicConv*=True   → CycPeptPPB model 3 (Augmentated CyclicConv)



# Pretrained weights
- Pretrained weights are not available.

# Prediction performance
- Prediction accuracy of external test data (DrugBank dataset):
- MOE descriptors version we used in the paper:
  - Baseline model (1DCNN): MAE=6.55, R=0.89.
  - CycPeptPPB model 1 (CyclicConv): MAE=15.60, R=0.66.
  - CycPeptPPB model 2 (Augmentated 1DCNN): **MAE=4.79, R=0.92.**
  - CycPeptPPB model 3 (Augmentated CyclicConv): MAE=8.97, R=0.87.

# Contact
- Jianan Li: li@bi.c.titech.ac.jp
