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
- **generate_input.py** 
  > Generate prediction model input feature map.  
  > In the paper, we used descriptors computed from MOE software, but since MOE is a commercial software, CycPeptPPB implementation on this site used descriptors computed by RDKit.
  > MOE descriptors used for the model in the paper: logP(o/w), PEOE_VSA-1, logS.
  > RDKit descriptors used in the CycPeptPPB implementation on this site: MolLogP, PEOE_VSA6, EState_VSA3.
- **generate_model.py**
  > Generate prediction model.  
  > You need to add the trained weights file of the model as "model_weight/model.npz".
- **draw_saliency_2Dmol.py**
  > Draw a heatmap for Salience Score.
- **get_output.py**
  > Make a prediction.

# Pretrained weights
- Pretrained weights are not available.
