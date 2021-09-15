# CycPeptPPB

The official implementation of the CycPeptPPB

<img width="400" alt="スクリーンショット 2021-09-16 4 46 08" src="https://user-images.githubusercontent.com/44156441/133499447-01b83422-20f2-4ce3-846c-2577f7ce5d47.png">
<img width="400" alt="スクリーンショット 2021-09-16 4 47 07" src="https://user-images.githubusercontent.com/44156441/133499567-7307e375-0f48-42ac-b9af-740c30bc1748.png">

# Requirements
- python:  3.7.9
- rdkit:  2020.03.1
- chainer:  7.1.0

# Code
- **EXAMPLE.ipynb** 
  > Jupyter notebook with an example of prediction (trained weight is required).
- **cut_ring.py** 
  > Divide the main chain of the cyclic peptide into substructures.
- **generate_input.py** 
  > Generate prediction model input feature map.  
  > In the paper, we used descriptors computed from MOE software, but since MOE is a commercial software, CycPeptPPB used descriptors computed by rdkit.
- **generate_model.py**
  > Generate prediction model.  
  > You need to add the trained weights file of the model as "model_weight/model.npz".
- **draw_saliency_2Dmol.py**
  > Draw a heatmap for Salience Score.
- **get_output.py**
  > Make a prediction.
