# CycPeptPPB

The official implementation of the CycPeptPPB

<img width="400" alt="スクリーンショット 2021-09-16 4 46 08" src="https://user-images.githubusercontent.com/44156441/133499447-01b83422-20f2-4ce3-846c-2577f7ce5d47.png">
<img width="400" alt="スクリーンショット 2021-09-16 4 47 07" src="https://user-images.githubusercontent.com/44156441/133499567-7307e375-0f48-42ac-b9af-740c30bc1748.png">


# Code
- **EXAMPLE.ipynb** 
  > jupyter notebook with an example of prediction (trained weight required)
- **cut_ring.py** divide the main chain of the cyclic peptide into substructures
- **generate_input.py** generate prediction model input feature map
  - In the paper, we used descriptors computed from MOE software, but since MOE is a commercial software, CycPeptPPB used descriptors computed by rdkit
