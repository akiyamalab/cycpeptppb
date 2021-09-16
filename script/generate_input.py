import cut_ring
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors



def __calc_rdkit_descriptors(mols):
    """
    calculate rdkit descriptors of substructures
    """
    [Chem.SanitizeMol(mol) for mol in mols]
    descriptors_ = []
    for mol in mols:
        tmp_ = []
        # version 1
        # tmp_.append(Descriptors.MolLogP(mol))
        # tmp_.append(Descriptors.PEOE_VSA6(mol))
        # tmp_.append(-Descriptors.MolMR(mol))
        
        # version 2
        tmp_.append(Descriptors.PEOE_VSA6(mol))
        tmp_.append(Descriptors.MolLogP(mol))
        tmp_.append(Descriptors.EState_VSA3(mol))
        descriptors_.append(tmp_)
    # list 
    return descriptors_



def __standardization_descriptors(all_descriptors):
    """
    standardization by Zscored
    """
    # parameters for standardization
    # v1
    # ave_MolLogP   = 0.2178
    # std_MolLogP   = 1.1260
    # ave_PEOE_VSA6 = 11.309
    # std_PEOE_VSA6 = 12.597
    # ave_MolMR     = -43.049
    # std_MolMR     = 14.313
    
    # v2
    ave_PEOE_VSA6   = 11.308605186884161
    std_PEOE_VSA6   = 12.597012892251113

    ave_MolLogP     = 0.2177629032770479
    std_MolLogP     = 1.1259929837682099

    ave_EState_VSA3 = 4.407030042420127
    std_EState_VSA3 = 4.222676051519849

    ave_list = [ave_PEOE_VSA6, ave_MolLogP, ave_EState_VSA3]
    std_list = [std_PEOE_VSA6, std_MolLogP, std_EState_VSA3]
    
    standardization_descriptors_ = []

    for i in range(len(all_descriptors)):
        descriptors_ = np.array(all_descriptors[i])
        for j, ave, std in zip([0,1,2], ave_list, std_list):
            descriptors_[:, j] = (descriptors_[:, j] - ave) / std
        standardization_descriptors_.append(descriptors_.tolist())
    # list
    return standardization_descriptors_



def __calc_substructure_descriptors(substructures_list):
    """
    Zscored
    """
    descriptors_ = []
    for mols in substructures_list:
        descriptors_.append(__calc_rdkit_descriptors(mols))
    descriptors_ = __standardization_descriptors(descriptors_)
    # list
    return descriptors_



def __make_up_feature_map(descriptors_, feature_num, max_len):
    """
    generate 1DCNN input
    """
    feature_map = np.array([])
    for descriptor in descriptors_:
        # feature_num * max_len
        map_tmp = np.array(descriptor).T
        padding_tmp = np.zeros([feature_num, max_len-len(descriptor)])
        map_tmp = np.concatenate([map_tmp, padding_tmp], axis=1)
        if len(feature_map) == 0:
            feature_map = np.expand_dims(map_tmp, 0)
        else:
            feature_map = np.concatenate([feature_map, np.expand_dims(map_tmp, 0)], 0)
    # array
    return feature_map



def __make_number_table(substructures_nums, max_len):
    """
    Create a table of substructures and their corresponding numbering.
    Numbers correspond to substructures of each peptides; comparisons between different peptides are not possible.
    """
    number_table = []

    for nums in substructures_nums:
        tmp_ = [i for i in range(nums)]
        padding_ = [-1 for j in range(max_len-nums)]
        number_table.append(tmp_+padding_)
    # list
    return number_table



def __perform_feature_map_augmentation(feature_map, substructures_nums, max_len):
    """
    Do augmentation of feature_map
    """
    aug_array = np.array([])
    aug_nums = []
    aug_index = []

    for i in range(len(feature_map)):
        num_now = substructures_nums[i]
        map_now = feature_map[i][:,:num_now]
        # rotation
        for j in range(num_now):
            map_now = np.hstack([map_now[:, 1:], map_now[:, 0].reshape(-1, 1)])
            # translation
            for k in range((max_len-num_now)+1):
                pad_start = np.zeros([feature_map.shape[1], k])
                pad_end = np.zeros([feature_map.shape[1], max_len-num_now-k])

                tmp_ = map_now
                tmp_ = np.concatenate([pad_start, tmp_], axis=1)
                tmp_ = np.concatenate([tmp_, pad_end], axis=1)  

                if len(aug_array) == 0:
                    aug_array = tmp_
                else:
                    aug_array = np.vstack([aug_array, tmp_])

                aug_nums.append(num_now)
                aug_index.append(i)

    aug_array = aug_array.reshape(-1, feature_map.shape[1]* max_len)
    aug_array = aug_array.reshape(-1, feature_map.shape[1], max_len)  
    # array, list, list
    return aug_array, aug_nums, aug_index



def __perform_number_table_augmentation(number_table, substructures_nums, max_len):
    """
    Do augmentation of number_table
    """
    aug_table = []

    for i in range(len(number_table)):
        num_now = substructures_nums[i]
        table_now = number_table[i][:num_now]
        # rotation
        for j in range(num_now):
            table_now = table_now[1:] + [table_now[0]]
            # translation
            for k in range((max_len-num_now)+1):
                pad_start = [-1 for l in range(k)]
                pad_end = [-1 for l in range(max_len-num_now-k)]

                tmp_ = table_now
                tmp_ = pad_start + tmp_
                tmp_ = tmp_ + pad_end
                aug_table.append(tmp_)
    # list
    return aug_table



def __make_up_CNN_input(smiles_list, feature_num=3, max_len=15, use_augmentation=True):
    """
    Receive input smiles and perform substructure partitioning using cut_ring.cut_cyclic_peptide_to_amino_acids.
    The partial features are then calculated by rdkit to generate the input feature map for the CNN model.
    This function will resturn:
    CNN model input (array), number of substructures in input data (aug_nums), index of input data (aug_index), 
    table of correspondence numbers of substructures (aug_table), substructure smiles (substructures_smiles), 
    substructure mapping atom list (mapping_list), peptide for drawing (signed_mol_list)
    """
    # canonical smiles
    pep_mols = [Chem.MolFromSmiles(mol) for mol in smiles_list]
    pep_smiles  = [Chem.MolToSmiles(mol) for mol in pep_mols]
    pep_mols = [Chem.MolFromSmiles(mol) for mol in pep_smiles]


    # divide peptide into substructures
    substructures_list = []
    mapping_list = []
    signed_mol_list = []
    for mol in pep_mols:
        substructures_list.append(cut_ring.cut_cyclic_peptide_to_amino_acids(mol)[0])
        mapping_list.append(cut_ring.cut_cyclic_peptide_to_amino_acids(mol)[1])          
        signed_mol_list.append(cut_ring.cut_cyclic_peptide_to_amino_acids(mol)[2])    


    # substructure smiles
    substructures_smiles = []
    for mols in substructures_list:
        substructures_smiles.append([Chem.MolToSmiles(mol) for mol in mols])
    # number of substructure  
    substructures_nums = [len(mols) for mols in substructures_list]
    # calculate rdkit descriptors
    descriptors_ = __calc_substructure_descriptors(substructures_list)
    # make up 1DCNN input feature map
    feature_map = __make_up_feature_map(descriptors_, feature_num=feature_num, max_len=max_len)
    # make up 1DCNN input correspondence number table
    number_table = __make_number_table(substructures_nums, max_len)

    
    # augmentation
    if use_augmentation:
    # sample
    #     feature_map = np.array([[[1,2,3,4,5,6,7,8,9,0,0,0,0,0,0],
    #                              [1,2,3,4,5,6,7,8,9,0,0,0,0,0,0],
    #                              [1,2,3,4,5,6,7,8,9,0,0,0,0,0,0]]])
        aug_maps, aug_nums, aug_index  = __perform_feature_map_augmentation(feature_map, substructures_nums, max_len=max_len)
        aug_table = __perform_number_table_augmentation(number_table, substructures_nums, max_len)
    else:
        aug_maps = feature_map
        aug_nums = substructures_nums 
        aug_index = [i for i in range(len(pep_mols))]
        aug_table = number_table

    return np.array(aug_maps).astype('float32'), aug_nums, aug_index, np.array(aug_table), substructures_smiles, mapping_list, signed_mol_list





def count_number_of_residues(smiles):
    """
    count residues number
    """
    # canonical smiles
    pep_mols = [Chem.MolFromSmiles(mol) for mol in [smiles]]
    pep_smiles  = [Chem.MolToSmiles(mol) for mol in pep_mols]
    pep_mols = [Chem.MolFromSmiles(mol) for mol in pep_smiles]

    # divide peptide into substructures
    substructures_list = []
    for mol in pep_mols:
        substructures_list.append(cut_ring.cut_cyclic_peptide_to_amino_acids(mol)[0])

    return len(substructures_list[0])



