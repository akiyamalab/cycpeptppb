from rdkit import Chem
from rdkit.Chem import AllChem, Draw



def __run_all_reaction(mol, flag, sign_num=None, ring_sele=None, n_cap_num=77, c_cap_num=66):
    """
    react as much as possible.
    """
    products = []
    Chem.SanitizeMol(mol)
    mols = [mol]
    
    ##################################################
    if   flag == 'sign_amide':
        rxn = AllChem.ReactionFromSmarts(f"[!1C;!2C;!3C;!4C;!5C;!6C;!7C;!8C;!9C;!10C;!11C;!12C;!13C;!14C;!15C;!16C;!17C;!18C;!19C;!20C;!21C;!22C;!23C;!24C;!25C;!26C;!27C;!28C;!29C;!30C;!31C;C;{ring_sele}:1](=[O:2])[N:3] >> [{sign_num}C:1](=[O:2])[N:3]")
    elif flag == 'cut_amide':
        if c_cap_num == 66:            
            rxn = AllChem.ReactionFromSmarts(f"[N:1][{sign_num}C:2](=[O:3])>>[N:1][{n_cap_num}*].[C:2](=[O:3])")
        elif c_cap_num == 55:
            rxn = AllChem.ReactionFromSmarts(f"[N:1][{sign_num}C:2](=[O:3])>>[N:1][{n_cap_num}*].[{c_cap_num}*][C:2](=[O:3])")
    elif flag == 'cut_disulfide':
        rxn = AllChem.ReactionFromSmarts(f"[*:1][C:2]([N:3])[C:4][{sign_num}S:5][{sign_num+100}S:6][C:7][C:8]([C:9](=[O:10]))[*:11] >> [*:1][C:2]([N:3])[C:4][S:5].[S:6][C:7][C:8]([C:9](=[O:10]))[*:11]")
    ##################################################
    
    intermediates = rxn.RunReactants(mols)

    if len(intermediates) == 0:
        products.append(mol)
        return products
    
    intermediates = intermediates[0]
    for m in intermediates:
        products += __run_all_reaction(m, flag, sign_num+1, ring_sele, n_cap_num, c_cap_num)
    
    return products



def __sign_C0_in_ring(mol, ring_size_min=15, ring_size_max=1000):
    """
    Recognizes the ring and labels the carbon atoms of the amide bond contained in the ring with range(1 to (number of residues + 1))
    """
    ring_sele = ",".join([f"r{i}" for i in range(ring_size_min, ring_size_max+1)])
    Chem.SanitizeMol(mol)
    return __run_all_reaction(mol, 'sign_amide', sign_num=1,  ring_sele=ring_sele)[0]



def __sign_disulfide_bonds(mol, sign_num=100):
    """
    Sulfur atoms in disulfide bonds are labeled with sign_num, sign_num+100
    """
    rxn = AllChem.ReactionFromSmarts(f"[*:1][C:2]([N:3])[C:4][S:5][S:6][C:7][C:8]([C:9](=[O:10]))[*:11] >> [*:1][C:2]([N:3])[C:4][{sign_num}S:5][{sign_num+100}S:6][C:7][C:8]([C:9](=[O:10]))[*:11]")
    Chem.SanitizeMol(mol)
    mols = [mol]
    if len(rxn.RunReactants(mols)) != 0:
        mols = rxn.RunReactants(mols)[0]
    return mols[0]



def __open_ring_by_cutting_signed_amide_bonds(mol, n_cap_num=77, c_cap_num=66):
    """
    Ring opening is performed by cleaving the amide bond containing the 1C carbon atom.
    """
    if c_cap_num == 66:
        rxn = AllChem.ReactionFromSmarts(f"([N:1][{1}C:2](=[O:3]))>>([N:1][{n_cap_num}*].[C:2](=[O:3]))")
    elif c_cap_num == 55:
        rxn = AllChem.ReactionFromSmarts(f"([N:1][{1}C:2](=[O:3]))>>([N:1][{n_cap_num}*].[{c_cap_num}*][C:2](=[O:3]))")
    Chem.SanitizeMol(mol)
    return rxn.RunReactants([mol])[0][0]



def __cut_signed_amide_bonds(mol, n_cap_num=77, c_cap_num=66):
    """
    Cleaves all amide bonds.
    """
    Chem.SanitizeMol(mol)
    return __run_all_reaction(mol, 'cut_amide', sign_num=2, n_cap_num=n_cap_num, c_cap_num=c_cap_num)



def __cut_signed_disulfide_bonds(mol, sign_num=100):
    """
    Cleaves the disulfide bond labeled with sign_num.
    """
    Chem.SanitizeMol(mol)
    return __run_all_reaction(mol, 'cut_disulfide', sign_num=sign_num)



def __get_mapping_atoms(signed_mol, mol, n_cap_num=77, c_cap_num=66):
    """
    Get the index of the atom to which the substructure corresponds.
    """
    Chem.SanitizeMol(mol)
    mols = [mol]
    if n_cap_num == 77:
        rxn = AllChem.ReactionFromSmarts(f"[N:1][{n_cap_num}*] >> [N:1]")
        if len(rxn.RunReactants(mols)) != 0:
            mols = rxn.RunReactants(mols)[0]
    if c_cap_num != 66:
        rxn = AllChem.ReactionFromSmarts(f"[C:1][{c_cap_num}*] >> [C:1]")
        if len(rxn.RunReactants(mols)) != 0:
            mols = rxn.RunReactants(mols)[0]
            
    mapping_list = signed_mol.GetSubstructMatches(mols[0])
    if len(mapping_list) == 0:
        print('Mapping error')
        return [0]
    else:
        return mapping_list[0]



def __replace_dummy_atom(mol, dummy_num, moiety):
    """
    Replace the dummy atom with another atom cluster.
    Specify moiety so that the location of the dummy atom is the first atom.
    """
    rxn = AllChem.ReactionFromSmarts(f"[{dummy_num}*:1]>>{moiety}")
    Chem.SanitizeMol(mol)
    mols = [mol]
    if len(rxn.RunReactants(mols)) != 0:
        mols = rxn.RunReactants(mols)[0]
    return mols[0]



def __remove_amide_sign(mol):
    """
    Remove all amide bond labels.
    """
    Chem.SanitizeMol(mol)
    mols = [mol]
    for i in range(1, 21):
        substruct_tmp = Chem.MolFromSmiles('['+str(i)+'CH](=O)C')
        if len(mol.GetSubstructMatches(substruct_tmp)) != 0:
            rxn = AllChem.ReactionFromSmarts(f"[{i}*:1] >> [0*:1]")
            if len(rxn.RunReactants(mols)) != 0:
                mols = rxn.RunReactants(mols)[0] 
            break
    return mols[0]



def __remove_disulfide_sign(mol, sign_num=100):
    """
    Remove all disulfide bond labels.
    """
    Chem.SanitizeMol(mol)
    mols = [mol]
    for i in range(5):
        substruct_tmp = Chem.MolFromSmiles('['+str(i+sign_num)+'SH]CC')
        substruct_tmp_1 = Chem.MolFromSmiles('['+str(i+sign_num+100)+'SH]CC')
        if len(mol.GetSubstructMatches(substruct_tmp)) != 0:
            rxn = AllChem.ReactionFromSmarts(f"[{i+sign_num}*:1] >> [0*:1]")
            if len(rxn.RunReactants(mols)) != 0:
                mols = rxn.RunReactants(mols)[0] 
            break
        elif len(mol.GetSubstructMatches(substruct_tmp_1)) != 0:
            rxn = AllChem.ReactionFromSmarts(f"[{i+sign_num+100}*:1] >> [0*:1]")
            if len(rxn.RunReactants(mols)) != 0:
                mols = rxn.RunReactants(mols)[0] 
            break
    return mols[0]



def cut_cyclic_peptide_to_amino_acids(mol, ring_size_min=15, ring_size_max=1000, 
                                      n_cap="[CH3:1]", c_cap="[H:1]",
                                      cut_disulfide=True, 
                                      n_cap_num=77, c_cap_num=66, sign_num=100):
    """
    Cleave the ring of cyclic peptide and give it a given N-terminal cap and C-terminal cap.
    If c_cap_num=66, only hydrogen atom will add to C end.
    If c_cap_num=55, the specified cap will add to C end.
    This function will return the substructures(mols), substructure atom mapping list(mapping_list), and a peptide for visualization(signed_mol).
    """
    # canonical smiles
    smiles = Chem.MolToSmiles(mol)
    mol = Chem.MolFromSmiles(smiles)
    
    # marking
    mol = __sign_C0_in_ring(mol, ring_size_min, ring_size_max)
    if cut_disulfide:
        mol = __sign_disulfide_bonds(mol, sign_num)  
    signed_mol = mol    
    smiles = Chem.MolToSmiles(signed_mol)
    signed_mol = Chem.MolFromSmiles(smiles)
        
    # cut
    mol = __open_ring_by_cutting_signed_amide_bonds(mol, n_cap_num=n_cap_num, c_cap_num=c_cap_num)
    mols = __cut_signed_amide_bonds(mol, n_cap_num=n_cap_num, c_cap_num=c_cap_num)
    if cut_disulfide:
        mols = sum([__cut_signed_disulfide_bonds(mol, sign_num) for mol in mols], [])
        
    # atoms mapping    
    mapping_list = [__get_mapping_atoms(signed_mol, mol, n_cap_num=n_cap_num, c_cap_num=c_cap_num) for mol in mols]
             
    # add cap
    mols = [__replace_dummy_atom(mol, n_cap_num, n_cap) for mol in mols]
    if c_cap_num == 55:
        mols = [__replace_dummy_atom(mol, c_cap_num, c_cap) for mol in mols]
        
    # remove mark
    mols = [__remove_amide_sign(mol) for mol in mols]
    if cut_disulfide:
        mols = [__remove_disulfide_sign(mol) for mol in mols]
    
    return mols, mapping_list, signed_mol
