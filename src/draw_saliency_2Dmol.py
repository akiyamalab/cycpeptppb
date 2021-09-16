import numpy as np
from rdkit.Chem.Draw import rdMolDraw2D
from matplotlib import cm


def __convert_to_color_dict(aDict, intensity=0.7):
    """
    Takes a dictionary type as input and returns a dictionary type with the value of the dictionary type converted to a blue-red colormap.
    """

    # intensity must be [0.1, 1.0]
    intensity = min(max(intensity, 0.1), 1.0)

    vals = np.array(list(aDict.values()))
    # scale values to [-1,1]
    vals = vals / np.max(np.abs(vals))
    # intensity calibration
    vals = vals * intensity
    # convert [-1,1] to [0,1]
    vals = (vals + 1) / 2

    color_dict = {k:cm.bwr(v)[:3] for k,v in zip(aDict.keys(), vals)}
    return color_dict



def __generate_molecule_svg(path, mol, dict_atom_values=None,
                            width=700, height=700,
                            fontsize=0.5):
    """
    Highlight each atom of the input mol object and save it as an svg file.
    """
    if dict_atom_values is None:
        dict_atom_values = {}
    highlightAtomColors = __convert_to_color_dict(dict_atom_values)

    view = rdMolDraw2D.MolDraw2DSVG(width,height)
    view.SetFontSize(fontsize)
    view.DrawMolecule(rdMolDraw2D.PrepareMolForDrawing(mol),
                      highlightAtoms=highlightAtomColors.keys(),
                      highlightAtomColors=highlightAtomColors,
                      highlightBonds=None)
    view.FinishDrawing()
    svg = view.GetDrawingText()
    with open(path, "w") as f:
        f.write(svg)
        
        
        
def __visualization_of_saliency_score(fig_path, signed_mol_list, mapping_list, saliency):
    """
    Using mapping_list, create dictionary dict_atom_values for saliency of corresponding atoms.
    """
    for i in range(len(signed_mol_list)):
        path = fig_path + 'input_' + str(i)
        dict_atom_values = {}

        for j in range(len(mapping_list[i])):
            for k in mapping_list[i][j]:
                dict_atom_values[k] = saliency[i][j]

        __generate_molecule_svg(path, signed_mol_list[i], dict_atom_values)



def __get_color_of_saliency_score(saliency):
    """
    Get the color of each substructure
    """
    saliency_color_list = []

    for pep_ in range(len(saliency)):
        now_saliency = saliency[pep_]
        now_color = {}
        
        for i in range(len(now_saliency)):
            now_color[i] = now_saliency[i]
        
        now_color = __convert_to_color_dict(now_color)
        
        saliency_color_list.append(now_color)


    return saliency_color_list
















