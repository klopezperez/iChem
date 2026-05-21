from rdkit import Chem # type: ignore
from rdkit.Chem import Draw # type: ignore
from rdkit.Chem import rdFMCS # type: ignore
from ..utils.utils import smiles_standarization 
import numpy as np # type: ignore

def smiles_to_grid_image(smiles,
                         mols_per_row=5,
                         sub_img_size=(250, 250),
                         legends=None,
                         standarize=True,
                         MSC=False):
    """
    Convert a list of SMILES strings to a grid image of molecules.

    Parameters:
    - smiles_list: List of SMILES strings.
    - mols_per_row: Number of molecules per row in the grid.
    - sub_img_size: Size of each sub-image (width, height).
    - legends: Optional list of legends for each molecule.
    Returns:
    - A PIL Image object containing the grid of molecule images.
    """
    if len(smiles) > 50:
        smiles = np.random.choice(smiles, 50, replace=False)
    mols = [Chem.MolFromSmiles(smile) for smile in smiles]
    if standarize:
        mols = [smiles_standarization(mol) for mol in mols]
    if MSC:
        return _mols_to_grid_MSC(mols,
                                    mols_per_row,
                                    sub_img_size,
                                    legends)
    if legends is not None:
        img = Draw.MolsToGridImage(mols,
                                   molsPerRow=mols_per_row,
                                   subImgSize=sub_img_size,
                                   legends=legends,
                                   useSVG=True)
    else:
        img = Draw.MolsToGridImage(mols,
                                   molsPerRow=mols_per_row,
                                   subImgSize=sub_img_size,
                                   useSVG=True)
    
    return img

def _mols_to_grid_MSC(mols,
                      mols_per_row=5,
                      sub_img_size=(250, 250),
                      legends=None):
    MCS = rdFMCS.FindMCS(mols, threshold=0.75)
    MCS_mol = Chem.MolFromSmarts(MCS.smartsString)
    for mol in mols:
        if mol.HasSubstructMatch(MCS_mol):
            match = mol.GetSubstructMatch(MCS_mol)
            atom_indices = list(match)
            highlight_atoms = atom_indices
            # Highlight the matching substructure
            mol.SetProp('_highlightAtoms', ','.join(map(str, highlight_atoms)))
    highlight_lists = []
    for mol in mols:
        if mol.HasProp('_highlightAtoms'):
            vals = mol.GetProp('_highlightAtoms').split(',')
            highlight_lists.append(list(map(int, vals)))
        else:
            highlight_lists.append([])
    if legends is not None:
        img = Draw.MolsToGridImage(mols,
                                   highlightAtomLists=highlight_lists,
                                   molsPerRow=mols_per_row,
                                   subImgSize=sub_img_size,
                                   legends=legends,
                                   useSVG=True)
    else:
        img = Draw.MolsToGridImage(mols,
                                   highlightAtomLists=highlight_lists,
                                   molsPerRow=mols_per_row,
                                   subImgSize=sub_img_size,
                                   useSVG=True)
    return img


def MSC_image(smiles,
              n_samples=50,
              MCS_threshold=0.75,
              standarize=True):
    if len(smiles) > n_samples:
        smiles = np.random.choice(smiles, n_samples, replace=False)
    mols = [Chem.MolFromSmiles(smile) for smile in smiles]
    if standarize:
        mols = [smiles_standarization(mol) for mol in mols]
    
    MSC = rdFMCS.FindMCS(mols, threshold=MCS_threshold)
    MCS_mol = Chem.MolFromSmarts(MSC.smartsString)
    return Draw.MolToImage(MCS_mol,
                           size=(350, 350),
                           useSVG=True)