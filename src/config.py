

aa_options = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']

smiles_options = [' ', '#', '%', '(', ')', '+', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', 
                  '@', 'A', 'B', 'C', 'F', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'V', 'X', 'Z', '[', 
                  '\\', ']', 'a', 'b', 'c', 'e', 'g', 'i', 'l', 'n', 'o', 'p', 'r', 's', 't', 'u', 'd', 'G', 'W']

padding_char = 'U'

max_aa_len = 2500

max_smiles_len = 250

# Kd/IC50 threshold to define a "strong" or "relevant" Drut-Target Interaction 
# results will be filtered to values less than this 
# units in nM 
affinty_threshold = 1000


assert padding_char not in aa_options, 'padding char in aa options'
assert padding_char not in smiles_options, 'padding char in smiles options'