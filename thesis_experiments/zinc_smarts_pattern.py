def get_zinc_smarts_patterns():
    # appear 5735 times in first 1k zinc molecules
    zinc_smarts_patterns = [
        'C=O', # 0
        'NC=O', # 1
        'C1=CC=CC=C1', # 2
        'NCCO', # 3
        'OC1=CC=CC=C1', # 4
        'NC1=CC=CC=C1', # 5
        'COC', # 6
        'C1=CC=NC=C1', # 7
        'NCN', # 8
        'O=S', # 9
        'CN1CCCCC1', # 10
        'C1CCNCC1', # 11
        'O=CO' # 12
    ]
    zinc_smarts_patterns = [ [m] for m in zinc_smarts_patterns ]
    return zinc_smarts_patterns

def get_zinc_smarts_patterns_xl():
    # smarts that appear 5% and >= 2 nodes. 43579 matches in first 1k zinc molecules
    zinc_smarts_patterns_xl = [
        'CC',
        'CN',
        'CCN',
        'C=O',
        'NC=O',
        'CCC=O',
        'CCCCC',
        'CCCCCC',
        'C1CC1',
        'CCCN',
        'CCO',
        'CCCC',
        'C1=CC=CC=C1',
        'CO',
        'CC=O',
        'CCC',
        'CCCO',
        'NCCO',
        'CC1=CC=CC=C1',
        'CF',
        'OC1=CC=CC=C1',
        'NC1=CC=CC=C1',
        'C1=CN=CN=C1',
        'COC',
        'C1=CSC=C1',
        'CS',
        'C1=NN=C[NH]1',
        'C1=CC=NC=C1',
        'CCS',
        'C1=N[NH]C=N1',
        'CC1=N[NH]C=C1',
        'C1=C[NH]N=C1',
        'CC1=CC=N[NH]1',
        'CN1CCCC1',
        'C1=CSC=N1',
        'C1CCNC1',
        'C1CCCCC1',
        'CCl',
        'C1CNCCN1',
        'CN1CCNCC1',
        'CC1=CC=CN=C1',
        'NCN',
        'CC[NH3+]',
        'CCC[NH3+]',
        'CCCC[NH3+]',
        'C[NH3+]',
        'C=C',
        'C=CC',
        'C1=COC=C1',
        'O=S',
        'CCCCO',
        'CN1CCCCC1',
        'CCCCC[NH3+]',
        'C1CCNCC1',
        'CN1C=CC=N1',
        'SC1=CC=CC=C1',
        'CCCS',
        'C=CCN',
        'CCCCCO',
        'O=CO',
        'C=N',
        'NS']
    zinc_smarts_patterns_xl = [ [m] for m in zinc_smarts_patterns_xl ]
    return zinc_smarts_patterns_xl