def canonicalize(sub):
    if sub == 'hfOrn':
        return '5hfOrn'
    elif sub == 'hOrn':
        return '5hOrn'
    elif sub == 'horn':
        return '5hOrn'
    elif sub == 'MePro':
        return '4-MePro'
    elif sub == 'Thr(Dhb)':
        return 'Thr'
    elif sub == 'Val(Dhv)':
        return 'Val'
    elif sub == 'A':
        return 'Ala'
    elif sub == 'R':
        return 'Arg'
    elif sub == 'N':
        return 'Asn'
    elif sub == 'D':
        return 'Asp'
    elif sub == 'C':
        return 'Cys'
    elif sub == 'Q':
        return 'Gln'
    elif sub == 'E':
        return 'Glu'
    elif sub == 'G':
        return 'Gly'
    elif sub == 'H':
        return 'His'
    elif sub == 'I':
        return 'Ile'
    elif sub == 'L':
        return 'Leu'
    elif sub == 'K':
        return 'Lys'
    elif sub == 'M':
        return 'Met'
    elif sub == 'F':
        return 'Phe'
    elif sub == 'P':
        return 'Pro'
    elif sub == 'S':
        return 'Ser'
    elif sub == 'T':
        return 'Thr'
    elif sub == 'W':
        return 'Trp'
    elif sub == 'Y':
        return 'Tyr'
    elif sub == 'V':
        return 'Val'
    elif sub == 'aad':
        return 'Aad'
    elif sub == 'dhpg':
        return 'Dhpg'
    elif sub == 'hpg':
        return 'Hpg'
    elif sub == 'bht':
        return 'Bht'
    elif sub == 'dht':
        return 'Dht'
    elif sub == 'meGlu':
        return 'meGlu'
    elif sub == 'omeTyr':
        return 'omeTyr'
    elif sub == sub.lower():
        return sub[0].upper() + sub[1:]
    elif sub == sub.upper():
        return sub[0] + sub[1:].lower()
    else:
        return sub

# Commented out substrates are for the zeroshot task, not relevant for now.
sub_to_smiles = {
    'Arg': 'C(CC(C(=O)O)N)CN=C(N)N',
    'His': 'C1=C(NC=N1)CC(C(=O)O)N',
    'Lys': 'C(CCN)CC(C(=O)O)N',
    'Asp': 'C(C(C(=O)O)N)C(=O)O',
    'Glu': 'C(CC(=O)O)C(C(=O)O)N',
    'Ser': 'C(C(C(=O)O)N)O',
    'Thr': 'CC(C(C(=O)O)N)O',
    'Asn': 'C(C(C(=O)O)N)C(=O)N',
    'Gln': 'C(CC(=O)N)C(C(=O)O)N',
    'Cys': 'C(C(C(=O)O)N)S',
    'Gly': 'C(C(=O)O)N',
    'Pro': 'C1CC(NC1)C(=O)O',
    'Ala': 'CC(C(=O)O)N',
    'Val': 'CC(C)C(C(=O)O)N',
    'Ile': 'CCC(C)C(C(=O)O)N',
    'Leu': 'CC(C)CC(C(=O)O)N',
    'Met': 'CSCCC(C(=O)O)N',
    'Phe': 'C1=CC=C(C=C1)CC(C(=O)O)N',
    'Tyr': 'C1=CC(=CC=C1CC(C(=O)O)N)O',
    'Trp': 'C1=CC=C2C(=C1)C(=CN2)CC(C(=O)O)N',
    'Orn': 'C(CC(C(=O)O)N)CN',
    '5hfOrn': 'C(CC(C(=O)O)N)CN(C=O)O',
    '5hOrn': 'C(CC(C(=O)O)N)CNO',
    '4-MePro': 'CC1CC(NC1)C(=O)O',
    'Bht': 'C1=CC(=CC=C1C(C(C(=O)O)N)O)O',
    'Hty': 'C1=CC(=CC=C1CCC(C(=O)O)N)O',
    'MeDap': 'CC(C(C(=O)O)N)N',
    'MeAsp': 'CC(C(C(=O)O)N)C(=O)O',
    'HyAsp': 'OC(C(C(=O)O)N)C(=O)O',
    'CAM': 'OC(=O)C(N)C1CCNC(=N)N1',
    'hEnd': 'C1C(NC(N1)=N)C(C(C(=O)O)N)O',
    'Dab': 'OC(=O)C(N)CCN',
    'l-2,3-DAP': 'OC(=O)C(N)CN',
    'Aad': 'C(CC(C(=O)O)N)CC(=O)O',
    'Dhpg': 'C1=C(C=C(C=C1O)O)C(C(=O)O)N',
    'Hpg': 'C1=CC(=CC=C1C(C(=O)O)N)O',
    'Dhb': 'C1=CC(=C(C(=C1)O)O)C(=O)O',
    'Beta-ala': 'C(CN)C(=O)O',
    'Hyv-d': 'CC(CO)C(C(=O)O)N',
    'Pip': 'C1CCNC(C1)C(=O)O',
    'Dht': 'CC(=C(C(=O)O)N)O',
#    'Pyr': 'C1=CC(=CN=C1)CC(C(=O)O)N',
#    'Dec': 'CCCCCCCCC(C(=O)O)N',
#    'omeTyr': 'COC1=CC=C(C=C1)CC(C(=O)O)N',
#    'meGlu': 'CC(CC(=O)O)C(C(=O)O)N',
#    'Adh': 'CC(C=C(C)C)C(C(=O)O)N',
#    'Kyn': 'C1=CC=C(C(=C1)C(=O)CC(C(=O)O)N)N',
#    'Sal': 'C1=CC=C(C(=C1)C(=O)O)O', 
#    'meSal': 'COC(=O)C1=CC=CC=C1O', 
#    'oxoDec': 'CCC(=O)CCCCCC(N)C(=O)O',
#    '1-mxTrp': 'CN1C=C(C2=CC=CC=C21)CC(C(=O)O)N',
#    'Cit': 'C(CC(C(=O)O)N)CNC(=O)N',
#    '4,5diOhOrn': 'C(C(C(N)O)O)C(C(=O)O)N',
#    'Aib': 'CC(C)(C(=O)O)N',
#    'aminoOxoDec': 'NC(CC(C)CC(O)CC(=O)CC)C(=O)O',
#    'Piz': 'N1NCCCC1(C(=O)O)',
}

def main():
    for sub in sub_to_smiles:
        print(sub, sub_to_smiles[sub], sep='\t')

if __name__ == '__main__':
    main()
