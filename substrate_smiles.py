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
    'Pyr': 'C1=CC(=CN=C1)CC(C(=O)O)N',
    'Dec': 'CCCCCCCCC(C(=O)O)N',
    'omeTyr': 'COC1=CC=C(C=C1)CC(C(=O)O)N',
    'meGlu': 'CC(CC(=O)O)C(C(=O)O)N',
    'Adh': 'CC(C=C(C)C)C(C(=O)O)N',
    'Kyn': 'C1=CC=C(C(=C1)C(=O)CC(C(=O)O)N)N',
    'Sal': 'C1=CC=C(C(=C1)C(=O)O)O',
    'meSal': 'COC(=O)C1=CC=CC=C1O',
    'oxoDec': 'CCC(=O)CCCCCC(N)C(=O)O',
    '1-mxTrp': 'CN1C=C(C2=CC=CC=C21)CC(C(=O)O)N',
    'Cit': 'C(CC(C(=O)O)N)CNC(=O)N',
    '4,5diOhOrn': 'C(C(C(N)O)O)C(C(=O)O)N',
    'Aib': 'CC(C)(C(=O)O)N',
    'aminoOxoDec': 'NC(CC(C)CC(O)CC(=O)CC)C(=O)O',
    'Piz': 'N1NCCCC1(C(=O)O)',
    'aIle': 'NC(C(C)CC)C(=O)O',
    'aThr': 'CC(O)C(N)C(=O)O',
    'diOH-Bz': 'Oc1c(O)cccc1C(=O)O',
    'bLys': 'NCCCC(N)CC(=O)O',
    'bOH-Tyr': 'NC(C(O)c1ccc(O)cc1)C(=O)O',
    'C12-Hpg': 'NC(c1cc(Cl)c(O)c(Cl)c1)C(=O)O',
    'D-Ala': 'NC(C)C(=O)O', # D-alanine
    'D-Hiv': 'OC(C(C)C)C(=O)O', # D-2-hydroxyisovalerate
    'D-Hmp': 'CCC(C)C(O)C(=O)O', # 2-hydroxy-3-methyl-pentanoic acid (D-hmp)
    'dhpg': 'NC(c1cc(O)cc(O)c1)C(=O)O', # 3,5-dihydroxy-phenylglycine
    'Hpr': 'N1C(CCCC1)C(=O)O', # pipecolic acid
    'IVal': 'NC(CC)(C)C(=O)O', # isovaline
    'OH-Orn': 'NC(CCCNO)C(=O)O', # N5-hydroxyornithine
    'Fo-OH-Orn': 'NC(CCCN(O)C=O)C(=O)O', # N5-formyl-N5-hydroxyornithine
    'Ac-OH-Orn': 'NC(CCCN(O)C(=O)C)C(=O)O', # L-δ-N-acetyl-δ-N-hydroxyornithine/L-Nδ-hydroxy-Nδ-acylornithine
    'C10:0-NH2(2)-Ep(9)-oxo(8)': 'NC(CCCCCC(=O)C1OC1)C(=O)O', # 2-amino-8-oxo-9,10-decanoate
    'Valol': 'NC(C(C)C)CO', # valinol
    'Pgl': 'NC(c1ccccc1)C(=O)O', # phenylglycine
    'pPro': 'N1CC(CCC)CC1C(=O)O', # 4-propyl-proline
    'aad': 'NC(CCCC(=O)O)C(=O)O', # 2-amino-adipic acid
    'abu': 'NC(C(C))C(=O)O', # 2-amino-butyric acid
    'bmt': 'NC(C(O)C(C)CC=CC)C(=O)O', # 4-butenyl-4-methyl threonine
    'cap': 'NC(C1CCN=C(N1)N)C(=O)O', # capreomycidine
    'dab': 'NC(CCN)C(=O)O', # 2,4-diaminobutyric acid
    'dht': 'NC(C(=O)C)C(=O)O', # dehydro-threonine/2,3-dehydroaminobutyric acid
    'hiv': 'OC(C(C)C)C(=O)O', # 2-hydroxyisovalerate
    'hyv': 'NC(C(CO)C)C(=O)O', # 4-hydroxy-L-valine
    'hyv-d': 'OC(C(C)C)C(=O)O', # 2-hydroxy-valeric acid
    'tcl': 'NC(CC(C)C(Cl)(Cl)(Cl))C(=O)O', # (4S)-5,5,5-trichloro-leucine
    'LDAP': 'NC(CCCC(N)C(=O)O)C(=O)O', # diaminopimelic acid
    'meval': 'NC(C(C)(C)C)C(=O)O', # Me-Val
    'alaninol': 'NC(C)CO',
    'N-(1,1-dimethyl-1-allyl)Trp': 'NC(CC1=CN(C(C)(C)C=C)c2c1cccc2)C(=O)O',
    'd-lyserg': 'CN1CC(C=C2C1CC3=CNC4=CC=CC2=C34)C(=O)O', # D-lysergic acid
    'ser-thr': 'NC(C([*])O)C(=O)O', # Serine or Threonine
    'mephe': 'NC(C(C)c1ccccc1)C(=O)O', # Cmethyl-phenylalanine?
    'hasn': 'NC(C(O)C(=O)N)C(=O)O', # hydroxyasparagine
    's-nmethoxy-trp': 'NC(CC1=CN(OC)c2c1cccc2)C(=O)O',
    'alpha-hydroxy-isocaproic-acid': 'OC(C(O)CC(C)C)C(=O)O', # 2S-Hic
    'MeHOval': 'O=C(C(C)CC)C(=O)O', # 3-Methyl-2-oxovaleric acid
    '2-oxo-isovaleric-acid': 'O=C(C(C)C)C(=O)O',
    'aoda': 'NC(CCCCCC(=O)CC)C(=O)O', # S-2-amino-8-oxodecanoic acid
    'mal': 'CC(=O)', # malonyl-CoA
    'ohmal': 'CC(O)',
    'ccmal': 'C=C', # double-bonded malonyl-CoA
    'redmal': 'CC', # reduced malonyl-CoA
    'me-mal': 'C(C)C(=O)',
    'me-ohmal': 'C(C)C(O)',
    'me-ccmal': 'C(C)=C',
    'me-redmal': 'C(C)C',
    'mxmal': 'C(OC)C(=O)',
    'ohmxmal': 'C(OC)C(O)',
    'ccmxmal': 'C(OC)=C',
    'redmxmal': 'C(OC)C',
    'emal': 'C(CC)C(=O)',
    'ohemal': 'C(CC)C(O)',
    'ccemal': 'C(CC)=C',
    'redemal': 'C(CC)C',
    'AHBA': 'C1=CC(=O)C(N)C=C1C(=O)O',
    'ohAHBA': 'C1=CC(O)C(N)C=C1C(=O)O',  # possible modification by a KR, consistent with PKS substrate modification naming
    'fatty_acid': 'C[*]C(=O)O',
    'NH2': 'N',
    'Acetyl-CoA': 'CC(=O)O',
    'shikimic_acid': 'C1C(O)C(O)C(O)C=C1C(=O)O'
}

def main():
    for sub in sub_to_smiles:
        print(sub, sub_to_smiles[sub], sep='\t')

if __name__ == '__main__':
    main()
