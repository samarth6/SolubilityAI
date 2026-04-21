import pandas as pd
import streamlit as st
import requests

MOLECULES = [
    {"name": "Benzene",           "smiles": "c1ccccc1",                             "logS": -1.77},
    {"name": "Ethanol",           "smiles": "CCO",                                   "logS":  0.83},
    {"name": "Acetic Acid",       "smiles": "CC(=O)O",                               "logS":  0.17},
    {"name": "Phenol",            "smiles": "c1ccc(cc1)O",                           "logS": -0.72},
    {"name": "Isopropanol",       "smiles": "CC(C)O",                                "logS":  0.35},
    {"name": "Butane",            "smiles": "CCCC",                                  "logS": -2.89},
    {"name": "Naphthalene",       "smiles": "c1ccc2ccccc2c1",                        "logS": -3.21},
    {"name": "Aspirin",           "smiles": "CC(=O)Oc1ccccc1C(=O)O",                "logS": -3.08},
    {"name": "Nicotine",          "smiles": "CN1CCC[C@H]1c2cccnc2",                 "logS": -1.02},
    {"name": "Testosterone",      "smiles": "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C", "logS": -5.12},
    {"name": "Benzoic Acid",      "smiles": "OC(=O)c1ccccc1",                       "logS": -0.84},
    {"name": "Pyridine",          "smiles": "c1ccncc1",                              "logS":  0.56},
    {"name": "Diethyl Ether",     "smiles": "CCOCC",                                 "logS": -0.41},
    {"name": "Sorbitol",          "smiles": "OCC(O)C(O)C(O)C(O)CO",                 "logS":  0.85},
    {"name": "Chlorobenzene",     "smiles": "c1ccc(cc1)Cl",                          "logS": -2.41},
    {"name": "Ethylbenzene",      "smiles": "CCc1ccccc1",                            "logS": -2.88},
    {"name": "Decane",            "smiles": "CCCCCCCCCC",                            "logS": -6.54},
    {"name": "Aniline",           "smiles": "c1ccc(cc1)N",                           "logS": -1.37},
    {"name": "Acetophenone",      "smiles": "CC(=O)c1ccccc1",                       "logS": -1.94},
    {"name": "Cyclohexanol",      "smiles": "OC1CCCCC1",                             "logS": -1.22},
    {"name": "Fluorobenzene",     "smiles": "c1ccc(cc1)F",                           "logS": -2.11},
    {"name": "Ethyl Propanoate",  "smiles": "CCOC(=O)CC",                            "logS": -1.89},
    {"name": "Furan",             "smiles": "c1ccoc1",                               "logS": -1.03},
    {"name": "Acetonitrile",      "smiles": "CC#N",                                  "logS":  0.28},
    {"name": "Malonic Acid",      "smiles": "OC(=O)CC(O)=O",                        "logS": -0.61},
    {"name": "Bromobenzene",      "smiles": "c1ccc(cc1)Br",                          "logS": -2.96},
    {"name": "Butanol",           "smiles": "CCCCO",                                 "logS": -0.72},
    {"name": "tert-Butanol",      "smiles": "CC(C)(C)O",                            "logS": -0.41},
    {"name": "Indole",            "smiles": "c1ccc2[nH]ccc2c1",                     "logS": -2.76},
    {"name": "Caffeine",          "smiles": "Cn1cnc2c1c(=O)n(c(=O)n2C)C",           "logS": -1.35},
]

@st.cache_data
def get_demo_dataset():
    return pd.DataFrame(MOLECULES)

def get_molecule_options():
    return [f"{m['name']}  —  {m['smiles']}" for m in MOLECULES]

def get_smiles_from_selection(selection):
    return selection.split("  —  ")[1].strip()

def get_name_from_smiles(smiles):
    for m in MOLECULES:
        if m["smiles"] == smiles:
            return m["name"]
    return smiles

def pubchem_name_to_smiles(name):
    """Fetch SMILES from PubChem by molecule common name."""
    try:
        url = (
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
            f"{name}/property/IsomericSMILES/JSON"
        )
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()["PropertyTable"]["Properties"][0]["IsomericSMILES"]
    except Exception:
        pass
    return None
