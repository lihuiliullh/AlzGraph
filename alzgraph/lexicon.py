"""Alzheimer's disease NER lexicon for dictionary-based entity recognition.

Maps surface forms (canonical names, synonyms, abbreviations) to a canonical
entity and one of the five AlzKG layers. Used by ``scripts/build_kg_from_corpus.py``
to recognize entities in real PubMed abstracts, from which cross-layer relations
and their literature paper counts are mined.

Each canonical entity lists:
  - ``ci``: case-insensitive surface forms (names, multi-word phrases)
  - ``cs``: case-sensitive surface forms (gene symbols, uppercase acronyms),
            so that e.g. "APP" (amyloid precursor protein) does not match "app".
"""

import re
from typing import Dict, List, Tuple

# canonical -> {layer, ci:[...], cs:[...]}
LEXICON: Dict[str, dict] = {
    # ------------------------------------------------------------------ genes
    "APP": {"layer": "gene", "cs": ["APP"], "ci": ["amyloid precursor protein"]},
    "PSEN1": {"layer": "gene", "cs": ["PSEN1", "PS1"], "ci": ["presenilin 1", "presenilin-1"]},
    "PSEN2": {"layer": "gene", "cs": ["PSEN2", "PS2"], "ci": ["presenilin 2", "presenilin-2"]},
    "APOE": {"layer": "gene", "cs": ["APOE", "ApoE", "APOE4", "ApoE4"], "ci": ["apolipoprotein e", "apoe e4", "apoe epsilon 4", "apoe ε4"]},
    "TREM2": {"layer": "gene", "cs": ["TREM2"], "ci": ["triggering receptor expressed on myeloid cells 2"]},
    "ABCA7": {"layer": "gene", "cs": ["ABCA7"], "ci": []},
    "CLU": {"layer": "gene", "cs": ["CLU"], "ci": ["clusterin"]},
    "CR1": {"layer": "gene", "cs": ["CR1"], "ci": ["complement receptor 1"]},
    "BIN1": {"layer": "gene", "cs": ["BIN1"], "ci": ["bridging integrator 1"]},
    "PICALM": {"layer": "gene", "cs": ["PICALM"], "ci": []},
    "SORL1": {"layer": "gene", "cs": ["SORL1"], "ci": ["sortilin-related receptor 1"]},
    "MAPT": {"layer": "gene", "cs": ["MAPT"], "ci": ["microtubule-associated protein tau", "tau gene"]},
    "PLCG2": {"layer": "gene", "cs": ["PLCG2"], "ci": ["phospholipase c gamma 2"]},
    "MS4A6A": {"layer": "gene", "cs": ["MS4A6A", "MS4A"], "ci": []},
    "CD33": {"layer": "gene", "cs": ["CD33"], "ci": ["siglec-3"]},
    "SPI1": {"layer": "gene", "cs": ["SPI1"], "ci": []},
    "INPP5D": {"layer": "gene", "cs": ["INPP5D", "SHIP1"], "ci": []},
    "ADAM10": {"layer": "gene", "cs": ["ADAM10"], "ci": []},
    "ABI3": {"layer": "gene", "cs": ["ABI3"], "ci": []},
    "CD2AP": {"layer": "gene", "cs": ["CD2AP"], "ci": []},
    "PTK2B": {"layer": "gene", "cs": ["PTK2B"], "ci": []},
    "CASS4": {"layer": "gene", "cs": ["CASS4"], "ci": []},
    "FERMT2": {"layer": "gene", "cs": ["FERMT2"], "ci": []},
    "GRN": {"layer": "gene", "cs": ["GRN"], "ci": ["progranulin"]},
    "ACE": {"layer": "gene", "cs": ["ACE1"], "ci": ["angiotensin-converting enzyme gene"]},

    # ------------------------------------------------------------- biomarkers
    "Amyloid-beta 42": {"layer": "biomarker", "cs": ["Aβ42", "Abeta42", "Aβ1-42", "Aβ42"], "ci": ["amyloid-beta 42", "amyloid beta 42", "amyloid-β42", "abeta 42", "a-beta 42"]},
    "Amyloid-beta 42/40 ratio": {"layer": "biomarker", "cs": ["Aβ42/40"], "ci": ["amyloid-beta 42/40", "abeta42/40", "aβ42/40 ratio", "42/40 ratio"]},
    "Amyloid PET": {"layer": "biomarker", "cs": ["PiB-PET", "PiB"], "ci": ["amyloid pet", "amyloid imaging", "amyloid positron emission", "florbetapir", "florbetaben", "flutemetamol", "amyloid-pet"]},
    "p-tau181": {"layer": "biomarker", "cs": ["p-tau181", "ptau181", "pTau181"], "ci": ["phosphorylated tau 181", "phospho-tau 181", "p-tau 181"]},
    "p-tau217": {"layer": "biomarker", "cs": ["p-tau217", "ptau217", "pTau217"], "ci": ["phosphorylated tau 217", "phospho-tau 217", "p-tau 217"]},
    "p-tau231": {"layer": "biomarker", "cs": ["p-tau231", "ptau231"], "ci": ["phospho-tau 231", "p-tau 231"]},
    "Total tau": {"layer": "biomarker", "cs": ["t-tau"], "ci": ["total tau", "total-tau"]},
    "Tau PET": {"layer": "biomarker", "cs": [], "ci": ["tau pet", "tau imaging", "flortaucipir", "tau positron emission", "tau-pet"]},
    "Neurofibrillary tangles": {"layer": "biomarker", "cs": ["NFT"], "ci": ["neurofibrillary tangle", "neurofibrillary tangles", "tau tangles"]},
    "Amyloid plaques": {"layer": "biomarker", "cs": [], "ci": ["amyloid plaque", "amyloid plaques", "senile plaque", "amyloid deposition", "amyloid burden"]},
    "Hippocampal atrophy": {"layer": "biomarker", "cs": [], "ci": ["hippocampal atrophy", "hippocampal volume", "medial temporal atrophy", "medial temporal lobe atrophy"]},
    "FDG-PET hypometabolism": {"layer": "biomarker", "cs": ["FDG-PET", "FDG PET"], "ci": ["fluorodeoxyglucose pet", "cerebral hypometabolism", "glucose hypometabolism"]},
    "Plasma NfL": {"layer": "biomarker", "cs": ["NfL", "NEFL"], "ci": ["neurofilament light", "neurofilament-light"]},
    "Plasma GFAP": {"layer": "biomarker", "cs": ["GFAP"], "ci": ["glial fibrillary acidic protein"]},
    "MMSE": {"layer": "biomarker", "cs": ["MMSE"], "ci": ["mini-mental state", "mini mental state"]},
    "MoCA": {"layer": "biomarker", "cs": ["MoCA"], "ci": ["montreal cognitive assessment"]},
    "CDR": {"layer": "biomarker", "cs": ["CDR", "CDR-SB", "CDR-SOB"], "ci": ["clinical dementia rating"]},
    "ADAS-Cog": {"layer": "biomarker", "cs": ["ADAS-Cog", "ADAS-cog", "ADAScog"], "ci": ["alzheimer's disease assessment scale"]},

    # ----------------------------------------------------------------- stages
    "Alzheimer's disease": {"layer": "stage", "cs": ["AD"], "ci": ["alzheimer's disease", "alzheimer disease", "alzheimers disease", "alzheimer's dementia"]},
    "Mild cognitive impairment": {"layer": "stage", "cs": ["MCI", "aMCI"], "ci": ["mild cognitive impairment", "amnestic mci", "prodromal alzheimer"]},
    "Preclinical AD": {"layer": "stage", "cs": [], "ci": ["preclinical alzheimer", "preclinical ad", "asymptomatic alzheimer"]},
    "Early-onset AD": {"layer": "stage", "cs": ["EOAD"], "ci": ["early-onset alzheimer", "early onset alzheimer", "autosomal dominant alzheimer", "familial alzheimer"]},
    "Late-onset AD": {"layer": "stage", "cs": ["LOAD"], "ci": ["late-onset alzheimer", "late onset alzheimer", "sporadic alzheimer"]},
    "AD dementia": {"layer": "stage", "cs": [], "ci": ["ad dementia", "dementia due to alzheimer", "alzheimer-type dementia", "dementia of the alzheimer"]},
    "Posterior cortical atrophy": {"layer": "stage", "cs": ["PCA"], "ci": ["posterior cortical atrophy"]},
    "Logopenic variant PPA": {"layer": "stage", "cs": ["lvPPA"], "ci": ["logopenic", "logopenic variant", "primary progressive aphasia"]},
    "Dementia with Lewy bodies": {"layer": "stage", "cs": ["DLB"], "ci": ["lewy body", "dementia with lewy bodies"]},
    "Frontotemporal dementia": {"layer": "stage", "cs": ["FTD"], "ci": ["frontotemporal dementia", "frontotemporal lobar"]},
    "Vascular dementia": {"layer": "stage", "cs": ["VaD"], "ci": ["vascular dementia", "vascular cognitive impairment"]},

    # -------------------------------------------------------------- treatments
    "Donepezil": {"layer": "treatment", "cs": [], "ci": ["donepezil", "aricept"]},
    "Rivastigmine": {"layer": "treatment", "cs": [], "ci": ["rivastigmine", "exelon"]},
    "Galantamine": {"layer": "treatment", "cs": [], "ci": ["galantamine", "razadyne"]},
    "Memantine": {"layer": "treatment", "cs": [], "ci": ["memantine", "namenda"]},
    "Lecanemab": {"layer": "treatment", "cs": [], "ci": ["lecanemab", "leqembi", "ban2401"]},
    "Donanemab": {"layer": "treatment", "cs": [], "ci": ["donanemab", "kisunla"]},
    "Aducanumab": {"layer": "treatment", "cs": [], "ci": ["aducanumab", "aduhelm"]},
    "Solanezumab": {"layer": "treatment", "cs": [], "ci": ["solanezumab"]},
    "Gantenerumab": {"layer": "treatment", "cs": [], "ci": ["gantenerumab"]},
    "Cholinesterase inhibitor": {"layer": "treatment", "cs": ["AChEI", "ChEI"], "ci": ["cholinesterase inhibitor", "acetylcholinesterase inhibitor"]},
    "Anti-amyloid antibody": {"layer": "treatment", "cs": [], "ci": ["anti-amyloid antibody", "anti-amyloid monoclonal", "anti-amyloid immunotherapy", "amyloid-targeting", "anti-aβ antibody"]},
    "Cognitive stimulation therapy": {"layer": "treatment", "cs": ["CST"], "ci": ["cognitive stimulation", "cognitive training", "cognitive intervention"]},
    "Physical exercise": {"layer": "treatment", "cs": [], "ci": ["physical exercise", "aerobic exercise", "physical activity"]},

    # ---------------------------------------------------------------- outcomes
    "ARIA-E": {"layer": "outcome", "cs": ["ARIA-E"], "ci": ["amyloid-related imaging abnormalities edema", "aria edema", "vasogenic edema"]},
    "ARIA-H": {"layer": "outcome", "cs": ["ARIA-H"], "ci": ["amyloid-related imaging abnormalities hemorrhage", "microhemorrhage", "microhaemorrhage", "superficial siderosis", "cerebral microbleed"]},
    "ARIA": {"layer": "outcome", "cs": ["ARIA"], "ci": ["amyloid-related imaging abnormalities", "amyloid related imaging"]},
    "Cognitive decline": {"layer": "outcome", "cs": [], "ci": ["cognitive decline", "cognitive deterioration", "cognitive impairment progression", "memory decline"]},
    "Functional decline": {"layer": "outcome", "cs": [], "ci": ["functional decline", "loss of function", "activities of daily living decline", "functional impairment"]},
    "Disease progression": {"layer": "outcome", "cs": [], "ci": ["disease progression", "progression to dementia", "clinical progression", "conversion to dementia", "conversion to alzheimer"]},
    "Amyloid clearance": {"layer": "outcome", "cs": [], "ci": ["amyloid clearance", "amyloid removal", "plaque reduction", "amyloid reduction", "amyloid lowering"]},
    "Slowing of cognitive decline": {"layer": "outcome", "cs": [], "ci": ["slowing of cognitive decline", "slowed cognitive decline", "slowing decline", "reduced clinical decline", "slowing of clinical decline"]},
    "Adverse events": {"layer": "outcome", "cs": [], "ci": ["adverse event", "adverse effect", "side effect", "tolerability"]},
    "Nausea": {"layer": "outcome", "cs": [], "ci": ["nausea", "vomiting", "gastrointestinal adverse", "diarrhea", "diarrhoea"]},
    "Bradycardia": {"layer": "outcome", "cs": [], "ci": ["bradycardia", "syncope"]},
    "Mortality": {"layer": "outcome", "cs": [], "ci": ["mortality", "death", "survival"]},
    "Hospitalization": {"layer": "outcome", "cs": [], "ci": ["hospitalization", "hospitalisation", "institutionalization", "nursing home"]},
    "Falls": {"layer": "outcome", "cs": [], "ci": ["falls", "fall risk"]},
}


def _compile():
    ci_map: Dict[str, Tuple[str, str]] = {}
    cs_map: Dict[str, Tuple[str, str]] = {}
    ci_surfaces: List[str] = []
    cs_surfaces: List[str] = []
    for canon, spec in LEXICON.items():
        layer = spec["layer"]
        for s in spec.get("ci", []):
            ci_map[s.casefold()] = (canon, layer)
            ci_surfaces.append(s)
        for s in spec.get("cs", []):
            cs_map[s] = (canon, layer)
            cs_surfaces.append(s)
    # Longest-first so multi-word phrases win over shorter substrings.
    ci_surfaces.sort(key=len, reverse=True)
    cs_surfaces.sort(key=len, reverse=True)
    ci_re = re.compile(r"(?<![A-Za-z0-9])(" + "|".join(re.escape(s) for s in ci_surfaces) + r")(?![A-Za-z0-9])", re.IGNORECASE)
    cs_re = re.compile(r"(?<![A-Za-z0-9])(" + "|".join(re.escape(s) for s in cs_surfaces) + r")(?![A-Za-z0-9])")
    return ci_re, ci_map, cs_re, cs_map


_CI_RE, _CI_MAP, _CS_RE, _CS_MAP = _compile()


def detect(text: str) -> Dict[str, set]:
    """Return {layer: set(canonical entities)} recognized in ``text``."""
    found: Dict[str, set] = {}
    for m in _CI_RE.finditer(text):
        hit = _CI_MAP.get(m.group(1).casefold())
        if hit:
            found.setdefault(hit[1], set()).add(hit[0])
    for m in _CS_RE.finditer(text):
        hit = _CS_MAP.get(m.group(1))
        if hit:
            found.setdefault(hit[1], set()).add(hit[0])
    return found


def all_entities() -> List[Tuple[str, str]]:
    return [(c, s["layer"]) for c, s in LEXICON.items()]
