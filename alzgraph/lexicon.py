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
# Hand-curated AD seed vocabulary. Always present; the large ontology-derived
# vocabulary (data/lexicon/lexicon_full.json, built by
# scripts/build_lexicon_from_ontologies.py following EpiGraph's ontology step)
# is merged on top when available.
_SEED_LEXICON: Dict[str, dict] = {
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


def _load_lexicon() -> Dict[str, dict]:
    """Seed vocabulary, with the large ontology-derived vocabulary merged on top
    when ``data/lexicon/lexicon_full.json`` exists (or ``$ALZKG_LEXICON``)."""
    import json
    import os
    from pathlib import Path

    lex: Dict[str, dict] = {c: dict(s) for c, s in _SEED_LEXICON.items()}
    # surface (casefold) -> owning canonical, so an ontology entity that shares any
    # surface form with an existing concept is merged into it (light entity
    # resolution in lieu of EpiGraph's UMLS CUI mapping). Seed concepts win.
    owner: Dict[str, str] = {}
    for canon, spec in lex.items():
        for s in spec.get("ci", []) + spec.get("cs", []):
            owner.setdefault(s.casefold(), canon)

    path = os.environ.get("ALZKG_LEXICON") or str(
        Path(__file__).resolve().parents[1] / "data" / "lexicon" / "lexicon_full.json"
    )
    p = Path(path)
    if not p.exists():
        return lex
    try:
        full = json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return lex

    for canon, spec in full.items():
        surfaces = spec.get("ci", []) + spec.get("cs", [])
        target = canon if canon in lex else None
        if target is None:
            for s in surfaces:  # merge into an existing concept sharing a surface form
                if s.casefold() in owner:
                    target = owner[s.casefold()]
                    break
        if target is None:
            target = canon
            lex[canon] = {"layer": spec["layer"], "ci": [], "cs": []}
        lex[target]["ci"] = sorted(set(lex[target].get("ci", [])) | set(spec.get("ci", [])))
        lex[target]["cs"] = sorted(set(lex[target].get("cs", [])) | set(spec.get("cs", [])))
        for s in surfaces:
            owner.setdefault(s.casefold(), target)
    return lex


LEXICON: Dict[str, dict] = _load_lexicon()

# Token-based gazetteer matcher (scales to ~10^5 surface forms, unlike a single
# regex alternation). Greedy longest-match over word tokens, so multi-word
# phrases win over their substrings and each span is counted once.
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_MAX_NGRAM = 8


def _norm(s: str) -> List[str]:
    return _TOKEN_RE.findall(s)


def _compile():
    ci_map: Dict[str, Tuple[str, str]] = {}
    cs_map: Dict[str, Tuple[str, str]] = {}
    ci_starts: set = set()
    cs_starts: set = set()
    max_n = 1
    for canon, spec in LEXICON.items():
        layer = spec["layer"]
        for s in spec.get("ci", []):
            toks = _norm(s.casefold())
            if not toks or len(toks) > _MAX_NGRAM:
                continue
            ci_map.setdefault(" ".join(toks), (canon, layer))
            ci_starts.add(toks[0])
            max_n = max(max_n, len(toks))
        for s in spec.get("cs", []):
            toks = _norm(s)
            if not toks or len(toks) > _MAX_NGRAM:
                continue
            cs_map.setdefault(" ".join(toks), (canon, layer))
            cs_starts.add(toks[0])
            max_n = max(max_n, len(toks))
    return ci_map, cs_map, ci_starts, cs_starts, max_n


_CI_MAP, _CS_MAP, _CI_STARTS, _CS_STARTS, _MAX_N = _compile()


def detect(text: str) -> Dict[str, set]:
    """Return {layer: set(canonical entities)} recognized in ``text``."""
    raw = _TOKEN_RE.findall(text)
    low = [t.casefold() for t in raw]
    found: Dict[str, set] = {}
    i, n_tok = 0, len(raw)
    while i < n_tok:
        if low[i] not in _CI_STARTS and raw[i] not in _CS_STARTS:
            i += 1
            continue
        hi = min(_MAX_N, n_tok - i)
        matched = 0
        for n in range(hi, 0, -1):
            hit = _CS_MAP.get(" ".join(raw[i:i + n]))
            if hit is None:
                hit = _CI_MAP.get(" ".join(low[i:i + n]))
            if hit is not None:
                found.setdefault(hit[1], set()).add(hit[0])
                matched = n
                break
        i += matched if matched else 1
    return found


def all_entities() -> List[Tuple[str, str]]:
    return [(c, s["layer"]) for c, s in LEXICON.items()]
