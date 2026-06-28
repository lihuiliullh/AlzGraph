"""Build a large AlzKG entity vocabulary from public ontologies (EpiGraph procedure).

EpiGraph constructs its 24k-entity vocabulary not by hand but by harvesting
authoritative ontologies (OMIM/GWAS for genes, ChEBI for drugs, MeSH/HPO for
diagnostics and phenotypes), normalizing aliases, and keeping the entities that
the literature actually grounds. This script reproduces that vocabulary step for
the five AlzKG layers, using only public, license-free endpoints:

  gene      <- HGNC complete set (protein-coding symbols + aliases + prev symbols)
  stage     <- MeSH dementia / neurodegeneration descriptor sub-trees (+ synonyms)
  biomarker <- MeSH diagnostic-technique / imaging sub-trees (+ synonyms) + curated
  treatment <- MeSH topical descriptors carrying a pharmacological action (+ synonyms)
  outcome   <- HPO nervous-system / cognition phenotype sub-tree (+ synonyms) + curated

The output (``data/lexicon/lexicon_full.json``) is the same schema the hand lexicon
uses -- ``canonical -> {layer, ci:[...], cs:[...]}`` -- so it is a drop-in for the
gazetteer matcher. Cross-layer conflicts are resolved by a fixed layer priority so
every entity lives in exactly one layer (required by the relation typing). Sources
are cached under ``data/lexicon/sources/`` (gitignored). Standard library only.
"""

import argparse
import gzip
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

HGNC_URL = "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt"
HPO_URL = "https://purl.obolibrary.org/obo/hp.obo"
CHEBI_URL = "https://ftp.ebi.ac.uk/pub/databases/chebi/ontology/chebi.obo.gz"
MESH_SPARQL = "https://id.nlm.nih.gov/mesh/sparql"

# ChEBI role IDs whose has_role members are AlzKG "treatments".
CHEBI_DRUG_ROLES = {"CHEBI:23888", "CHEBI:52217"}  # drug, pharmaceutical

# MeSH descriptor tree prefixes per AlzKG layer.
MESH_STAGE_TREES = [
    "C10.228.140.380",   # Dementia and subtypes
    "F03.087",           # Mental disorders -> Dementia
    "C10.574",           # Neurodegenerative diseases
    "C10.228.662",       # Neurodegenerative (alt)
]
MESH_BIOMARKER_TREES = [
    "E01.370.350",       # Diagnostic Imaging
    "E01.370.225",       # Clinical laboratory techniques
    "E05.196",           # Biomarkers / evaluation
    "E01.370",           # Diagnostic techniques and procedures
]
# HPO roots whose descendants are AlzKG "outcomes" (clinical course / phenotype).
HPO_OUTCOME_ROOTS = {"HP:0000707", "HP:0011446", "HP:0100543", "HP:0040279"}

# Curated additions that ontologies phrase awkwardly or miss (AD-specific).
CURATED = {
    "biomarker": {
        "Amyloid-beta 42": ["amyloid-beta 42", "abeta42", "aβ42", "amyloid beta 42", "a-beta 42"],
        "p-tau181": ["p-tau181", "ptau181", "phosphorylated tau 181", "p-tau 181"],
        "p-tau217": ["p-tau217", "ptau217", "phosphorylated tau 217", "p-tau 217"],
        "Amyloid PET": ["amyloid pet", "amyloid-pet", "pib-pet", "florbetapir", "flutemetamol"],
        "Tau PET": ["tau pet", "tau-pet", "flortaucipir"],
        "Neurofilament light": ["neurofilament light", "nfl", "neurofilament light chain"],
        "GFAP": ["gfap", "glial fibrillary acidic protein"],
        "Hippocampal atrophy": ["hippocampal atrophy", "hippocampal volume", "medial temporal atrophy"],
        "MMSE": ["mmse", "mini-mental state examination", "mini mental state"],
        "MoCA": ["moca", "montreal cognitive assessment"],
        "CDR": ["cdr", "clinical dementia rating"],
    },
    "stage": {
        "Mild cognitive impairment": ["mild cognitive impairment", "mci", "mci due to ad"],
        "Preclinical AD": ["preclinical ad", "preclinical alzheimer", "preclinical alzheimer's"],
        "Subjective cognitive decline": ["subjective cognitive decline", "scd"],
    },
    "treatment": {
        "Lecanemab": ["lecanemab", "leqembi"],
        "Donanemab": ["donanemab", "kisunla"],
        "Aducanumab": ["aducanumab", "aduhelm"],
        "Donepezil": ["donepezil", "aricept"],
        "Memantine": ["memantine", "namenda"],
        "Rivastigmine": ["rivastigmine", "exelon"],
        "Galantamine": ["galantamine", "razadyne"],
    },
    "outcome": {
        "ARIA-E": ["aria-e", "amyloid-related imaging abnormalities edema", "aria edema"],
        "ARIA-H": ["aria-h", "amyloid-related imaging abnormalities hemorrhage", "aria microhemorrhage"],
        "ARIA": ["aria", "amyloid-related imaging abnormalities"],
        "Amyloid clearance": ["amyloid clearance", "amyloid removal", "plaque clearance"],
        "Cognitive decline": ["cognitive decline", "cognitive deterioration", "cognitive worsening"],
        "Disease progression": ["disease progression", "clinical progression", "progression to dementia"],
        "Mortality": ["mortality", "death", "survival"],
    },
}

# Layer priority: an entity claimed by an earlier layer is not re-used by a later one.
LAYER_PRIORITY = ["gene", "stage", "treatment", "biomarker", "outcome"]

# Tokens too generic to be safe entity surface forms.
STOP = {
    "disease", "syndrome", "disorder", "disorders", "diseases", "imaging", "test",
    "tests", "marker", "markers", "protein", "proteins", "gene", "genes", "cell",
    "cells", "factor", "level", "levels", "score", "index", "type", "group",
    "analysis", "study", "patient", "patients", "human", "mouse", "model", "age",
    "time", "risk", "care", "death", "rate", "function", "activity", "response",
    "treatment", "therapy", "drug", "drugs", "agent", "agents", "disease,",
    "and", "the", "of", "in", "with", "for", "to", "or", "by", "a", "an",
}


def http_get(url: str, retries: int = 4, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "AlzGraph/1.0"})
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception:  # noqa: BLE001
            if attempt == retries:
                raise
            time.sleep(min(20, 2 ** attempt))
    raise RuntimeError("unreachable")


def cached(src_dir: Path, name: str, url: str) -> bytes:
    p = src_dir / name
    if p.exists() and p.stat().st_size > 0:
        return p.read_bytes()
    print(f"  downloading {name} ...", flush=True)
    data = http_get(url)
    p.write_bytes(data)
    return data


def sparql(query: str, retries: int = 4) -> list[dict]:
    params = {"query": query, "format": "JSON", "inference": "true"}
    url = f"{MESH_SPARQL}?{urllib.parse.urlencode(params)}"
    raw = http_get(url, retries=retries, timeout=120)
    return json.loads(raw)["results"]["bindings"]


def deinvert(label: str) -> list[str]:
    """MeSH inverts labels ('Dementia, Vascular'); return natural + inverted forms."""
    forms = [label]
    if ", " in label:
        parts = [p.strip() for p in label.split(",")]
        forms.append(" ".join(reversed(parts)))
    return forms


def good_surface(s: str) -> bool:
    s = s.strip()
    if len(s) < 3:
        return False
    if s.lower() in STOP:
        return False
    if not re.search(r"[A-Za-z]", s):
        return False
    return True


# ----------------------------------------------------------------------- genes
def harvest_genes(src_dir: Path) -> dict:
    data = cached(src_dir, "hgnc_complete_set.txt", HGNC_URL).decode("utf-8", "replace")
    lines = data.splitlines()
    header = lines[0].split("\t")
    idx = {c: i for i, c in enumerate(header)}
    out: dict = {}
    for line in lines[1:]:
        f = line.split("\t")
        if len(f) <= idx["symbol"]:
            continue
        if f[idx["locus_group"]] != "protein-coding gene":
            continue
        if f[idx["status"]] != "Approved":
            continue
        symbol = f[idx["symbol"]].strip()
        if len(symbol) < 3:  # 1-2 char symbols are too ambiguous
            continue
        cs = {symbol}
        for col in ("alias_symbol", "prev_symbol"):
            raw = f[idx[col]] if idx[col] < len(f) else ""
            for a in raw.split("|"):
                a = a.strip().strip('"')
                if len(a) >= 3:
                    cs.add(a)
        out[symbol] = {"layer": "gene", "cs": sorted(cs), "ci": []}
    return out


# ------------------------------------------------------------------- mesh trees
def harvest_mesh_trees(trees: list[str], layer: str) -> dict:
    out: dict = {}
    for prefix in trees:
        q = f"""PREFIX meshv: <http://id.nlm.nih.gov/mesh/vocab#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?dl ?tl WHERE {{
  ?d a meshv:TopicalDescriptor ; meshv:treeNumber ?tn ; rdfs:label ?dl .
  FILTER(STRSTARTS(STR(?tn), "http://id.nlm.nih.gov/mesh/{prefix}"))
  OPTIONAL {{ ?d meshv:concept ?c . ?c meshv:term ?t .
             {{ ?t meshv:prefLabel ?tl }} UNION {{ ?t meshv:altLabel ?tl }} }}
}}"""
        try:
            rows = sparql(q)
        except Exception as e:  # noqa: BLE001
            print(f"  [mesh] tree {prefix} failed: {e}", file=sys.stderr)
            continue
        syn = defaultdict(set)
        for r in rows:
            canon = r["dl"]["value"]
            syn[canon].add(canon)
            if "tl" in r:
                syn[canon].add(r["tl"]["value"])
        for canon, labels in syn.items():
            ci = set()
            for lab in labels:
                for form in deinvert(lab):
                    if good_surface(form):
                        ci.add(form)
            if ci:
                out.setdefault(canon, {"layer": layer, "cs": [], "ci": []})
                out[canon]["ci"] = sorted(set(out[canon]["ci"]) | ci)
        print(f"  [mesh:{layer}] {prefix}: {len(syn)} descriptors (cumulative {len(out)})", flush=True)
        time.sleep(0.3)
    return out


def _parse_obo(text: str) -> dict:
    """Parse an OBO file into {id: {name, syn:[], is_a:[], has_role:[]}}."""
    terms: dict = {}
    cur = None

    def flush():
        if cur and cur.get("id"):
            terms[cur["id"]] = cur

    for line in text.splitlines():
        if line == "[Term]":
            flush()
            cur = {"id": None, "name": None, "syn": [], "is_a": [], "has_role": []}
            continue
        if line.startswith("[") and line != "[Term]":
            flush()
            cur = None
            continue
        if cur is None:
            continue
        if line.startswith("id: "):
            cur["id"] = line[4:].strip()
        elif line.startswith("name: "):
            cur["name"] = line[6:].strip()
        elif line.startswith("synonym: "):
            m = re.search(r'"([^"]+)"', line)
            if m:
                cur["syn"].append(m.group(1))
        elif line.startswith("is_a: "):
            cur["is_a"].append(line[6:].split("!")[0].strip())
        elif line.startswith("relationship: "):
            parts = line.split()
            # ChEBI encodes "has role" as the RO:0000087 relation; OBO uses has_role.
            if len(parts) >= 3 and parts[1] in ("has_role", "RO:0000087"):
                cur["has_role"].append(parts[2].strip())
    flush()
    return terms


def _descendants(terms: dict, roots: set, edge: str = "is_a") -> set:
    children = defaultdict(list)
    for tid, t in terms.items():
        for parent in t.get(edge, []):
            children[parent].append(tid)
    keep, stack = set(), list(roots)
    while stack:
        n = stack.pop()
        if n in keep:
            continue
        keep.add(n)
        stack.extend(children.get(n, []))
    return keep


def harvest_chebi_drugs(src_dir: Path) -> dict:
    raw = cached(src_dir, "chebi.obo.gz", CHEBI_URL)
    text = gzip.decompress(raw).decode("utf-8", "replace")
    terms = _parse_obo(text)
    drug_roles = _descendants(terms, CHEBI_DRUG_ROLES, edge="is_a")
    out: dict = {}
    for tid, t in terms.items():
        if not t["name"]:
            continue
        if not any(r in drug_roles for r in t["has_role"]):
            continue
        ci = {s for s in [t["name"], *t["syn"]] if good_surface(s)}
        if ci:
            out[t["name"]] = {"layer": "treatment", "cs": [], "ci": sorted(ci)}
    print(f"  [chebi:treatment] drug/pharmaceutical chemicals: {len(out)} "
          f"(from {len(drug_roles)} drug roles)", flush=True)
    return out


# -------------------------------------------------------------------------- hpo
def harvest_hpo(src_dir: Path) -> dict:
    data = cached(src_dir, "hp.obo", HPO_URL).decode("utf-8", "replace")
    terms = _parse_obo(data)
    keep = _descendants(terms, set(HPO_OUTCOME_ROOTS), edge="is_a")
    out: dict = {}
    for tid in keep:
        t = terms.get(tid)
        if not t or not t["name"]:
            continue
        ci = {s for s in [t["name"], *t["syn"]] if good_surface(s)}
        if ci:
            out[t["name"]] = {"layer": "outcome", "cs": [], "ci": sorted(ci)}
    print(f"  [hpo:outcome] phenotype descendants: {len(out)}", flush=True)
    return out


# ------------------------------------------------------------------------- main
def resolve_conflicts(layers: dict) -> dict:
    """Merge per-layer dicts into one, enforcing single-layer-per-entity by priority,
    and dropping any surface form already claimed (case-insensitively) by a higher layer."""
    final: dict = {}
    claimed_ci: dict = {}   # casefolded surface -> canonical
    claimed_cs: dict = {}
    for layer in LAYER_PRIORITY:
        for canon, spec in layers.get(layer, {}).items():
            if canon in final:
                continue
            cs = [s for s in spec.get("cs", []) if s not in claimed_cs]
            ci = [s for s in spec.get("ci", []) if s.casefold() not in claimed_ci]
            if not cs and not ci:
                continue
            final[canon] = {"layer": layer, "cs": sorted(set(cs)), "ci": sorted(set(ci))}
            for s in cs:
                claimed_cs[s] = canon
            for s in ci:
                claimed_ci[s.casefold()] = canon
    return final


def main() -> None:
    ap = argparse.ArgumentParser(description="Harvest the AlzKG entity vocabulary from public ontologies.")
    ap.add_argument("--out", default="data/lexicon/lexicon_full.json")
    ap.add_argument("--src_dir", default="data/lexicon/sources")
    ap.add_argument("--skip", default="", help="comma list of layers to skip (gene,stage,biomarker,treatment,outcome)")
    args = ap.parse_args()
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    src_dir = Path(args.src_dir)
    src_dir.mkdir(parents=True, exist_ok=True)

    layers: dict = {}
    if "gene" not in skip:
        print("[gene] HGNC ...", flush=True)
        layers["gene"] = harvest_genes(src_dir)
    if "stage" not in skip:
        print("[stage] MeSH dementia/neurodegeneration ...", flush=True)
        layers["stage"] = harvest_mesh_trees(MESH_STAGE_TREES, "stage")
    if "biomarker" not in skip:
        print("[biomarker] MeSH diagnostic techniques ...", flush=True)
        layers["biomarker"] = harvest_mesh_trees(MESH_BIOMARKER_TREES, "biomarker")
    if "treatment" not in skip:
        print("[treatment] ChEBI drug/pharmaceutical chemicals ...", flush=True)
        layers["treatment"] = harvest_chebi_drugs(src_dir)
    if "outcome" not in skip:
        print("[outcome] HPO nervous-system phenotypes ...", flush=True)
        layers["outcome"] = harvest_hpo(src_dir)

    # curated AD-specific entries go into their layer dicts before conflict resolution
    for layer, entries in CURATED.items():
        layers.setdefault(layer, {})
        for canon, ci in entries.items():
            layers[layer].setdefault(canon, {"layer": layer, "cs": [], "ci": []})
            layers[layer][canon]["ci"] = sorted(set(layers[layer][canon]["ci"]) | set(ci))

    final = resolve_conflicts(layers)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(final, ensure_ascii=False, indent=1))

    from collections import Counter
    per_layer = Counter(v["layer"] for v in final.values())
    n_surfaces = sum(len(v["ci"]) + len(v["cs"]) for v in final.values())
    print("\n==== AlzKG ontology vocabulary ====")
    print(f"  entities      : {len(final)}")
    print(f"  surface forms : {n_surfaces}")
    print(f"  per layer     : {dict(per_layer)}")
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
