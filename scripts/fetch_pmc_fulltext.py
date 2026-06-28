"""Fetch PMC open-access FULL-TEXT for the AlzGraph corpus (EpiGraph procedure).

EpiGraph mines relations from *full-text* articles (PubMed/PMC), not abstracts.
This script reproduces that source step:

  1. map each cached PMID -> PMCID via the NCBI ID Converter (batched);
  2. efetch the PMC full-text XML (``db=pmc``) for the open-access subset;
  3. segment the article ``<body>`` into sentences and keep the *candidate
     sentences* that contain at least two AlzKG entity mentions (the only
     sentences the rule-based / LLM relation extractor can use).

Only candidate sentences are cached (``data/corpus/fulltext_sentences.jsonl``)
so the on-disk footprint stays small while remaining fully re-processable by
``scripts/build_kg_from_fulltext.py``. Standard-library only; NCBI etiquette
(tool+email, <=3 req/s, retries). Pass ``--api_key`` to raise the rate limit.
"""

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alzgraph.lexicon import detect  # noqa: E402

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
IDCONV = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
TOOL = "AlzGraph"
EMAIL = "lihui.liu.mailbox@gmail.com"

# Drop reference lists / boilerplate sections from the body before sentence split.
_DROP_TAGS = {"ref-list", "back", "fn-group", "table-wrap", "fig"}
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def _get(url: str, retries: int = 4, timeout: int = 60) -> bytes:
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except Exception:  # noqa: BLE001
            if attempt == retries:
                raise
            time.sleep(min(20, 2 ** attempt))
    raise RuntimeError("unreachable")


def pmid_to_pmcid(pmids: list[str], api_key: str | None, batch: int = 200, sleep: float = 0.35) -> dict:
    mapping: dict[str, str] = {}
    for start in range(0, len(pmids), batch):
        chunk = pmids[start : start + batch]
        params = {"ids": ",".join(chunk), "format": "json", "tool": TOOL, "email": EMAIL}
        if api_key:
            params["api_key"] = api_key
        try:
            data = json.loads(_get(f"{IDCONV}?{urllib.parse.urlencode(params)}"))
        except Exception as e:  # noqa: BLE001
            print(f"[idconv] batch {start} failed: {e}", file=sys.stderr)
            continue
        for rec in data.get("records", []):
            if rec.get("pmcid"):
                mapping[str(rec.get("pmid"))] = rec["pmcid"]
        print(f"[idconv] {min(start+batch,len(pmids)):6d}/{len(pmids)}  PMCIDs so far: {len(mapping)}")
        sys.stdout.flush()
        time.sleep(sleep)
    return mapping


def _strip(node: ET.Element) -> None:
    for tag in list(_DROP_TAGS):
        for child in node.findall(f".//{tag}"):
            child.clear()


def parse_body_sentences(xml_bytes: bytes) -> list[str]:
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return []
    body = root.find(".//body")
    if body is None:
        return []
    _strip(body)
    text = re.sub(r"\s+", " ", " ".join(body.itertext())).strip()
    if not text:
        return []
    return [s.strip() for s in _SENT_SPLIT.split(text) if len(s.strip()) > 20]


def candidate_sentences(sentences: list[str]) -> list[dict]:
    """Keep sentences mentioning >=2 distinct AlzKG entities; record them."""
    out = []
    for sent in sentences:
        found = detect(sent)
        ents = sorted({(e, layer) for layer, es in found.items() for e in es})
        if len(ents) >= 2:
            out.append({"text": sent, "entities": [{"entity": e, "layer": l} for e, l in ents]})
    return out


def fetch_fulltext(pmcid: str, api_key: str | None) -> list[str]:
    num = pmcid.replace("PMC", "")
    params = {"db": "pmc", "id": num, "retmode": "xml", "tool": TOOL, "email": EMAIL}
    if api_key:
        params["api_key"] = api_key
    return parse_body_sentences(_get(f"{EUTILS}/efetch.fcgi?{urllib.parse.urlencode(params)}"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch PMC OA full text for the AlzGraph corpus.")
    ap.add_argument("--pmids", default="data/corpus/pmids.txt")
    ap.add_argument("--abstracts", default="data/corpus/abstracts.jsonl",
                    help="used for year/title metadata when available")
    ap.add_argument("--out", default="data/corpus/fulltext_sentences.jsonl")
    ap.add_argument("--map_out", default="data/corpus/pmid2pmcid.json")
    ap.add_argument("--max_papers", type=int, default=0, help="0 = all OA papers")
    ap.add_argument("--api_key", default=None)
    ap.add_argument("--sleep", type=float, default=0.35)
    ap.add_argument("--resume", action="store_true", help="skip PMCIDs already in --out")
    args = ap.parse_args()

    pmids = [l.strip() for l in Path(args.pmids).read_text().splitlines() if l.strip()]
    meta = {}
    ap_path = Path(args.abstracts)
    if ap_path.exists():
        for line in ap_path.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                meta[str(r.get("pmid"))] = {
                    "year": r.get("year", ""),
                    "title": r.get("title", ""),
                    "abstract": r.get("abstract", "") or r.get("text", ""),
                }

    map_path = Path(args.map_out)
    if args.resume and map_path.exists():
        mapping = json.loads(map_path.read_text())
        print(f"[idconv] loaded cached map: {len(mapping)} PMCIDs")
    else:
        mapping = pmid_to_pmcid(pmids, args.api_key, sleep=args.sleep)
        map_path.parent.mkdir(parents=True, exist_ok=True)
        map_path.write_text(json.dumps(mapping, indent=2))
    print(f"[idconv] total PMCIDs: {len(mapping)} / {len(pmids)} PMIDs "
          f"({100*len(mapping)/max(len(pmids),1):.0f}% OA-linked)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_pmcids = set()
    mode = "w"
    if args.resume and out_path.exists():
        for line in out_path.open(encoding="utf-8"):
            if line.strip():
                done_pmcids.add(json.loads(line).get("pmcid"))
        mode = "a"
        print(f"[resume] already have {len(done_pmcids)} papers")

    items = [(p, c) for p, c in mapping.items() if c not in done_pmcids]
    if args.max_papers:
        items = items[: args.max_papers]

    n_papers, n_sentences = len(done_pmcids), 0
    with out_path.open(mode, encoding="utf-8") as fh:
        for i, (pmid, pmcid) in enumerate(items, 1):
            try:
                sents = fetch_fulltext(pmcid, args.api_key)
                cands = candidate_sentences(sents)
            except Exception as e:  # noqa: BLE001
                print(f"[efetch] {pmcid} failed: {e}", file=sys.stderr)
                time.sleep(args.sleep)
                continue
            source = "fulltext"
            if not cands:
                # Publisher-restricted papers return no <body>; fall back to the abstract
                # so the paper still contributes evidence (entity-tagged sentences).
                abstract = meta.get(pmid, {}).get("abstract", "")
                if abstract:
                    abs_sents = [s.strip() for s in _SENT_SPLIT.split(abstract) if len(s.strip()) > 20]
                    abs_cands = candidate_sentences(abs_sents)
                    if abs_cands:
                        cands, source = abs_cands, "abstract"
            rec = {
                "pmid": pmid, "pmcid": pmcid,
                "year": meta.get(pmid, {}).get("year", ""),
                "n_body_sentences": len(sents),
                "source": source,
                "candidate_sentences": cands,
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            n_papers += 1
            n_sentences += len(cands)
            if i % 25 == 0 or i == len(items):
                print(f"[efetch] {i:6d}/{len(items)}  papers={n_papers}  cand_sentences={n_sentences}")
                sys.stdout.flush()
            time.sleep(args.sleep)
    print(f"[done] {n_papers} full-text papers, {n_sentences} candidate sentences -> {out_path}")


if __name__ == "__main__":
    main()
