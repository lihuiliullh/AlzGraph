"""Collect a real Alzheimer's disease corpus from PubMed via NCBI E-utilities.

Standard-library only (urllib / xml / json) so it runs without third-party
dependencies. Retrieves AD-anchored, theme-focused literature, then fetches
titles, abstracts, MeSH headings, and year for each PMID and caches them to
``data/corpus/abstracts.jsonl``.

NCBI etiquette: we send ``tool`` and ``email`` and stay under three requests per
second. Provide ``--api_key`` (NCBI) to raise the rate limit.
"""

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
TOOL = "AlzGraph"
EMAIL = "lihui.liu.mailbox@gmail.com"

BASE = '("alzheimer disease"[MeSH Terms] OR alzheimer*[Title/Abstract] OR "mild cognitive impairment"[Title/Abstract])'

THEMES = {
    "genetics": '(APOE[tiab] OR APP[tiab] OR PSEN1[tiab] OR PSEN2[tiab] OR TREM2[tiab] OR SORL1[tiab] OR ABCA7[tiab] OR BIN1[tiab] OR "genome-wide association"[tiab] OR genetic*[tiab])',
    "amyloid_tau": '(amyloid[tiab] OR "amyloid beta"[tiab] OR tau[tiab] OR "p-tau"[tiab] OR "phosphorylated tau"[tiab] OR biomarker*[tiab] OR "amyloid PET"[tiab] OR "tau PET"[tiab] OR "cerebrospinal fluid"[tiab] OR neurofilament[tiab] OR GFAP[tiab])',
    "imaging_staging": '(MRI[tiab] OR hippocamp*[tiab] OR atrophy[tiab] OR "FDG-PET"[tiab] OR MMSE[tiab] OR MoCA[tiab] OR "clinical dementia rating"[tiab] OR preclinical[tiab] OR prodromal[tiab] OR dementia[tiab])',
    "treatment": '(lecanemab[tiab] OR donanemab[tiab] OR aducanumab[tiab] OR donepezil[tiab] OR rivastigmine[tiab] OR galantamine[tiab] OR memantine[tiab] OR cholinesterase[tiab] OR "anti-amyloid"[tiab] OR "monoclonal antibody"[tiab])',
    "outcomes": '(ARIA[tiab] OR "amyloid-related imaging"[tiab] OR microhemorrhage*[tiab] OR "cognitive decline"[tiab] OR progression[tiab] OR "adverse event*"[tiab] OR efficacy[tiab] OR safety[tiab])',
}


def _get(url: str, retries: int = 4) -> bytes:
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=40) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            if attempt == retries:
                raise
            time.sleep(min(20, 2 ** attempt))
    raise RuntimeError("unreachable")


def _post(url: str, params: dict, retries: int = 4) -> bytes:
    data = urllib.parse.urlencode(params).encode()
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, data=data)
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            if attempt == retries:
                raise
            time.sleep(min(20, 2 ** attempt))
    raise RuntimeError("unreachable")


def esearch(term: str, retmax: int, api_key: str | None, mindate: str, maxdate: str) -> list[str]:
    params = {
        "db": "pubmed", "term": term, "retmax": str(retmax), "retmode": "json",
        "datetype": "pdat", "mindate": mindate, "maxdate": maxdate,
        "tool": TOOL, "email": EMAIL,
    }
    if api_key:
        params["api_key"] = api_key
    url = f"{EUTILS}/esearch.fcgi?{urllib.parse.urlencode(params)}"
    data = json.loads(_get(url))
    return data.get("esearchresult", {}).get("idlist", [])


def _text(node) -> str:
    return " ".join("".join(n.itertext()) for n in node) if node is not None else ""


def parse_articles(xml_bytes: bytes) -> list[dict]:
    out = []
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return out
    for art in root.findall(".//PubmedArticle"):
        pmid = art.findtext(".//MedlineCitation/PMID") or ""
        title_node = art.find(".//Article/ArticleTitle")
        title = "".join(title_node.itertext()) if title_node is not None else ""
        abstract = " ".join("".join(a.itertext()) for a in art.findall(".//Abstract/AbstractText"))
        year = art.findtext(".//JournalIssue/PubDate/Year") or art.findtext(".//JournalIssue/PubDate/MedlineDate") or ""
        mesh = [d.text for d in art.findall(".//MeshHeadingList/MeshHeading/DescriptorName") if d.text]
        journal = art.findtext(".//Journal/Title") or ""
        if pmid and abstract.strip():
            out.append({
                "pmid": pmid, "year": str(year)[:4], "journal": journal,
                "title": " ".join(title.split()), "abstract": " ".join(abstract.split()),
                "mesh": mesh,
            })
    return out


def efetch_batch(pmids: list[str], api_key: str | None) -> list[dict]:
    params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml", "rettype": "abstract",
              "tool": TOOL, "email": EMAIL}
    if api_key:
        params["api_key"] = api_key
    return parse_articles(_post(f"{EUTILS}/efetch.fcgi", params))


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch a real AD corpus from PubMed.")
    ap.add_argument("--out", default="data/corpus/abstracts.jsonl")
    ap.add_argument("--per_theme", type=int, default=4000, help="max PMIDs per theme query")
    ap.add_argument("--max_papers", type=int, default=12000, help="cap on abstracts fetched")
    ap.add_argument("--mindate", default="2010")
    ap.add_argument("--maxdate", default="2026")
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--api_key", default=None)
    ap.add_argument("--sleep", type=float, default=0.4)
    args = ap.parse_args()

    # 1) Collect and union PMIDs across AD-anchored theme queries.
    seen: list[str] = []
    seen_set: set[str] = set()
    for name, theme in THEMES.items():
        term = f"{BASE} AND {theme}"
        ids = esearch(term, args.per_theme, args.api_key, args.mindate, args.maxdate)
        new = [i for i in ids if i not in seen_set]
        seen_set.update(new)
        seen.extend(new)
        print(f"[esearch] {name:16s} +{len(new):5d} new  (total {len(seen)})")
        time.sleep(args.sleep)
    pmids = seen[: args.max_papers]
    print(f"[esearch] unique PMIDs: {len(seen)}; fetching abstracts for {len(pmids)}")

    # 2) efetch abstracts in batches, stream to JSONL.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Path(out_path.parent / "pmids.txt").write_text("\n".join(pmids), encoding="utf-8")
    n_written = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for start in range(0, len(pmids), args.batch):
            batch = pmids[start : start + args.batch]
            records = efetch_batch(batch, args.api_key)
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += len(records)
            done = start + len(batch)
            print(f"[efetch] {done:6d}/{len(pmids)}  (+{len(records)} abstracts, {n_written} total)")
            sys.stdout.flush()
            time.sleep(args.sleep)
    print(f"[done] wrote {n_written} abstracts to {out_path}")


if __name__ == "__main__":
    main()
