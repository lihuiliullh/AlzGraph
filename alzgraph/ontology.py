"""Curated Alzheimer's disease ontology seed for AlzKG.

This module defines the AlzKG five-layer schema, a curated entity vocabulary with
source provenance, the cross-layer relation types, and a curated edge table that
encodes well-established Alzheimer's disease knowledge from public ontologies and
clinical guidelines.

Layers (mirroring a gene -> mechanism -> disease -> therapy -> outcome reasoning
chain):
  - gene       : risk / causal genes (OMIM, ADSP, GWAS Catalog)
  - biomarker  : ATN fluid + imaging markers and cognitive instruments (NIA-AA, MeSH, HPO)
  - stage      : disease stages and AD syndromes (NIA-AA, MeSH, UMLS)
  - treatment  : pharmacological and non-pharmacological interventions (ChEBI, AAN/AA guidelines)
  - outcome    : clinical endpoints and adverse events (HPO, MeSH)

Edge evidence tiers:
  3 = guideline / landmark trial / autosomal-dominant causal
  2 = established association
  1 = emerging association
"""

from typing import Dict, List, Tuple

# --------------------------------------------------------------------- layers
LAYERS: Dict[str, List[str]] = {
    "gene": [
        "APP", "PSEN1", "PSEN2", "APOE", "TREM2", "ABCA7", "CLU", "CR1",
        "BIN1", "PICALM", "SORL1", "MAPT", "PLCG2", "MS4A6A", "CD33",
        "SPI1", "INPP5D",
    ],
    "biomarker": [
        "CSF Abeta42", "CSF Abeta42/40 ratio", "Amyloid PET", "Plasma Abeta42/40",
        "CSF p-tau181", "CSF p-tau217", "Plasma p-tau217", "Plasma p-tau181",
        "Tau PET", "CSF total tau", "Hippocampal atrophy", "FDG-PET hypometabolism",
        "Plasma NfL", "Plasma GFAP", "MMSE", "MoCA", "CDR", "ADAS-Cog",
    ],
    "stage": [
        "Preclinical AD", "MCI due to AD", "AD dementia", "Mild AD dementia",
        "Moderate AD dementia", "Severe AD dementia", "Early-onset AD",
        "Late-onset AD", "Posterior cortical atrophy", "Logopenic variant PPA",
    ],
    "treatment": [
        "Donepezil", "Rivastigmine", "Galantamine", "Memantine", "Lecanemab",
        "Donanemab", "Aducanumab", "Cognitive stimulation therapy",
        "Physical exercise", "Cardiovascular risk management",
    ],
    "outcome": [
        "Cognitive decline", "Functional decline", "Progression to AD dementia",
        "ARIA-E", "ARIA-H", "Amyloid plaque reduction", "Slowing of cognitive decline",
        "Symptomatic benefit", "Nausea", "Bradycardia", "Mortality", "Hospitalization",
    ],
}

# ----------------------------------------------------------------- provenance
SOURCES: Dict[str, str] = {
    # genes
    "APP": "OMIM", "PSEN1": "OMIM", "PSEN2": "OMIM", "APOE": "OMIM",
    "TREM2": "ADSP", "ABCA7": "GWAS_Catalog", "CLU": "GWAS_Catalog",
    "CR1": "GWAS_Catalog", "BIN1": "GWAS_Catalog", "PICALM": "GWAS_Catalog",
    "SORL1": "ADSP", "MAPT": "OMIM", "PLCG2": "ADSP", "MS4A6A": "GWAS_Catalog",
    "CD33": "GWAS_Catalog", "SPI1": "GWAS_Catalog", "INPP5D": "GWAS_Catalog",
    # biomarkers
    "CSF Abeta42": "NIA_AA", "CSF Abeta42/40 ratio": "NIA_AA", "Amyloid PET": "NIA_AA",
    "Plasma Abeta42/40": "NIA_AA", "CSF p-tau181": "NIA_AA", "CSF p-tau217": "NIA_AA",
    "Plasma p-tau217": "NIA_AA", "Plasma p-tau181": "NIA_AA", "Tau PET": "NIA_AA",
    "CSF total tau": "NIA_AA", "Hippocampal atrophy": "MeSH", "FDG-PET hypometabolism": "MeSH",
    "Plasma NfL": "NIA_AA", "Plasma GFAP": "NIA_AA", "MMSE": "MeSH", "MoCA": "MeSH",
    "CDR": "MeSH", "ADAS-Cog": "MeSH",
    # stages
    "Preclinical AD": "NIA_AA", "MCI due to AD": "NIA_AA", "AD dementia": "NIA_AA",
    "Mild AD dementia": "NIA_AA", "Moderate AD dementia": "NIA_AA",
    "Severe AD dementia": "NIA_AA", "Early-onset AD": "OMIM", "Late-onset AD": "OMIM",
    "Posterior cortical atrophy": "UMLS", "Logopenic variant PPA": "UMLS",
    # treatments
    "Donepezil": "ChEBI", "Rivastigmine": "ChEBI", "Galantamine": "ChEBI",
    "Memantine": "ChEBI", "Lecanemab": "AAN_AA_AUC", "Donanemab": "AAN_AA_AUC",
    "Aducanumab": "AAN_AA_AUC", "Cognitive stimulation therapy": "AAN_guideline",
    "Physical exercise": "AAN_guideline", "Cardiovascular risk management": "AAN_guideline",
    # outcomes
    "Cognitive decline": "HPO", "Functional decline": "HPO",
    "Progression to AD dementia": "MeSH", "ARIA-E": "MeSH", "ARIA-H": "MeSH",
    "Amyloid plaque reduction": "MeSH", "Slowing of cognitive decline": "MeSH",
    "Symptomatic benefit": "MeSH", "Nausea": "HPO", "Bradycardia": "HPO",
    "Mortality": "MeSH", "Hospitalization": "MeSH",
}

# ------------------------------------------------------ relation type catalog
RELATION_HINTS: Dict[Tuple[str, str], str] = {
    ("gene", "stage"): "risk_gene_for",
    ("gene", "biomarker"): "modulates_biomarker",
    ("gene", "treatment"): "pharmacogenomic_consideration",
    ("stage", "biomarker"): "characterized_by_biomarker",
    ("stage", "treatment"): "treated_with",
    ("stage", "outcome"): "associated_outcome",
    ("biomarker", "outcome"): "predicts_outcome",
    ("treatment", "outcome"): "has_outcome",
}

# --------------------------------------------------------- curated edge table
# (head, relation, tail, evidence_tier, evidence_note)
CURATED_EDGES: List[Tuple[str, str, str, int, str]] = [
    # --- causal / risk genetics (gene -> stage) -------------------------------
    ("APP", "causal_gene_for", "Early-onset AD", 3, "Autosomal-dominant amyloid precursor protein mutations cause early-onset AD."),
    ("PSEN1", "causal_gene_for", "Early-onset AD", 3, "PSEN1 mutations are the most common cause of autosomal-dominant early-onset AD."),
    ("PSEN2", "causal_gene_for", "Early-onset AD", 3, "PSEN2 mutations cause autosomal-dominant early-onset AD."),
    ("APOE", "risk_gene_for", "Late-onset AD", 3, "APOE e4 is the strongest common genetic risk factor; e2 is protective."),
    ("TREM2", "risk_gene_for", "Late-onset AD", 2, "Rare TREM2 R47H variant confers substantially increased AD risk via microglial dysfunction."),
    ("ABCA7", "risk_gene_for", "Late-onset AD", 2, "ABCA7 loss-of-function variants increase late-onset AD risk."),
    ("SORL1", "risk_gene_for", "Late-onset AD", 2, "SORL1 variants affect APP trafficking and increase AD risk."),
    ("CLU", "risk_gene_for", "Late-onset AD", 2, "CLU (clusterin) is a GWAS-confirmed late-onset AD risk locus."),
    ("CR1", "risk_gene_for", "Late-onset AD", 2, "CR1 complement-pathway locus is associated with late-onset AD."),
    ("BIN1", "risk_gene_for", "Late-onset AD", 2, "BIN1 is the second strongest common late-onset AD risk locus after APOE."),
    ("PICALM", "risk_gene_for", "Late-onset AD", 2, "PICALM modulates APP processing and is a confirmed AD risk locus."),
    ("MS4A6A", "risk_gene_for", "Late-onset AD", 1, "MS4A gene cluster is associated with late-onset AD risk."),
    ("CD33", "risk_gene_for", "Late-onset AD", 1, "CD33 microglial locus is associated with late-onset AD."),
    ("SPI1", "risk_gene_for", "Late-onset AD", 1, "SPI1 (PU.1) regulates myeloid AD risk genes."),
    ("INPP5D", "risk_gene_for", "Late-onset AD", 1, "INPP5D (SHIP1) microglial locus is associated with AD."),
    ("PLCG2", "protective_gene_for", "Late-onset AD", 2, "PLCG2 P522R is a protective microglial variant."),

    # --- gene -> biomarker mechanisms ----------------------------------------
    ("APP", "modulates_biomarker", "CSF Abeta42", 3, "APP mutations increase amyloidogenic Abeta42 production, lowering CSF Abeta42."),
    ("APP", "modulates_biomarker", "Amyloid PET", 3, "APP mutations drive fibrillar amyloid deposition detected by amyloid PET."),
    ("PSEN1", "modulates_biomarker", "CSF Abeta42", 3, "PSEN1 alters gamma-secretase cleavage, raising Abeta42/40 aggregation propensity."),
    ("PSEN1", "modulates_biomarker", "Amyloid PET", 3, "PSEN1 carriers show early amyloid PET positivity."),
    ("PSEN2", "modulates_biomarker", "CSF Abeta42", 2, "PSEN2 alters gamma-secretase cleavage and amyloid processing."),
    ("APOE", "modulates_biomarker", "Amyloid PET", 3, "APOE e4 accelerates fibrillar amyloid deposition and amyloid PET positivity."),
    ("APOE", "modulates_biomarker", "CSF Abeta42", 2, "APOE e4 lowers CSF Abeta42 reflecting amyloid sequestration."),
    ("MAPT", "modulates_biomarker", "Tau PET", 2, "MAPT influences tau aggregation measured by tau PET."),
    ("MAPT", "modulates_biomarker", "CSF p-tau181", 1, "MAPT haplotypes modulate tau phosphorylation markers."),
    ("TREM2", "modulates_biomarker", "Plasma GFAP", 1, "TREM2 microglial dysfunction is reflected in astro-microglial activation markers."),

    # --- stage -> biomarker signatures (ATN) ---------------------------------
    ("Preclinical AD", "characterized_by_biomarker", "Amyloid PET", 3, "Preclinical AD is defined by amyloid positivity with normal cognition."),
    ("Preclinical AD", "characterized_by_biomarker", "CSF Abeta42", 2, "Low CSF Abeta42 marks the earliest amyloid (A+) stage."),
    ("MCI due to AD", "characterized_by_biomarker", "Amyloid PET", 3, "MCI due to AD requires biomarker evidence of amyloid pathology."),
    ("MCI due to AD", "characterized_by_biomarker", "CSF p-tau217", 2, "p-tau217 elevation supports AD as the cause of MCI."),
    ("MCI due to AD", "characterized_by_biomarker", "Hippocampal atrophy", 2, "Medial temporal atrophy supports neurodegeneration in prodromal AD."),
    ("AD dementia", "characterized_by_biomarker", "Amyloid PET", 3, "AD dementia shows cortical amyloid PET positivity."),
    ("AD dementia", "characterized_by_biomarker", "CSF Abeta42", 3, "Low CSF Abeta42 is a core AD biomarker."),
    ("AD dementia", "characterized_by_biomarker", "CSF p-tau181", 3, "Elevated CSF p-tau181 is a core AD tau biomarker."),
    ("AD dementia", "characterized_by_biomarker", "CSF p-tau217", 2, "p-tau217 has the strongest separation of AD from non-AD."),
    ("AD dementia", "characterized_by_biomarker", "Tau PET", 2, "Tau PET tracks neurofibrillary tangle burden in AD dementia."),
    ("AD dementia", "characterized_by_biomarker", "Hippocampal atrophy", 3, "Hippocampal and medial temporal atrophy are hallmark MRI findings."),
    ("AD dementia", "characterized_by_biomarker", "FDG-PET hypometabolism", 2, "Temporoparietal FDG-PET hypometabolism is typical of AD."),
    ("Mild AD dementia", "characterized_by_biomarker", "MMSE", 2, "Mild AD dementia typically corresponds to MMSE ~21-26."),
    ("Mild AD dementia", "characterized_by_biomarker", "CDR", 2, "Mild AD dementia corresponds to CDR 1."),
    ("Moderate AD dementia", "characterized_by_biomarker", "MMSE", 2, "Moderate AD dementia typically corresponds to MMSE ~10-20."),
    ("Moderate AD dementia", "characterized_by_biomarker", "CDR", 2, "Moderate AD dementia corresponds to CDR 2."),
    ("Severe AD dementia", "characterized_by_biomarker", "CDR", 2, "Severe AD dementia corresponds to CDR 3."),
    ("Severe AD dementia", "characterized_by_biomarker", "MMSE", 2, "Severe AD dementia typically corresponds to MMSE <10."),
    ("Posterior cortical atrophy", "characterized_by_biomarker", "FDG-PET hypometabolism", 2, "PCA shows occipito-parietal hypometabolism with underlying AD pathology."),
    ("Posterior cortical atrophy", "characterized_by_biomarker", "Amyloid PET", 2, "Most PCA cases are amyloid-positive (atypical AD)."),
    ("Logopenic variant PPA", "characterized_by_biomarker", "Amyloid PET", 2, "Logopenic variant PPA is most often an atypical amyloid-positive AD presentation."),

    # --- biomarker -> outcome prognostics ------------------------------------
    ("Amyloid PET", "predicts_outcome", "Progression to AD dementia", 2, "Amyloid positivity in MCI predicts progression to AD dementia."),
    ("CSF p-tau217", "predicts_outcome", "Progression to AD dementia", 2, "p-tau217 strongly predicts progression and tau accumulation."),
    ("Plasma p-tau217", "predicts_outcome", "Progression to AD dementia", 2, "Plasma p-tau217 predicts incident AD dementia in at-risk individuals."),
    ("CSF p-tau181", "predicts_outcome", "Cognitive decline", 2, "Elevated p-tau181 predicts faster cognitive decline."),
    ("Hippocampal atrophy", "predicts_outcome", "Cognitive decline", 2, "Hippocampal atrophy rate predicts cognitive trajectory."),
    ("Plasma NfL", "predicts_outcome", "Cognitive decline", 2, "Plasma NfL indexes neurodegeneration and predicts decline."),
    ("Plasma GFAP", "predicts_outcome", "Progression to AD dementia", 1, "Plasma GFAP (astrocytic) predicts progression along the AD continuum."),
    ("FDG-PET hypometabolism", "predicts_outcome", "Functional decline", 2, "Temporoparietal hypometabolism predicts functional decline."),
    ("MMSE", "predicts_outcome", "Functional decline", 1, "Lower MMSE tracks loss of instrumental function."),

    # --- stage -> treatment (guideline therapy) ------------------------------
    ("Mild AD dementia", "treated_with", "Donepezil", 3, "Cholinesterase inhibitors are first-line for mild-moderate AD dementia."),
    ("Mild AD dementia", "treated_with", "Rivastigmine", 3, "Rivastigmine is indicated for mild-moderate AD dementia."),
    ("Mild AD dementia", "treated_with", "Galantamine", 3, "Galantamine is indicated for mild-moderate AD dementia."),
    ("Moderate AD dementia", "treated_with", "Donepezil", 3, "Donepezil is indicated across mild-to-severe AD dementia."),
    ("Moderate AD dementia", "treated_with", "Memantine", 3, "Memantine is indicated for moderate-to-severe AD dementia."),
    ("Severe AD dementia", "treated_with", "Memantine", 3, "Memantine is indicated for moderate-to-severe AD dementia."),
    ("Severe AD dementia", "treated_with", "Donepezil", 3, "Donepezil is approved for severe AD dementia."),
    ("MCI due to AD", "treated_with", "Lecanemab", 3, "Lecanemab is indicated for amyloid-confirmed MCI due to AD and mild AD dementia."),
    ("MCI due to AD", "treated_with", "Donanemab", 3, "Donanemab is indicated for early symptomatic amyloid-positive AD."),
    ("Mild AD dementia", "treated_with", "Lecanemab", 3, "Lecanemab targets early symptomatic AD with confirmed amyloid."),
    ("Mild AD dementia", "treated_with", "Donanemab", 3, "Donanemab targets early symptomatic AD with confirmed amyloid."),
    ("AD dementia", "treated_with", "Cognitive stimulation therapy", 2, "Cognitive stimulation therapy is recommended for mild-moderate dementia."),
    ("AD dementia", "treated_with", "Physical exercise", 1, "Physical activity is recommended as supportive management."),
    ("Late-onset AD", "treated_with", "Cardiovascular risk management", 2, "Midlife vascular risk control reduces dementia risk."),

    # --- gene -> treatment pharmacogenomics (ARIA risk) ----------------------
    ("APOE", "pharmacogenomic_consideration", "Lecanemab", 3, "APOE e4 homozygotes have markedly higher ARIA risk; genotype-informed counseling and MRI monitoring are advised."),
    ("APOE", "pharmacogenomic_consideration", "Donanemab", 3, "APOE e4 status stratifies ARIA risk during donanemab therapy."),
    ("APOE", "pharmacogenomic_consideration", "Aducanumab", 2, "APOE e4 carriers show higher ARIA-E incidence with aducanumab."),

    # --- treatment -> outcome (efficacy + adverse) ---------------------------
    ("Lecanemab", "has_outcome", "Amyloid plaque reduction", 3, "Lecanemab produces marked amyloid PET reduction (CLARITY-AD)."),
    ("Lecanemab", "has_outcome", "Slowing of cognitive decline", 3, "Lecanemab slowed CDR-SB decline by ~27% over 18 months."),
    ("Lecanemab", "has_outcome", "ARIA-E", 3, "ARIA-E (edema) is the principal lecanemab safety risk, elevated in e4 homozygotes."),
    ("Lecanemab", "has_outcome", "ARIA-H", 2, "ARIA-H (microhemorrhage/siderosis) occurs with lecanemab."),
    ("Donanemab", "has_outcome", "Amyloid plaque reduction", 3, "Donanemab produces rapid amyloid clearance (TRAILBLAZER-ALZ 2)."),
    ("Donanemab", "has_outcome", "Slowing of cognitive decline", 3, "Donanemab slowed iADRS decline, greatest in low-tau participants."),
    ("Donanemab", "has_outcome", "ARIA-E", 3, "ARIA-E is the principal donanemab safety risk."),
    ("Donanemab", "has_outcome", "ARIA-H", 2, "ARIA-H occurs with donanemab and requires MRI monitoring."),
    ("Aducanumab", "has_outcome", "Amyloid plaque reduction", 2, "Aducanumab reduces amyloid PET dose-dependently."),
    ("Aducanumab", "has_outcome", "ARIA-E", 3, "ARIA-E was the most common aducanumab adverse event."),
    ("Aducanumab", "has_outcome", "ARIA-H", 2, "ARIA-H is observed with aducanumab."),
    ("Donepezil", "has_outcome", "Symptomatic benefit", 3, "Donepezil yields modest symptomatic cognitive benefit."),
    ("Donepezil", "has_outcome", "Nausea", 2, "Cholinergic nausea/GI effects are common with donepezil."),
    ("Donepezil", "has_outcome", "Bradycardia", 2, "Donepezil can cause bradycardia/syncope via cholinergic tone."),
    ("Rivastigmine", "has_outcome", "Symptomatic benefit", 3, "Rivastigmine yields modest symptomatic benefit."),
    ("Rivastigmine", "has_outcome", "Nausea", 2, "GI adverse effects are common; patch reduces them."),
    ("Galantamine", "has_outcome", "Symptomatic benefit", 3, "Galantamine yields modest symptomatic benefit."),
    ("Galantamine", "has_outcome", "Nausea", 2, "GI adverse effects are common with galantamine."),
    ("Memantine", "has_outcome", "Symptomatic benefit", 3, "Memantine provides modest benefit in moderate-severe AD."),
    ("Cognitive stimulation therapy", "has_outcome", "Symptomatic benefit", 2, "Cognitive stimulation improves cognition/quality of life in mild-moderate dementia."),
    ("Physical exercise", "has_outcome", "Slowing of cognitive decline", 1, "Exercise has modest supportive cognitive benefits."),
    ("Cardiovascular risk management", "has_outcome", "Slowing of cognitive decline", 2, "Vascular risk control slows decline and lowers dementia incidence."),

    # --- additional biomarker coverage (plasma + ratios + instruments) -------
    ("AD dementia", "characterized_by_biomarker", "CSF Abeta42/40 ratio", 3, "The CSF Abeta42/40 ratio improves amyloid classification over Abeta42 alone."),
    ("MCI due to AD", "characterized_by_biomarker", "Plasma Abeta42/40", 2, "A low plasma Abeta42/40 ratio is an accessible amyloid screening marker."),
    ("MCI due to AD", "characterized_by_biomarker", "Plasma p-tau181", 2, "Plasma p-tau181 rises early along the AD continuum."),
    ("AD dementia", "characterized_by_biomarker", "CSF total tau", 2, "Elevated CSF total tau indexes neuronal injury (N) in AD."),
    ("MCI due to AD", "characterized_by_biomarker", "MoCA", 2, "The MoCA is sensitive to amnestic MCI."),
    ("AD dementia", "characterized_by_biomarker", "ADAS-Cog", 2, "ADAS-Cog is the standard cognitive endpoint in AD trials."),
    ("Plasma p-tau181", "predicts_outcome", "Cognitive decline", 2, "Plasma p-tau181 predicts subsequent cognitive decline."),
    ("CSF total tau", "predicts_outcome", "Cognitive decline", 1, "High total tau is associated with faster decline."),

    # --- disease-stage clinical outcomes (stage -> outcome) ------------------
    ("Severe AD dementia", "associated_outcome", "Mortality", 2, "Severe AD dementia carries high short-term mortality."),
    ("Severe AD dementia", "associated_outcome", "Hospitalization", 2, "Advanced AD is associated with frequent hospitalization."),
    ("Moderate AD dementia", "associated_outcome", "Hospitalization", 1, "Behavioral and medical complications drive hospitalization in moderate AD."),
    ("AD dementia", "associated_outcome", "Functional decline", 2, "Progressive loss of activities of daily living defines AD dementia course."),
]


def layer_of(entity: str) -> str:
    for layer, members in LAYERS.items():
        if entity in members:
            return layer
    return "unknown"


def all_entities() -> List[str]:
    return [e for members in LAYERS.values() for e in members]
