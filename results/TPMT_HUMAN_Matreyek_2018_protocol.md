# Bench protocol — Thiopurine S-methyltransferase

Assay: `TPMT_HUMAN_Matreyek_2018`
Compiled from: records + gemini-3.5-flash (gaps only)

## Before you start

This protocol contains **8 sourced** values (from a cited record or publication), **10** standard-technique steps and statements, and **4 estimates**.

The 4 estimates are inferences, not measurements. Each is marked, carries its reasoning, and names an experiment that would confirm it. Verify them before committing material:

- Melting temperature (Tm) — confirm by: Differential scanning fluorimetry (DSF) using SYPRO Orange dye on purified recombinant TPMT.
- Isoelectric point (pI) — confirm by: Isoelectric focusing (IEF) gel electrophoresis of cell lysates.
- Intracellular protein half-life (WT) — confirm by: Cycloheximide shut-off time-course assay followed by flow cytometry tracking of GFP fluorescence.
- FACS preparation buffer pH — confirm by: Flow cytometry evaluation of GFP signal retention across a buffer pH gradient ranging from 6.5 to 8.0.

Provenance labels used below:

- **Sourced** — from a cited record or publication
- **Standard technique** — general technique, not specific to this protein
- **ESTIMATE — verify** — inferred, not measured — confirm before relying on it

## Target

Thiopurine S-methyltransferase from Homo sapiens, 245 residues, region mutated 1-245. Fitness is measured by: Protein abundance (FACS sorting for abundance of GFP-fused target).

Function: Catalyzes the S-methylation of thiopurine drugs such as 6-mercaptopurine (also called mercaptopurine, 6-MP or its brand name Purinethol) and 6-thioguanine (also called tioguanine or 6-TG) using S-adenosyl-L-methionine as the methyl donor (PubMed:18484748, PubMed:657528). TPMT activity modulates the cytotoxic effects of thiopurine prodrugs. A natural substrate for this enzyme has yet to be identified

Structures available: 2BZG, 2H11 (PDB).

- **Originating measurements:** Matreyek 2018, Multiplex Assessment of Protein Variant Abundance by Massively Parallel Sequencing  
  Source: ProteinGym reference table
- **UniProt entry:** P51580  
  Source: [UniProt](https://www.uniprot.org/uniprotkb/P51580/entry)

## Biosafety and handling

This assay uses mammalian cell culture and flow cytometry. Cell-line provenance and sorter decontamination are the usual concerns.

**Containment level is not stated here.** It depends on the strain, the host, the construct and your local regulation, and it is your institution's determination to make. Confirm it with your biosafety officer before starting, and follow your local waste procedure.

## Conditions

Reproduce the originating assay's conditions where they are stated: the fitness values Drucion was trained on were measured under them, so results obtained under different conditions are not on the same scale.

- **Optimum pH:** Optimum pH is 7.5  
  Source: [UniProt P51580 (PMID 657528)](https://www.uniprot.org/uniprotkb/P51580/entry)
- **Kinetics:** Vmax 1.23 nmol/sec/mg — enzyme toward 6-mercaptopurine (at pH 6.5 and 37 degrees Celsius)  
  Source: [UniProt P51580 (PMID 18484748)](https://www.uniprot.org/uniprotkb/P51580/entry)
- **Kinetics:** Km 18.5 uM for S-adenosyl-L-methionine (at pH 6.5 and 37 degrees Celsius)  
  Source: [UniProt P51580 (PMID 18484748)](https://www.uniprot.org/uniprotkb/P51580/entry)
- **Kinetics:** Km 0.68 mM for 6-mercaptopurine (at pH 6.5 and 37 degrees Celsius)  
  Source: [UniProt P51580 (PMID 18484748)](https://www.uniprot.org/uniprotkb/P51580/entry)
- **Kinetics:** Km 1.7 uM for S-adenosyl-L-methionine (at pH 7.5 and 37 degrees Celsius)  
  Source: [UniProt P51580 (PMID 657528)](https://www.uniprot.org/uniprotkb/P51580/entry)
- **Subcellular location:** Cytoplasm  
  Source: [UniProt P51580](https://www.uniprot.org/uniprotkb/P51580/entry)

## Procedure

General technique for this assay class. It is not specific to this protein; adapt it to your strain, vector and equipment.

1. Produce a stable line per variant with the fusion construct and a separately expressed normalisation marker.
2. Culture to the density your sorter protocol expects; include wild-type and a known-destabilised variant on every run.
3. Sort on the ratio of fusion signal to normalisation marker, not on raw fluorescence, so expression differences divide out.
4. Collect the sorted bins and quantify variant frequency in each.
5. Convert bin frequencies to an abundance score per variant.
6. Record gating, bin boundaries and run date in the logbook — these drift between sessions and explain more disagreements than anything else.

## Controls

Drucion compares variants to each other, so a run without controls cannot be merged with earlier rounds.

- Wild-type, on every plate or run. Drucion's scores are relative to it, so a run without it cannot be placed on the same scale as the others.
- A known-null or known-destabilised variant, to confirm the assay can still resolve a bad variant on this particular day.
- A no-template or untransformed well, to catch contamination.
- Technical replicates of at least two variants per plate, to estimate within-run noise before trusting between-variant differences.

## Estimates — verify before relying on these

None of the following was found in a record. Each is an inference, shown with the reasoning behind it and an experiment that would confirm it. **Do not treat these as measurements.**

- **Melting temperature (Tm):** 47.0 - 52.0 °C  
  *Estimate — not measured.* Reasoning: Human cytosolic TPMT is a monomeric globular protein exhibiting moderate basal thermal stability under physiological conditions.  
  **Confirm by:** Differential scanning fluorimetry (DSF) using SYPRO Orange dye on purified recombinant TPMT.
- **Isoelectric point (pI):** 5.2 - 5.8 pH units  
  *Estimate — not measured.* Reasoning: Calculated theoretical pI based on the primary 245-amino-acid sequence of human TPMT falls within the slightly acidic range.  
  **Confirm by:** Isoelectric focusing (IEF) gel electrophoresis of cell lysates.
- **Intracellular protein half-life (WT):** 14 - 22 hours  
  *Estimate — not measured.* Reasoning: Wild-type TPMT undergoes steady baseline proteasomal degradation in human cell culture environments.  
  **Confirm by:** Cycloheximide shut-off time-course assay followed by flow cytometry tracking of GFP fluorescence.
- **FACS preparation buffer pH:** 7.2 - 7.6 pH units  
  *Estimate — not measured.* Reasoning: Preserving a physiological pH during cell sorting maintains native intracellular folding and optimizes GFP fluorophore brightness.  
  **Confirm by:** Flow cytometry evaluation of GFP signal retention across a buffer pH gradient ranging from 6.5 to 8.0.

- Handling: Keep harvested cell samples strictly on ice or at 4 °C prior to FACS sorting to minimize proteasomal degradation of unstable TPMT variants.
- Handling: Add fresh reducing agents (such as 1 mM DTT or TCEP) if lysing cells, as TPMT contains free cysteine residues prone to inhibitory oxidation.
- Handling: Avoid repeated freeze-thaw cycles of cell pellets or tagged protein constructs to prevent protein aggregation and loss of GFP fluorescence.
- Common pitfall: Fixation or cell viability dye staining steps can quench GFP fluorescence or introduce autofluorescence background, misrepresenting protein abundance.
- Common pitfall: Steric hindrance or misfolding introduced by the terminal GFP fusion tag may artificially accelerate or delay native TPMT turnover.
- Common pitfall: Variability in plasmid expression or lentiviral multiplicity of infection (MOI) can cause cell-to-cell GFP fluorescence variance unrelated to protein stability.

## Recording results

Enter every measurement, including failures and outliers, against its variant identifier. A missing value and a measured low value are different inputs to the next round, and only one of them is information.
