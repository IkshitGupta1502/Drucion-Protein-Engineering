# Bench protocol — Beta-lactamase VIM-2

Assay: `A4GRB6_PSEAI_Chen_2020`
Compiled from: records + gemini-3.5-flash (gaps only)

## Before you start

This protocol contains **6 sourced** values (from a cited record or publication), **10** standard-technique steps and statements, and **4 estimates**.

The 4 estimates are inferences, not measurements. Each is marked, carries its reasoning, and names an experiment that would confirm it. Verify them before committing material:

- Zinc cofactor supplementation — confirm by: EDTA inhibition followed by zinc ion titration growth recovery assay
- Inoculum density — confirm by: Serial dilution and colony forming unit (CFU) quantification on agar plates
- Incubation duration — confirm by: Time-course OD600 kinetic growth curve monitoring using a microplate reader
- Growth medium pH — confirm by: Spectrophotometric assay of nitrocefin hydrolysis across a pH gradient

Provenance labels used below:

- **Sourced** — from a cited record or publication
- **Standard technique** — general technique, not specific to this protein
- **ESTIMATE — verify** — inferred, not measured — confirm before relying on it

## Target

Beta-lactamase VIM-2 from Pseudomonas aeruginosa, 266 residues, region mutated 1-266. Fitness is measured by: drug resistance (128/16/2.0 ug/mL ampicillin, 4.0/0.5 ug/mL cefotaxime, 0.031 ug/mL meropenem @ 25C, 37C).

- **Originating measurements:** Chen 2020, Comprehensive exploration of the translocation, stability and substrate recognition requirements in VIM-2 lactamase  
  Source: ProteinGym reference table
- **UniProt entry:** A4GRB6  
  Source: [UniProt](https://www.uniprot.org/uniprotkb/A4GRB6/entry)

## Biosafety and handling

This assay propagates an antibiotic-resistance gene and uses clinically relevant antibiotics. Resistance-gene work is regulated at most institutions and the waste stream is controlled.

**Containment level is not stated here.** It depends on the strain, the host, the construct and your local regulation, and it is your institution's determination to make. Confirm it with your biosafety officer before starting, and follow your local waste procedure.

## Conditions

Reproduce the originating assay's conditions where they are stated: the fitness values Drucion was trained on were measured under them, so results obtained under different conditions are not on the same scale.

- **Incubation temperature:** 25 °C, 37 °C  
  Source: Assay conditions as recorded in ProteinGym
- **Ampicillin concentration:** 128, 16, 2.0 ug/mL  
  Source: Assay conditions as recorded in ProteinGym
- **Cefotaxime concentration:** 4.0, 0.5 ug/mL  
  Source: Assay conditions as recorded in ProteinGym
- **Meropenem concentration:** 0.031 ug/mL  
  Source: Assay conditions as recorded in ProteinGym

## Procedure

General technique for this assay class. It is not specific to this protein; adapt it to your strain, vector and equipment.

1. Transform each variant plasmid into the expression strain and plate on non-selective medium; pick single colonies.
2. Grow overnight cultures, then dilute to a standardised inoculum (OD600 is the usual handle) so every variant starts equal.
3. Prepare a two-fold antibiotic dilution series spanning the concentrations above, in a 96-well plate.
4. Inoculate, seal, and incubate at the temperature given above.
5. Read growth after the fixed endpoint and record the lowest concentration with no visible growth as that variant's MIC.
6. Record every reading in the logbook, including wells that failed.

## Controls

Drucion compares variants to each other, so a run without controls cannot be merged with earlier rounds.

- Wild-type, on every plate or run. Drucion's scores are relative to it, so a run without it cannot be placed on the same scale as the others.
- A known-null or known-destabilised variant, to confirm the assay can still resolve a bad variant on this particular day.
- A no-template or untransformed well, to catch contamination.
- Technical replicates of at least two variants per plate, to estimate within-run noise before trusting between-variant differences.

## Estimates — verify before relying on these

None of the following was found in a record. Each is an inference, shown with the reasoning behind it and an experiment that would confirm it. **Do not treat these as measurements.**

- **Zinc cofactor supplementation:** 10 - 100 µM  
  *Estimate — not measured.* Reasoning: VIM-2 is a Class B metallo-beta-lactamase requiring catalytic zinc ions for full enzymatic activity during resistance assays.  
  **Confirm by:** EDTA inhibition followed by zinc ion titration growth recovery assay
- **Inoculum density:** 1.0x10^5 - 5.0x10^5 CFU/mL  
  *Estimate — not measured.* Reasoning: Standard microdilution antibiotic susceptibility testing requires a defined bacterial cell concentration to avoid inoculum effects.  
  **Confirm by:** Serial dilution and colony forming unit (CFU) quantification on agar plates
- **Incubation duration:** 16 - 20 hours  
  *Estimate — not measured.* Reasoning: Determining resistance phenotypes in Pseudomonas aeruginosa typically requires overnight growth to allow clear visual or spectrophotometric readout.  
  **Confirm by:** Time-course OD600 kinetic growth curve monitoring using a microplate reader
- **Growth medium pH:** pH 6.8 - 7.4  
  *Estimate — not measured.* Reasoning: Metallo-beta-lactamases display optimal stability and active site zinc retention within physiological pH limits.  
  **Confirm by:** Spectrophotometric assay of nitrocefin hydrolysis across a pH gradient

- Handling: Avoid using buffers containing chelating agents such as EDTA or EGTA, as they stripped essential zinc cofactors from the VIM-2 active site.
- Handling: Meropenem solutions are unstable in aqueous media at room temperature; prepare fresh stocks immediately prior to dispensing.
- Handling: Ensure uniform bacterial suspension when pipetting Pseudomonas aeruginosa to prevent cell clumping and inconsistent inocula.
- Common pitfall: Trace chelators in media or plasticware can strip zinc from VIM-2, causing false sensitivity to carbapenems.
- Common pitfall: Over-inoculation (inoculum effect) can lead to artificially elevated MIC results due to rapid enzymatic depletion of antibiotic.
- Common pitfall: Degradation of carbapenem stock solutions stored improperly leads to underestimation of resistance levels.

## Recording results

Enter every measurement, including failures and outliers, against its variant identifier. A missing value and a measured low value are different inputs to the next round, and only one of them is information.
