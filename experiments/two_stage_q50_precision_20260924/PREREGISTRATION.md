# Supplemental q=50, T=2 precision sample (ten fixed runs)

Registered UTC: 2026-09-24T14:31:38.289400+00:00

This supplement was authorized after all outcomes of the previous 30-run restart evaluation were known: discovery 25/30, joint certification 22/25 candidates, end-to-end 22/30. It is not an independent fixed-N confirmation of a previously unknown result. The previous report and its ten restart triples remain unchanged.

## Fixed design

Exactly ten additional independent random initializations: seeds 10231–10240 inclusive. These seeds were absent from the existing E1 builder's experiment/result/config JSON seed scan immediately before registration. Execute every scheduled run; do not stop for favorable results, add seeds for unfavorable results, substitute seeds, or change any training or verification parameter.

Economic configuration is q=50, T=2, w_h=6, w_l=2, k=1/3500, effort range [0,100]. Every run is copied from the previous manifest with only run name, seed, and output directory changed. Retain the same runner and locked protocol, all existing module safeguards, fixed phase caps, first-eligible candidate rule, and final verifier. No warm starts, checkpoint substitution, resumed training after final rejection, or modified threshold. Each candidate receives the same final joint certification.

Use CPU, ten concurrent processes, one Torch/OMP/MKL/OPENBLAS thread per process. Each run initializes all its own model and RNG state. No new restart grouping is constructed from these ten runs.

## Outcomes and denominators

Report the new ten runs separately first, the prior 30 separately, and a clearly labeled descriptive pooled n=40 second. Do not rewrite the old summary or its ten fixed groups. All scheduled runs remain in the discovery and end-to-end denominators; operational problems are explicit, not silently excluded.

Candidate discovery = first valid development C check with dReach/DeltaW <=0.01 and normalized concentration <=0.04.
Conditional joint certification = discovered candidates also satisfying the existing valid development/final tiers, final dReach/DeltaW <=0.01, both refinement differences <=0.002, and dense C_all <=0.04, divided by all discovered candidates. If there are none, report N/A.
End-to-end = jointly certified candidates divided by all scheduled runs.
No-candidate terminal evaluations remain diagnostics and cannot become selected candidates.

Report Wilson 95% intervals for each rate, every seed's operation/candidate/certification status, failure reason, numerical values and costs. Report actual batch elapsed time separately from summed process wall/CPU time.

## Statistical interpretation and fixed stopping

This is a supplemental batch initiated after seeing the prior 30 outcomes. Pooled Wilson intervals are descriptive fixed-final-N summaries; nominal fixed-sample coverage is not guaranteed under an unmodeled outcome-dependent decision to extend sampling. No significance, equivalence, publication-readiness, or guaranteed precision-improvement claim is made.

Compare actual Wilson interval widths for prior n=30 and pooled n=40. Larger n does not guarantee a narrower realized interval when the estimated probability or conditional denominator changes. For a reference comparison only, show widths at an unchanged prior observed probability and a stated larger denominator; distinguish this arithmetic reference from the realized pooled result.

Stop after exactly ten scheduled runs. Scientific failures are retained and not rerun. Any confirmed infrastructure interruption is documented; no replacement seed is permitted. This batch makes no restart-policy decision and starts no three-stage run. No new hashes, contracts, baselines, or gates are introduced.
