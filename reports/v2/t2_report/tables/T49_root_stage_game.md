# T49: Root stage game

- priority: core; status: generated; tier: final
- sources: `results/v2_pilots/pilot4/analysis/root_game/own_curvature.csv`, `results/v2_pilots/pilot4/analysis/root_game/br_slope.csv`, `tools/v2/pilot4_root_game.py`, `reports/v2/pilot4_stabilization.md`
- built by: `tools/v2/report/sec_stage1.py:build_t49`; base commit `cb0b541`
- transformation: Blocks 'per fit' are verbatim copies of own_curvature.csv (12 rows) and br_slope.csv (60 rows); block 'summary' gives, per q and tier, n/min/max/median over the fit widths W (and steps h for the slope) and whether the reference lies in [min, max]. The references in the CSVs equal the PI references of the request (section 5.4): BR slope -0.961/-0.309, E[V2'']/(2k) 0.490/0.236. Tier: final (requested) and fine (diagnostic finer tier), labelled in column tier.

Root stage game at e_opp = e1*: numeric BR slope and own curvature from the verifier's stage-1 Q with the closed-form continuation, against the PI references.

| block | q | tier | fit_half_width | fit_points | h | e1_star | BR_at_e1star | BR_minus_e1star | grid_argmax | own_curv_2a | own_curv_over_2k | ev2pp_over_2k | slope_implied_by_curv | ref_ev2pp_over_2k | ref_own_curv_over_2k | BR_plus | BR_minus | slope | ref_slope | quantity | n | min | max | median | reference | reference_in_range | reference_source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| summary | 50 | final |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | dBR/de_opp, all fit widths W and steps h | 15 | -1.057 | -0.8533 | -0.9113 | -0.961 | True | PI reference (ref_slope) |
| summary | 50 | final |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | dBR/de_opp, h <= 1 | 9 | -1.057 | -0.8741 | -0.9243 | -0.961 | True | PI reference (ref_slope) |
| summary | 50 | final |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | dBR/de_opp, h = 1, W = 2 | 1 | -0.9288 | -0.9288 | -0.9288 | -0.961 | False | PI reference (ref_slope) |
| summary | 50 | final |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | own curvature 2a/(2k) | 3 | -0.528 | -0.4869 | -0.5239 | -0.51 | True | ref_own_curv_over_2k = ref E[V2'']/(2k) - 1 |
| summary | 50 | final |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | E[V2'']/(2k) = 1 + 2a/(2k) | 3 | 0.472 | 0.5131 | 0.4761 | 0.49 | True | PI reference (ref_ev2pp_over_2k) |
| summary | 50 | final |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | slope implied by the curvature, -r/(1-r) with r = E[V2'']/(2k) | 3 | -1.054 | -0.894 | -0.9088 | -0.961 | True | PI reference slope (ref_slope) |
| summary | 50 | final |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | BR(e1*) - e1* (fixed-point check) | 3 | -0.04676 | 0.03256 | 0.0025 | 0 | True | 0 at the fixed point e1* |
| summary | 50 | fine |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | dBR/de_opp, all fit widths W and steps h | 15 | -0.9471 | -0.8537 | -0.9021 | -0.961 | False | PI reference (ref_slope) |
| summary | 50 | fine |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | dBR/de_opp, h <= 1 | 9 | -0.9471 | -0.9002 | -0.9226 | -0.961 | False | PI reference (ref_slope) |
| summary | 50 | fine |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | dBR/de_opp, h = 1, W = 2 | 1 | -0.9224 | -0.9224 | -0.9224 | -0.961 | False | PI reference (ref_slope) |
| summary | 50 | fine |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | own curvature 2a/(2k) | 3 | -0.5223 | -0.5076 | -0.5205 | -0.51 | True | ref_own_curv_over_2k = ref E[V2'']/(2k) - 1 |
| summary | 50 | fine |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | E[V2'']/(2k) = 1 + 2a/(2k) | 3 | 0.4777 | 0.4924 | 0.4795 | 0.49 | True | PI reference (ref_ev2pp_over_2k) |
| summary | 50 | fine |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | slope implied by the curvature, -r/(1-r) with r = E[V2'']/(2k) | 3 | -0.9699 | -0.9146 | -0.9212 | -0.961 | True | PI reference slope (ref_slope) |
| summary | 50 | fine |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | BR(e1*) - e1* (fixed-point check) | 3 | -0.04506 | -0.004796 | -0.01149 | 0 | False | 0 at the fixed point e1* |
| summary | 60 | final |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | dBR/de_opp, all fit widths W and steps h | 15 | -0.3322 | -0.2644 | -0.2944 | -0.309 | True | PI reference (ref_slope) |
| summary | 60 | final |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | dBR/de_opp, h <= 1 | 9 | -0.3322 | -0.2644 | -0.2942 | -0.309 | True | PI reference (ref_slope) |
| summary | 60 | final |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | dBR/de_opp, h = 1, W = 2 | 1 | -0.3138 | -0.3138 | -0.3138 | -0.309 | False | PI reference (ref_slope) |
| summary | 60 | final |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | own curvature 2a/(2k) | 3 | -0.7917 | -0.7781 | -0.7871 | -0.764 | False | ref_own_curv_over_2k = ref E[V2'']/(2k) - 1 |
| summary | 60 | final |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | E[V2'']/(2k) = 1 + 2a/(2k) | 3 | 0.2083 | 0.2219 | 0.2129 | 0.236 | False | PI reference (ref_ev2pp_over_2k) |
| summary | 60 | final |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | slope implied by the curvature, -r/(1-r) with r = E[V2'']/(2k) | 3 | -0.2853 | -0.2631 | -0.2706 | -0.309 | False | PI reference slope (ref_slope) |
| summary | 60 | final |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | BR(e1*) - e1* (fixed-point check) | 3 | -0.01861 | 0.01227 | -0.01177 | 0 | True | 0 at the fixed point e1* |
| summary | 60 | fine |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | dBR/de_opp, all fit widths W and steps h | 15 | -0.3108 | -0.2926 | -0.2998 | -0.309 | True | PI reference (ref_slope) |
| summary | 60 | fine |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | dBR/de_opp, h <= 1 | 9 | -0.3108 | -0.2965 | -0.3001 | -0.309 | True | PI reference (ref_slope) |
| summary | 60 | fine |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | dBR/de_opp, h = 1, W = 2 | 1 | -0.3015 | -0.3015 | -0.3015 | -0.309 | False | PI reference (ref_slope) |
| summary | 60 | fine |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | own curvature 2a/(2k) | 3 | -0.7693 | -0.7663 | -0.768 | -0.764 | False | ref_own_curv_over_2k = ref E[V2'']/(2k) - 1 |
| summary | 60 | fine |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | E[V2'']/(2k) = 1 + 2a/(2k) | 3 | 0.2307 | 0.2337 | 0.232 | 0.236 | False | PI reference (ref_ev2pp_over_2k) |
| summary | 60 | fine |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | slope implied by the curvature, -r/(1-r) with r = E[V2'']/(2k) | 3 | -0.305 | -0.2999 | -0.302 | -0.309 | False | PI reference slope (ref_slope) |
| summary | 60 | fine |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | BR(e1*) - e1* (fixed-point check) | 3 | -0.01868 | -0.001071 | -0.005135 | 0 | False | 0 at the fixed point e1* |
| per fit: own curvature (own_curvature.csv) | 50 | final | 1 | 5 |  | 46.67 | 46.7 | 0.03256 | 47 | -0.0002782 | -0.4869 | 0.5131 | -1.054 | 0.49 | -0.51 |  |  |  |  |  |  |  |  |  |  |  |  |
| per fit: own curvature (own_curvature.csv) | 50 | final | 2 | 9 |  | 46.67 | 46.67 | 0.0025 | 47 | -0.0003017 | -0.528 | 0.472 | -0.894 | 0.49 | -0.51 |  |  |  |  |  |  |  |  |  |  |  |  |
| per fit: own curvature (own_curvature.csv) | 50 | final | 4 | 17 |  | 46.67 | 46.62 | -0.04676 | 47 | -0.0002994 | -0.5239 | 0.4761 | -0.9088 | 0.49 | -0.51 |  |  |  |  |  |  |  |  |  |  |  |  |
| per fit: own curvature (own_curvature.csv) | 50 | fine | 1 | 9 |  | 46.67 | 46.66 | -0.004796 | 46.75 | -0.0002901 | -0.5076 | 0.4924 | -0.9699 | 0.49 | -0.51 |  |  |  |  |  |  |  |  |  |  |  |  |
| per fit: own curvature (own_curvature.csv) | 50 | fine | 2 | 17 |  | 46.67 | 46.66 | -0.01149 | 46.75 | -0.0002974 | -0.5205 | 0.4795 | -0.9212 | 0.49 | -0.51 |  |  |  |  |  |  |  |  |  |  |  |  |
| per fit: own curvature (own_curvature.csv) | 50 | fine | 4 | 33 |  | 46.67 | 46.62 | -0.04506 | 46.75 | -0.0002985 | -0.5223 | 0.4777 | -0.9146 | 0.49 | -0.51 |  |  |  |  |  |  |  |  |  |  |  |  |
| per fit: own curvature (own_curvature.csv) | 60 | final | 1 | 5 |  | 38.89 | 38.9 | 0.01227 | 39 | -0.0004497 | -0.7871 | 0.2129 | -0.2706 | 0.236 | -0.764 |  |  |  |  |  |  |  |  |  |  |  |  |
| per fit: own curvature (own_curvature.csv) | 60 | final | 2 | 9 |  | 38.89 | 38.88 | -0.01177 | 39 | -0.0004524 | -0.7917 | 0.2083 | -0.2631 | 0.236 | -0.764 |  |  |  |  |  |  |  |  |  |  |  |  |
| per fit: own curvature (own_curvature.csv) | 60 | final | 4 | 17 |  | 38.89 | 38.87 | -0.01861 | 39 | -0.0004446 | -0.7781 | 0.2219 | -0.2853 | 0.236 | -0.764 |  |  |  |  |  |  |  |  |  |  |  |  |
| per fit: own curvature (own_curvature.csv) | 60 | fine | 1 | 9 |  | 38.89 | 38.89 | -0.001071 | 39 | -0.0004389 | -0.768 | 0.232 | -0.302 | 0.236 | -0.764 |  |  |  |  |  |  |  |  |  |  |  |  |
| per fit: own curvature (own_curvature.csv) | 60 | fine | 2 | 17 |  | 38.89 | 38.88 | -0.005135 | 39 | -0.0004379 | -0.7663 | 0.2337 | -0.305 | 0.236 | -0.764 |  |  |  |  |  |  |  |  |  |  |  |  |
| per fit: own curvature (own_curvature.csv) | 60 | fine | 4 | 33 |  | 38.89 | 38.87 | -0.01868 | 39 | -0.0004396 | -0.7693 | 0.2307 | -0.2999 | 0.236 | -0.764 |  |  |  |  |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 50 | final | 1 |  | 0.25 |  |  |  |  |  |  |  |  |  |  | 46.37 | 46.89 | -1.057 | -0.961 |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 50 | final | 1 |  | 0.5 |  |  |  |  |  |  |  |  |  |  | 46.15 | 47.05 | -0.8924 | -0.961 |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 50 | final | 1 |  | 1 |  |  |  |  |  |  |  |  |  |  | 45.72 | 47.57 | -0.9232 | -0.961 |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 50 | final | 1 |  | 2 |  |  |  |  |  |  |  |  |  |  | 44.63 | 48.33 | -0.9269 | -0.961 |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 50 | final | 1 |  | 4 |  |  |  |  |  |  |  |  |  |  | 42.08 | 49.18 | -0.8876 | -0.961 |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 50 | final | 2 |  | 0.25 |  |  |  |  |  |  |  |  |  |  | 46.39 | 46.88 | -0.9741 | -0.961 |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 50 | final | 2 |  | 0.5 |  |  |  |  |  |  |  |  |  |  | 46.18 | 47.11 | -0.9243 | -0.961 |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 50 | final | 2 |  | 1 |  |  |  |  |  |  |  |  |  |  | 45.65 | 47.51 | -0.9288 | -0.961 |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 50 | final | 2 |  | 2 |  |  |  |  |  |  |  |  |  |  | 44.69 | 48.27 | -0.8966 | -0.961 |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 50 | final | 2 |  | 4 |  |  |  |  |  |  |  |  |  |  | 42.56 | 49.43 | -0.8585 | -0.961 |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 50 | final | 4 |  | 0.25 |  |  |  |  |  |  |  |  |  |  | 46.37 | 46.85 | -0.9628 | -0.961 |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 50 | final | 4 |  | 0.5 |  |  |  |  |  |  |  |  |  |  | 46.17 | 47.04 | -0.8741 | -0.961 |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 50 | final | 4 |  | 1 |  |  |  |  |  |  |  |  |  |  | 45.66 | 47.48 | -0.9113 | -0.961 |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 50 | final | 4 |  | 2 |  |  |  |  |  |  |  |  |  |  | 44.67 | 48.2 | -0.8825 | -0.961 |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 50 | final | 4 |  | 4 |  |  |  |  |  |  |  |  |  |  | 42.53 | 49.36 | -0.8533 | -0.961 |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 50 | fine | 1 |  | 0.25 |  |  |  |  |  |  |  |  |  |  | 46.42 | 46.9 | -0.9471 | -0.961 |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 50 | fine | 1 |  | 0.5 |  |  |  |  |  |  |  |  |  |  | 46.19 | 47.11 | -0.9226 | -0.961 |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 50 | fine | 1 |  | 1 |  |  |  |  |  |  |  |  |  |  | 45.69 | 47.54 | -0.9248 | -0.961 |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 50 | fine | 1 |  | 2 |  |  |  |  |  |  |  |  |  |  | 44.67 | 48.27 | -0.8993 | -0.961 |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 50 | fine | 1 |  | 4 |  |  |  |  |  |  |  |  |  |  | 42.58 | 49.43 | -0.8567 | -0.961 |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 50 | fine | 2 |  | 0.25 |  |  |  |  |  |  |  |  |  |  | 46.42 | 46.88 | -0.9324 | -0.961 |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 50 | fine | 2 |  | 0.5 |  |  |  |  |  |  |  |  |  |  | 46.18 | 47.11 | -0.9288 | -0.961 |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 50 | fine | 2 |  | 1 |  |  |  |  |  |  |  |  |  |  | 45.68 | 47.53 | -0.9224 | -0.961 |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 50 | fine | 2 |  | 2 |  |  |  |  |  |  |  |  |  |  | 44.68 | 48.27 | -0.8981 | -0.961 |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 50 | fine | 2 |  | 4 |  |  |  |  |  |  |  |  |  |  | 42.56 | 49.45 | -0.8611 | -0.961 |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 50 | fine | 4 |  | 0.25 |  |  |  |  |  |  |  |  |  |  | 46.39 | 46.84 | -0.9002 | -0.961 |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 50 | fine | 4 |  | 0.5 |  |  |  |  |  |  |  |  |  |  | 46.16 | 47.06 | -0.9079 | -0.961 |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 50 | fine | 4 |  | 1 |  |  |  |  |  |  |  |  |  |  | 45.67 | 47.48 | -0.9021 | -0.961 |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 50 | fine | 4 |  | 2 |  |  |  |  |  |  |  |  |  |  | 44.67 | 48.22 | -0.8876 | -0.961 |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 50 | fine | 4 |  | 4 |  |  |  |  |  |  |  |  |  |  | 42.57 | 49.4 | -0.8537 | -0.961 |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 60 | final | 1 |  | 0.25 |  |  |  |  |  |  |  |  |  |  | 38.79 | 38.96 | -0.3322 | -0.309 |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 60 | final | 1 |  | 0.5 |  |  |  |  |  |  |  |  |  |  | 38.72 | 39.05 | -0.332 | -0.309 |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 60 | final | 1 |  | 1 |  |  |  |  |  |  |  |  |  |  | 38.58 | 39.13 | -0.2723 | -0.309 |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 60 | final | 1 |  | 2 |  |  |  |  |  |  |  |  |  |  | 38.23 | 39.45 | -0.3061 | -0.309 |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 60 | final | 1 |  | 4 |  |  |  |  |  |  |  |  |  |  | 37.67 | 39.91 | -0.2792 | -0.309 |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 60 | final | 2 |  | 0.25 |  |  |  |  |  |  |  |  |  |  | 38.82 | 38.95 | -0.2644 | -0.309 |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 60 | final | 2 |  | 0.5 |  |  |  |  |  |  |  |  |  |  | 38.73 | 39.03 | -0.2942 | -0.309 |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 60 | final | 2 |  | 1 |  |  |  |  |  |  |  |  |  |  | 38.56 | 39.19 | -0.3138 | -0.309 |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 60 | final | 2 |  | 2 |  |  |  |  |  |  |  |  |  |  | 38.24 | 39.45 | -0.3019 | -0.309 |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 60 | final | 2 |  | 4 |  |  |  |  |  |  |  |  |  |  | 37.54 | 39.91 | -0.2958 | -0.309 |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 60 | final | 4 |  | 0.25 |  |  |  |  |  |  |  |  |  |  | 38.79 | 38.94 | -0.2928 | -0.309 |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 60 | final | 4 |  | 0.5 |  |  |  |  |  |  |  |  |  |  | 38.72 | 39.01 | -0.2922 | -0.309 |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 60 | final | 4 |  | 1 |  |  |  |  |  |  |  |  |  |  | 38.55 | 39.16 | -0.3034 | -0.309 |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 60 | final | 4 |  | 2 |  |  |  |  |  |  |  |  |  |  | 38.24 | 39.42 | -0.2944 | -0.309 |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 60 | final | 4 |  | 4 |  |  |  |  |  |  |  |  |  |  | 37.58 | 39.89 | -0.2894 | -0.309 |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 60 | fine | 1 |  | 0.25 |  |  |  |  |  |  |  |  |  |  | 38.81 | 38.96 | -0.3108 | -0.309 |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 60 | fine | 1 |  | 0.5 |  |  |  |  |  |  |  |  |  |  | 38.74 | 39.03 | -0.2968 | -0.309 |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 60 | fine | 1 |  | 1 |  |  |  |  |  |  |  |  |  |  | 38.58 | 39.18 | -0.2998 | -0.309 |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 60 | fine | 1 |  | 2 |  |  |  |  |  |  |  |  |  |  | 38.24 | 39.46 | -0.3041 | -0.309 |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 60 | fine | 1 |  | 4 |  |  |  |  |  |  |  |  |  |  | 37.55 | 39.94 | -0.2978 | -0.309 |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 60 | fine | 2 |  | 0.25 |  |  |  |  |  |  |  |  |  |  | 38.81 | 38.96 | -0.3022 | -0.309 |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 60 | fine | 2 |  | 0.5 |  |  |  |  |  |  |  |  |  |  | 38.73 | 39.03 | -0.3019 | -0.309 |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 60 | fine | 2 |  | 1 |  |  |  |  |  |  |  |  |  |  | 38.57 | 39.18 | -0.3015 | -0.309 |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 60 | fine | 2 |  | 2 |  |  |  |  |  |  |  |  |  |  | 38.25 | 39.45 | -0.3004 | -0.309 |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 60 | fine | 2 |  | 4 |  |  |  |  |  |  |  |  |  |  | 37.58 | 39.92 | -0.2928 | -0.309 |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 60 | fine | 4 |  | 0.25 |  |  |  |  |  |  |  |  |  |  | 38.8 | 38.94 | -0.2965 | -0.309 |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 60 | fine | 4 |  | 0.5 |  |  |  |  |  |  |  |  |  |  | 38.72 | 39.02 | -0.3001 | -0.309 |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 60 | fine | 4 |  | 1 |  |  |  |  |  |  |  |  |  |  | 38.56 | 39.16 | -0.298 | -0.309 |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 60 | fine | 4 |  | 2 |  |  |  |  |  |  |  |  |  |  | 38.24 | 39.43 | -0.2965 | -0.309 |  |  |  |  |  |  |  |  |
| per fit: BR slope (br_slope.csv) | 60 | fine | 4 |  | 4 |  |  |  |  |  |  |  |  |  |  | 37.57 | 39.91 | -0.2926 | -0.309 |  |  |  |  |  |  |  |  |
