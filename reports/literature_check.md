# Dated literature check: closest prior work vs this endpoint + comparator
# Date of search: 2026-08-22

## Question

Among children a fever-based testing rule would not test, can a calibrated, survey-weighted model identify malaria parasite carriers, with incremental net benefit over that rule, and does that benefit hold across wealth and residence?

## Closest papers

1. Semakula HM, et al. A Bayesian belief network modelling of household factors influencing the risk of malaria: a study of parasitaemia in children under five years of age in sub-Saharan Africa. Environmental Modelling & Software. 2015. Uses MIS/DHS household factors; reports accuracy/AUC (~0.82) without calibration plots, decision-curve net benefit, restriction to afebrile/untested children, or leave-one-country-out validation.

2. Front. Artif. Intell. 2021. Development and validation of manually modified and supervised machine learning clinical assessment algorithms for malaria in Nigerian children. Clinic/CHW setting; IMCI fever rule is the testing trigger, so children the rule would not test never receive an RDT label. That is the opposite sample from this paper.

3. Awoyale et al. (2025) and related MIS machine-learning papers (e.g. Nigeria MIS 2021). Predict RDT positivity in all tested children 6–59 months. Discrimination-focused; not restricted to the untested (afebrile) group; typically no per-country calibration plots or DCA against the fever rule.

4. Geostatistical MIS mapping (e.g. national prevalence surfaces). Population-level mapping, not individual targeting among children a fever rule would miss.

5. Clinic CART/IMCI algorithm papers (Bayesian network + decision tree, Malawi 2021; historical CART vs IMCI). Named comparator precedent for benchmarking against IMCI, but clinic-presenting (usually febrile) samples.

## Point of difference (none of the above combine all four)

- Restriction to children a deployed fever-based rule would not test
- Decision-curve net benefit against that named rule (fever vs fever-plus-model on the full 6–59 month RDT sample)
- Calibration reported per held-out country (plots plus O/E, slope, intercept)
- Subgroup equity (wealth, urban/rural) with intervals

RDT is an imperfect reference in exactly this target group (HRP2 RDTs miss a share of low-density asymptomatic infections relative to molecular methods). That limitation is stated in the abstract, not only the discussion. Microscopy comparison is scoped to Burkina Faso 2021 and Nigeria 2021.
