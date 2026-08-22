# Phase 1 dataset lock — Step 2 study population

Date: 2026-08-22

This report is the hard gate before modeling. The endpoint is malaria RDT positivity among children aged 6–59 months with no reported fever in the preceding two weeks (`h22==0`). Care-seeking items (`h32y`) are asked only after fever and are not independently informative in the afebrile subset.

## Merge failures

| country | survey_year | recode | n_kr | n_hr_person | kr_no_rdt_merge | rdt_no_kr | kr_no_gps | rdt_source | rdt_present | skip_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Burkina Faso | 2010 | 62 | 15044 | 82095 | 9317 | 398 | 885 | hml35 | True |  |
| Burkina Faso | 2014 | 71 | 6841 | 39529 | 1339 | 652 | 108 | hml35 | True |  |
| Burkina Faso | 2017-2018 | 7A | 6061 | 36672 | 882 | 393 | 566 | hml35 | True |  |
| Burkina Faso | 2021 | 81 | 12343 | 83949 | 7222 | 648 | 0 | hml35 | True |  |
| Gambia | 2013 | 61 | 8088 | 52691 | 5385 | 595 | 8088 | hml35 | True |  |
| Gambia | 2019-2020 | 81 | 8362 | 55640 | 5106 | 412 | 0 | hml35 | True |  |
| Mali | 2010 | 61 | 0 | 0 | 0 | 0 | 0 | none | False | no_kr_or_no_hr |
| Mali | 2012-2013 | 6A | 10326 | 58330 | 6139 | 1519 | 0 | hml35 | True |  |
| Mali | 2015 | 72 | 7749 | 39408 | 1256 | 809 | 0 | hml35 | True |  |
| Mali | 2018 | 7A | 9940 | 54571 | 6068 | 476 | 0 | hml35 | True |  |
| Mali | 2021 | 83 | 9464 | 54419 | 1605 | 871 | 191 | hml35 | True |  |
| Mauritania | 2019-2021 | 71 | 11628 | 73796 | 2961 | 1477 | 19 | hml35 | True |  |
| Niger | 2021 | 82 | 5120 | 30305 | 976 | 580 | 0 | hml35 | True |  |
| Nigeria | 2010 | 61 | 5978 | 30537 | 1625 | 794 | 0 | hml35 | True |  |
| Nigeria | 2015 | 71 | 6524 | 38442 | 1229 | 730 | 139 | hml35 | True |  |
| Nigeria | 2018 | 7B | 33924 | 188010 | 23772 | 1045 | 149 | hml35 | True |  |
| Nigeria | 2021 | 81 | 10988 | 70428 | 1861 | 1596 | 0 | hml35 | True |  |
| Senegal | 2008-2009 | 5A | 15595 | 90824 | 15427 | 3864 | 73 | sh212 | True |  |
| Senegal | 2010 | 61 | 12326 | 77269 | 8980 | 1370 | 223 | hml35 | True |  |
| Senegal | 2012 | 6D | 6862 | 41593 | 1510 | 1964 | 0 | hml35 | True |  |
| Senegal | 2014 | 71 | 6842 | 40723 | 1424 | 1344 | 120 | hml35 | True |  |
| Senegal | 2015 | 7A | 6935 | 41902 | 1478 | 1438 | 0 | hml35 | True |  |
| Senegal | 2016 | 7I | 6725 | 41632 | 1401 | 1379 | 0 | hml35 | True |  |
| Senegal | 2017 | 7Z | 12185 | 78950 | 2585 | 1297 | 0 | hml35 | True |  |
| Senegal | 2018 | 81 | 6719 | 43564 | 6719 | 0 | 0 | none | False |  |
| Senegal | 2019 | 8B | 6125 | 41050 | 6125 | 0 | 0 | none | False |  |
| Senegal | 2020-2021 | 8I | 6395 | 49270 | 3803 | 811 | 0 | hml35 | True |  |

## Filter flow (N at each step)

| country | survey_year | recode | 0_kr_rows | 1_age_6_59 | 2_valid_rdt | 3_fever_item_nonmissing | 4_afebrile_h22eq0 | 5_afebrile_no_recent_antimalarial |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Burkina Faso | 2010 | 62 | 1.504e+04 | 6068 | 5727 | 5727 | 4372 | 4367 |
| Burkina Faso | 2014 | 71 | 6841 | 5628 | 5502 | 5495 | 3100 | 3100 |
| Burkina Faso | 2017-2018 | 7A | 6061 | 5362 | 5179 | 5150 | 4026 | 4026 |
| Burkina Faso | 2021 | 81 | 1.234e+04 | 1.104e+04 | 5120 | 5120 | 3826 | 3816 |
| Gambia | 2013 | 61 | 8088 | 3171 | 2703 | 2703 | 2324 | 2320 |
| Gambia | 2019-2020 | 81 | 8362 | 7309 | 3255 | 3255 | 2681 | 2681 |
| Mali | 2012-2013 | 6A | 1.033e+04 | 4361 | 4148 | 4148 | 3702 | 3690 |
| Mali | 2015 | 72 | 7749 | 6532 | 6493 | 6493 | 4509 | 4509 |
| Mali | 2018 | 7A | 9940 | 8861 | 3871 | 3871 | 3218 | 3210 |
| Mali | 2021 | 83 | 9464 | 8407 | 7857 | 7787 | 5624 | 5624 |
| Mauritania | 2019-2021 | 71 | 1.163e+04 | 1.036e+04 | 8665 | 8665 | 7097 | 7063 |
| Niger | 2021 | 82 | 5120 | 4634 | 4143 | 4143 | 2737 | 2737 |
| Nigeria | 2010 | 61 | 5978 |  |  |  |  |  |
| Nigeria | 2015 | 71 | 6524 | 5552 | 5295 | 5295 | 2890 | 2890 |
| Nigeria | 2018 | 7B | 3.392e+04 | 3.052e+04 | 1.015e+04 | 1.015e+04 | 7380 | 7353 |
| Nigeria | 2021 | 81 | 1.099e+04 | 9981 | 9127 | 9127 | 5736 | 5736 |
| Senegal | 2008-2009 | 5A | 1.56e+04 | 4076 | 159 | 159 | 113 | 113 |
| Senegal | 2010 | 61 | 1.233e+04 | 3784 | 3346 | 3346 | 2599 | 2588 |
| Senegal | 2012 | 6D | 6862 | 5518 | 5350 | 5350 | 4341 | 4332 |
| Senegal | 2014 | 71 | 6842 | 5574 | 5417 | 5417 | 4739 | 4738 |
| Senegal | 2015 | 7A | 6935 | 5668 | 5457 | 5457 | 4504 | 4502 |
| Senegal | 2016 | 7I | 6725 | 5529 | 5324 | 5324 | 4476 | 4476 |
| Senegal | 2017 | 7Z | 1.218e+04 | 1.098e+04 | 9597 | 9597 | 7490 | 7475 |
| Senegal | 2018 | 81 | 6719 | 6074 |  |  |  |  |
| Senegal | 2019 | 8B | 6125 | 5470 |  |  |  |  |
| Senegal | 2020-2021 | 8I | 6395 | 5640 | 2591 | 2591 | 1850 | 1850 |

## Table 2 — afebrile subgroup with valid RDT

| country | survey_year | recode | n_afebrile | rdt_positive | rdt_negative | weighted_prev | unweighted_prev | flag_rdtpos_lt30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Burkina Faso | 2010 | 62 | 4367 | 3284 | 1083 | 0.7571 | 0.752 | 0 |
| Burkina Faso | 2014 | 71 | 3100 | 1769 | 1331 | 0.532 | 0.5706 | 0 |
| Burkina Faso | 2017-2018 | 7A | 4026 | 733 | 3293 | 0.1811 | 0.1821 | 0 |
| Burkina Faso | 2021 | 81 | 3816 | 827 | 2989 | 0.222 | 0.2167 | 0 |
| Gambia | 2013 | 61 | 2320 | 38 | 2282 | 0.02291 | 0.01638 | 0 |
| Gambia | 2019-2020 | 81 | 2681 | 8 | 2673 | 0.003541 | 0.002984 | 1 |
| Mali | 2012-2013 | 6A | 3690 | 1553 | 2137 | 0.4522 | 0.4209 | 0 |
| Mali | 2015 | 72 | 4509 | 1170 | 3339 | 0.2761 | 0.2595 | 0 |
| Mali | 2018 | 7A | 3210 | 443 | 2767 | 0.16 | 0.138 | 0 |
| Mali | 2021 | 83 | 5624 | 756 | 4868 | 0.1501 | 0.1344 | 0 |
| Mauritania | 2019-2021 | 71 | 7063 | 54 | 7009 | 0.009376 | 0.007645 | 0 |
| Niger | 2021 | 82 | 2737 | 410 | 2327 | 0.195 | 0.1498 | 0 |
| Nigeria | 2015 | 71 | 2890 | 976 | 1914 | 0.3561 | 0.3377 | 0 |
| Nigeria | 2018 | 7B | 7353 | 2399 | 4954 | 0.3055 | 0.3263 | 0 |
| Nigeria | 2021 | 81 | 5736 | 1820 | 3916 | 0.3272 | 0.3173 | 0 |
| Senegal | 2008-2009 | 5A | 113 | 14 | 99 | 0.1317 | 0.1239 | 1 |
| Senegal | 2010 | 61 | 2588 | 82 | 2506 | 0.02695 | 0.03168 | 0 |
| Senegal | 2012 | 6D | 4332 | 142 | 4190 | 0.02734 | 0.03278 | 0 |
| Senegal | 2014 | 71 | 4738 | 70 | 4668 | 0.008528 | 0.01477 | 0 |
| Senegal | 2015 | 7A | 4502 | 36 | 4466 | 0.005141 | 0.007996 | 0 |
| Senegal | 2016 | 7I | 4476 | 61 | 4415 | 0.006671 | 0.01363 | 0 |
| Senegal | 2017 | 7Z | 7475 | 89 | 7386 | 0.006513 | 0.01191 | 0 |
| Senegal | 2020-2021 | 8I | 1850 | 79 | 1771 | 0.04326 | 0.0427 | 0 |

## Flags: country-year afebrile RDT+ < 30

| country | survey_year | recode | n_afebrile | rdt_positive |
| --- | --- | --- | --- | --- |
| Gambia | 2019-2020 | 81 | 2681 | 8 |
| Senegal | 2008-2009 | 5A | 113 | 14 |

These survey-years are too small for stable calibration plots if analysed alone. They remain in the pooled training set unless a later decision excludes them from LOCO test folds.

## Pooled by country (LOCO test-fold size)

| country | n | rdt_pos | flag_lt30 |
| --- | --- | --- | --- |
| Burkina Faso | 15309 | 6613 | False |
| Gambia | 5001 | 46 | False |
| Mali | 17033 | 3922 | False |
| Mauritania | 7063 | 54 | False |
| Niger | 2737 | 410 | False |
| Nigeria | 15979 | 5195 | False |
| Senegal | 30074 | 573 | False |

## Notes

- RDT lives in HR person slots (`hml35_nn`); KR has no `hml35`. Merge key: `v001+v002+b16` = `hv001+hv002+hvidx`.
- Senegal 2008–09 uses `sh212` (1=positive, 2=negative) if `hml35` is absent.
- Mali 2010 has no KR file and is excluded from the afebrile endpoint.
- One-week fever recall is not collected; `h22` is a two-week window.
- Recent antimalarial items are typically asked only of febrile children, so HRP2 persistence after treatment cannot be fully excluded in the afebrile group.
- Nigeria 2010 KR has no `b19`/`hw1` (all missing); it contributes no 6–59 month rows under the locked age rule. Person-level `hml16` exists in HR and can be used in a later lock if needed.
- Senegal 2018 and 2019 do not collect `hml35` (dictionary `NA-`); they are excluded from the endpoint.
- Senegal 2014 `6R` was dropped as a duplicate of recode `71` (same modal interview year). Senegal 2016 `G0` was dropped as a duplicate of 2015 recode `7A`.
- Gambia 2013 has no GPS file (`kr_no_gps` = all KR rows).