# Market-characteristic data audit

Date: September 19, 2026. Independent review of the prepared cache before interpreting the new linear study's results. No pipeline or prepared data were changed during this review. **No implementation blocker was found; a small set of extreme source observations requires caution.**

The cache [35ef3a5eb0cb5e42](../.runs/characteristics_v1/prepared/35ef3a5eb0cb5e42/manifest.json) preserves the previous social-data universe: **3,514,785 stock-days for 7,974 securities**, June 2, 2010–December 29, 2023. The evaluation period, 2014–2022, contains **3,034,035 stock-days** before target-availability restrictions. There are 439 unchanged social inputs, 17 added market controls and 17 corresponding missingness indicators: 473 input columns altogether. All parent stock-day keys occur in the primitive source; no observation was dropped because a control was unavailable.

The builder used 16,575,260 source-history rows, beginning June 2, 2008, from the raw CRSP panel. The full exchange calendar spans 2006–2024. Source histories include explicit missing stock-session observations, so the audit's zero *absent rows* does not mean complete return or price coverage. Return windows require complete exchange-session observations and a valid preceding close. Each control uses information through signal close **t**, while the inherited prediction target begins at that close. Momentum compounds the 231 returns from t−251 through t−21; it excludes the latest 21 sessions.

Observed coverage during 2014–2022, without dropping rows or conditioning on target availability:

| Control group | Observed coverage |
|---|---:|
| Return ending today; 5-session return | 99.954%; 99.819% |
| 21-, 63-, 126-session returns | 99.307%; 97.986%; 96.011% |
| 252-session return; momentum excluding latest month | 91.793%; 91.826% |
| Current capitalization, price, turnover and dollar volume | approximately 99.995% |
| 21-session turnover and dollar volume | 99.339% |
| 21-/63-session volatility; 21-session maximum return | 99.307% / 97.986%; 99.307% |
| 21-session Amihud illiquidity | 98.839% |

All 17 controls are jointly observed on **91.411%** of evaluation-period rows. The 252-session return's coverage falls to **85.563% in 2021**, consistent with shorter or interrupted histories; the audit does not establish which explanation dominates. Missing-value flags therefore matter economically and must remain in every matched characteristic specification.

Independent numerical checks passed:

- Exponentiating log capitalization matches inherited CRSP capitalization on **3,514,599 finite pairs**, with zero mismatches at relative/absolute tolerance 1e−10; maximum relative error was approximately 1e−15. Units are **millions of dollars per security**, rather than aggregate issuer capitalization. AAPL on July 31, 2023 implies approximately $3.071 trillion, confirming the scale.
- The one-session return matches inherited CRSP `ret` on **3,513,212 finite pairs**, with zero mismatches. Direct products and sample standard deviations calculated from the separate full-market CRSP history reproduce six return horizons, momentum, two volatility horizons and maximum return for PSTV, SPI and MARA observations; maximum absolute difference was below **1.8e−13**.
- Independently calculated observed-value ranks match **377,145 transformed values across 20 sampled dates**, including every parent input row regardless of target availability. All 17 missingness indicators match the raw controls over the full panel; every missing scalar has neutral rank zero.

Raw extremes were retained and ranked, without introducing new clipping rules. SPI's September 23, 2020 daily return is 12.365155 (1,236.5%); MARA's April 5, 2021 trailing annual return is 124.659697 (12,466.0%). These values reproduce the source return histories. More concerning, PSTV on August 16, 2019 implies 9,000 shares outstanding, a $14.44 price, $129,960 capitalization and 16.7 million shares traded: **1,858 times shares outstanding in one day**. The capitalization also matches the inherited panel. This warrants a later source/corporate-action check; it is not evidence that the newly implemented unit conversion is wrong. Only eight evaluation-period rows have turnover above 100 times shares outstanding, including one above 1,000. Date-local ranks bound their numerical magnitude but cannot correct an erroneous ordering. A predeclared microcap/low-price sensitivity is appropriate after the fixed experiment.

These controls cover return history, size, price, trading activity and liquidity/risk proxies. **They are not a comprehensive accounting-characteristic benchmark.** Legacy daily book-to-market was merged at fiscal-period dates without a verified reporting lag and is excluded from predictors; inherited DGTW outcomes retain that upstream limitation. The existing nominal 16:00 message routing, early-close issue and executable-close assumptions also remain. Any incremental-information conclusion applies to the existing social-data stock-day universe and this market-based control set.

Recorded SHA-256 identities:

| Item | SHA-256 |
|---|---|
| Prepared fingerprint | `35ef3a5eb0cb5e4211f94b4e7c057b8b44f4dc60206dd1d07870ec0e7be5849f` |
| Manifest, independently checked | `4d7fc5a6edd68b85effd5f7aaf84accf9c93fcc7a96317702fad0b579a9765d3` |
| Builder audit, independently checked | `ed9f2c67e94dc7f53d9a183e80294b3da58507eadda0a29ed3727006c2066c32` |
| Primitive `dsf.pkl`, recorded at preparation | `1de90f741dd9f3209784df1a34fd5667c60f5820301b8831106217e151347cdc` |
| Derived `X.npy`, recorded at preparation | `6950aaa0aa36615df903d7605efb060369e7f04ce299733f1b92e23fa6c1a787` |

The source pickle was not reloaded for this independent review. Source hashes and parent-preservation records are in the manifest; detailed formulas, counts and distributions are in [characteristic_audit.json](../.runs/characteristics_v1/prepared/35ef3a5eb0cb5e42/characteristic_audit.json) and the manifest's `characteristics` section.
