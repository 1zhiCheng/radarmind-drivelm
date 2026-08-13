# B10 vs v0.37B selected checkpoint

| Metric | B10 | Candidate | Delta |
| --- | ---: | ---: | ---: |
| Coverage | 100.0000 | 100.0000 | +0.0000 |
| Exact Match | 43.4575 | 43.5768 | +0.1192 |
| Token-F1 | 72.9999 | 73.1231 | +0.1232 |
| ROUGE-L | 71.0756 | 71.2034 | +0.1278 |
| MC accuracy | 83.8202 | 84.1573 | +0.3371 |
| Planning /100 | 70.6348 | 70.8571 | +0.2223 |
| Coordinate F1 | 13.1313 | 13.3838 | +0.2525 |
| DriveLM-DS Final | 0.5946 | 0.5964 | +0.0017 |

Promotion decision: **PASS**

## Frozen gates

- [x] coverage_100_percent
- [x] judge_complete
- [x] final_strictly_improved
- [x] planning_regression_at_most_0_5_points
- [x] coordinate_f1_regression_at_most_0_5pp
- [x] mc_not_below_b10
