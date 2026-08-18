# DriveLM-DS official-style local leaderboard

This table reports every retained **complete 3,355-QA local-dev evaluation** in
the same column layout as the public DriveLM evaluator. `chatgpt*` is the one
intentional protocol substitution: it is a cached, temperature-zero
DeepSeek-V4-Flash semantic score because the official GPT judge was unavailable.
Consequently, `final_score` is a local **DriveLM-DS proxy**, not a hidden-server
score.

| rank | id | accuracy | chatgpt* | language/Bleu_1 | language/Bleu_2 | language/Bleu_3 | language/Bleu_4 | language/ROUGE_L | language/CIDEr | match | final_score |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | **v0.43-Graph-Anchor-MoL-Downstream 🏆** | **0.813154** | **73.0197** | **0.789634** | **0.724770** | **0.665099** | **0.607520** | **0.724571** | 0.198875 | 30.7513 | **0.612293** |
| 2 | v0.42A-Graph-CE-600 | 0.802691 | 71.2974 | 0.789634 | 0.724770 | 0.665099 | 0.607520 | 0.724571 | 0.198875 | 34.0947 | **0.609998** |
| 3 | **v0.39B-MoL-700 🏆** | **0.808735** | **72.4769** | 0.772320 | 0.709429 | 0.651638 | 0.595859 | 0.723957 | **0.200502** | 30.7513 | **0.608245** |
| 4 | v0.40-GRPO-70 | **0.808735** | 72.0910 | 0.772320 | 0.709429 | 0.651638 | 0.595859 | 0.723957 | **0.200502** | 30.7513 | 0.606702 |
| 5 | v0.40-GSPO-90 | **0.808735** | 72.0756 | 0.772320 | 0.709429 | 0.651638 | 0.595859 | 0.723957 | **0.200502** | 30.7513 | 0.606640 |
| 6 | v0.42B-Graph-Anchor-600 | 0.795107 | 70.6977 | 0.781084 | 0.716538 | 0.657221 | 0.600066 | 0.722131 | 0.196251 | **34.1263** | 0.605430 |
| 7 | **v0.37B-Conservative-DPO-75 🏆** | 0.799058 | 70.8571 | 0.782339 | 0.717131 | 0.657242 | 0.599524 | 0.724009 | 0.187116 | 28.8321 | 0.596356 |
| 8 | **v0.36-B10-CE-LoRA 🏆** | 0.799080 | 70.6348 | 0.781974 | 0.717055 | 0.657538 | 0.600139 | 0.721674 | 0.172968 | 28.5354 | 0.594636 |
| 9 | v0.37A-Standard-DPO-100 | 0.803125 | 69.7532 | 0.786984 | 0.722430 | 0.663428 | 0.606546 | **0.725154** | 0.194467 | 29.3497 | 0.594300 |
| 10 | v0.38A-Grounding-DPO-50 | 0.803053 | 69.7295 | 0.778150 | 0.713411 | 0.654237 | 0.597213 | 0.719581 | 0.181578 | 28.8321 | 0.592092 |
| 11 | v0.36-B11-CE-LoRA | 0.755518 | 70.6768 | 0.714093 | 0.657443 | 0.604516 | 0.553648 | 0.711807 | 0.174952 | 28.1439 | 0.580881 |
| 12 | v0.36-B00-CE-LoRA | 0.771200 | 68.4839 | 0.757321 | 0.695157 | 0.638281 | 0.583530 | 0.719165 | 0.173017 | 29.0783 | 0.580001 |
| 13 | Qwen2.5-VL-7B-Zero-Shot | 0.413276 | 36.8308 | 0.092504 | 0.019063 | 0.005750 | 0.001500 | 0.078660 | 0.000042 | 10.3788 | 0.257961 |

The displayed `rank` is a descending sort of candidate-dependent
`final_score`. It is **not** the checkpoint promotion order because Graph
gating can produce different eligible QA sets for different candidates.

## Coverage and promotion audit

| id | predictions | graph eligible | judge complete | decision |
| --- | ---: | ---: | ---: | --- |
| **v0.43-Graph-Anchor-MoL-Downstream** | 3,355/3,355 | 1,927 | 791/791 cache-only | **Current promoted full-system baseline** |
| v0.42A-Graph-CE-600 | 3,355/3,355 | 1,927 | 791/791 | Rejected: same-ID Final −0.003428, Planning −2.2240, MC −1.25 pp |
| **v0.39B-MoL-700** | 3,355/3,355 | 1,911 | 780/780 | Promoted at v0.39B; component baseline for v0.43 |
| v0.40-GRPO-70 | 3,355/3,355 | 1,911 | 780/780 | Rejected: full-system Final below MoL |
| v0.40-GSPO-90 | 3,355/3,355 | 1,911 | 780/780 | Rejected: full-system Final below MoL |
| v0.42B-Graph-Anchor-600 | 3,355/3,355 | 1,898 | 777/777 | Rejected: same-ID Final −0.005678 |
| **v0.37B-Conservative-DPO-75** | 3,355/3,355 | 1,866 | 762/762 | Promoted at v0.37B |
| **v0.36-B10-CE-LoRA** | 3,355/3,355 | 1,889 | 770/770 | Promoted at v0.36 |
| v0.37A-Standard-DPO-100 | 3,355/3,355 | 1,867 | 760/760 | Rejected |
| v0.38A-Grounding-DPO-50 | 3,355/3,355 | 1,901 | 779/779 | Rejected |
| v0.36-B11-CE-LoRA | 3,355/3,355 | 1,779 | 723/723 | Rejected |
| v0.36-B00-CE-LoRA | 3,355/3,355 | 1,844 | 752/752 | Rejected |
| Qwen2.5-VL-7B-Zero-Shot | 3,355/3,355 | 1,533 | 599/599 | Raw control |

The machine-readable source is
[`official_style_metrics.json`](official_style_metrics.json). Partial API runs,
smoke tests, checkpoint screens without a complete semantic judge, and fixed-ID
audit subsets are intentionally excluded from the main table.
