# Quantitative Results

Tested on **50 plates** across 7 scenarios. OCR is simulated (visible characters read, ~12% confusable misread rate).

| Scenario | Hidden | Traditional ANPR | Our system | Avg confidence |
|---|---|---|---|---|
| Clear (no hiding) | 0 | 100% | **98.0%** | 67.9% |
| 1 number digit | 1 | 0% | **90.0%** | 65.3% |
| 2 number digits | 2 | 0% | **98.0%** | 62.7% |
| 3 number digits | 3 | 0% | **84.0%** | 60.0% |
| 1 series letter | 1 | 0% | **100.0%** | 64.7% |
| 1 district digit | 1 | 0% | **100.0%** | 65.1% |
| 1 state letter | 1 | 0% | **100.0%** | 64.8% |

**Overall recovery on hidden plates: 95.3%** (traditional ANPR: 0%).
