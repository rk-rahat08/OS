# Experiment Summary

This file was generated from the saved experiment outputs.

- Dataset windows: 1800 (1350 train, 450 held-out test)
- Common comparison workload: the same 450 held-out windows and starting-head values were used for FCFS, SCAN, C-SCAN, SSTF, ML, and oracle results
- Held-out selection accuracy: 0.858
- Mean ML-selected movement: 186.22 cylinders/window
- Mean oracle movement: 182.02 cylinders/window
- Mean regret: 4.20 cylinders/window
- Mean confidence when correct: 0.945
- Mean confidence when wrong: 0.887
- Expected Calibration Error: 0.140
- Multiclass Brier score: 0.263
- Best fixed scheduler by workload: bursty: SSTF, random: SSTF, sequential: SSTF

## Shift metrics

| metric | before_shift | after_shift | difference |
| --- | --- | --- | --- |
| accuracy | 0.933 | 0.900 | -0.033 |
| confidence | 0.927 | 0.969 | 0.043 |
| movement | 159.267 | 196.367 | 37.100 |
| regret | 0.133 | 0.400 | 0.267 |

## Generalization test

| workload_type | accuracy | mean_regret | mean_confidence |
| --- | --- | --- | --- |
| bursty | 0.800 | 1.600 | 1.000 |
| random | 0.733 | 4.067 | 0.893 |
| sequential | 0.200 | 1.333 | 0.906 |
