# Demo Notes (3-5 minutes)

## 0:00-0:30 — Problem
Disk head movement is expensive, and the best scheduling rule can depend on the request pattern. The project selects FCFS, SCAN, C-SCAN, or SSTF for each window.

## 0:30-1:20 — Classical algorithms
Open `src/schedulers.py`. Mention true boundary movement for SCAN/C-SCAN and lower-cylinder SSTF tie-breaking.

## 1:20-2:00 — Workloads and features
Show sequential, random, bursty generators and the 16 leakage-free features in `src/features.py`.

## 2:00-2:40 — Learned selector
Show the decision tree, trace-level train/test split, predicted scheduler, and maximum class probability as confidence. Held-out accuracy: 0.858.

## 2:40-3:30 — Shift
Open `results/figures/workload_shift_timeline.png`. The timeline changes from sequential to random at window 30.

## 3:30-4:15 — Results
Show the classical comparison, confusion matrix, and reliability diagram. Mean selector regret: 4.20 cylinders/window.

## 4:15-5:00 — Finding and limitation
Explain the measured best schedulers (bursty: SSTF, random: SSTF, sequential: SSTF) and that cylinder movement is not real SSD or HDD latency.

## Likely instructor questions

- **Why a decision tree?** It is lightweight, supports multiclass probabilities, and can be explained using feature splits.
- **Why these features?** They describe spread, locality, ordering and the head's position without using scheduler outcomes.
- **Why does workload type matter?** Local, ordered and wide requests create different travel paths.
- **SCAN vs C-SCAN?** SCAN reverses at a boundary; C-SCAN wraps and keeps one service direction.
- **SCAN vs LOOK?** SCAN reaches the physical boundary; LOOK reverses at the last request.
- **How is confidence calculated?** It is the largest `predict_proba` value.
- **What is calibration?** Predictions near a confidence value should be correct about that often.
- **What is distribution shift?** The request-generating pattern changes from the training or earlier pattern.
- **How is ground truth made?** Run all four schedulers on the same window and choose minimum movement.
- **Why is the oracle not deployable?** It knows all four outcomes before choosing and would do four simulations.
- **What if schedulers tie?** FCFS, SSTF, SCAN, then C-SCAN is the fixed tie order.
- **Why total movement?** It is a clear conventional proxy for mechanical seek effort.
- **Does this apply directly to SSDs?** No. SSDs do not have mechanical head seek movement.
