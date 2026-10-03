# Learning-Augmented Disk Scheduling: Adaptive Scheduler Selection Under Workload Shift

## 1. Problem Framing

Disk scheduling decide which pending request should be served next. A mechnical disk head has to move between cylinder positions, so more movement normally means more seek time. FCFS keeps arrival order, SSTF chooses the closest request, SCAN travels like an elevator, and C-SCAN only travels in one direction before wrapping [1]. No policy was assumed to be best for every request pattern. This project checks if a small learned selector can choose a suitable policy for each request window.

   A second issue is workload shift. A workload can be local and sequential at first, but later become wide and random. The selector only sees features from the current re quest window; it does not see the scheduler results before making its choice.

## 2. Implementation

The simulator implement FCFS, true SCAN, true C-SCAN and SSTF from scratch. SCAN goes to the disk boundary before reversing. C-SCAN counts both movement to the boundary and the wrap. Equal SSTF distances select the lower cylinder. Tied s cheduler labels use FCFS, SSTF, SCAN, then C-SCAN order.

 For each 20-request window, the model uses 16 simple features: count, mean, standard deviation, range, two head-distance values, fractions above and below the head, jump statistics, unique ratio, locality, slope, direction changes and histogram entropy. These can show if requests are spread out, ordered or clusterd. The classifier is a decision tree, because its choices are easy to inspect and explain in a class demo [2], and the code uses scikit-learn [3]. Confidence is the highest predicted class probability.

## 3. Experimental Setup

Requests use cylinders 0 to 199, initial head 100, and initial direction `up`. The generated dataset has 1800 windows from sequential, random and bursty traces. There was 60 traces for each type and 10 windows per trace. A group split kept complete traces together: 1350 windows for training and 450 for held-out testing. The fixed experment seed was 307. The shift timeline has 30 sequential windows followed by 30 random windows.

  Total head movement in cylinders is used as a simulated seek-cost proxy, not millisecond hardware timing. The oracle is the cheapest of four schedulers after running all four, so it is only a reference and cannot be deployed as an online choice.

Every primary comparison uses the exact same 450 held-out request windows and starting-head values. The common rows are saved in `results/raw/common_workload_manifest.csv`. Inside the shift and generalization tests, each algorithm also receives the identical window before costs are compared.

## 4. Results

The held-out selector accuracy were 0.858. Its mean movement was 186.22 cylinders per window, while oracle movement was 182.02; mean regret was 4.20. Confidence averaged 0.945 on correct predictions and 0.887 on wrong predictions. Calibraton ECE was 0.140 and multiclass Brier score was 0.263.

 Table files are in `results/tables/`. The main comparison is shown in `results/figures/classical_scheduler_comparison.png`, with prediction errors in `confusion_matrix.png` and calibration in `confidence_calibration.png`.

   Before shift, accuracy was 0.933, confidence 0.927, movement 159.27, and regret 0.13. After shift these value became 0.900, 0.969, 196.37, and 0.40. The marked timeline is in `results/figures/workload_shift_timeline.png`.

## 5. Analysis

**Observation:** The best fixed schedulers by workload were bursty: SSTF, random: SSTF, sequential: SSTF. The selector did not reach oracle performance, and its measured penalty was 4.20 cylinders per window.

No C-SCAN window was the ground-truth winner in the held-out set, so its per-class support and score are zero. This result was kept as measured instead of creating artificial examples to make every class appear.

**Interpretation:** SSTF can reduce local movment because it repeatedly takes a nearby request. SCAN gives a structured sweep, while C-SCAN can pay a big wrap cost under this metric. FCFS changes strongly with arrival order. The confidence gap between correct and wrong cases gives some useful evidence, but the ECE and reliability plot must also be checked before calling it well cal ibrated [4].

  The distribution-shift test use stronger burst concentration and different sequential noise. This is still synthetic, so it shows generalization only inside this simulator. The project also models a simple cylinder line and not queue arrival times, rotational delay or real device caching.

## 6. Conclusion

The experiment showed how a decision tree can select among four disk schedulers using only one incoming request window. It measured both label correctness and the actual movement cost of mistakes. Results changed across the sequential-to-random shift, which supports checking confidence and regret together instead of accuracy alone. These findings apply to classical mechanical-disk simulation; modern SSDs do not have the same head movement.

## 7. AI Assistance Disclosure

An AI coding assistant was used for software development, testing support and document drafting. The experiment numbers were generated by running the included code, and raw outputs are saved for checking.

## References

1. A. Silberschatz, P. B. Galvin, and G. Gagne, *Operating System Concepts*, 10th ed., Wiley, 2018.
2. L. Breiman, J. H. Friedman, R. A. Olshen, and C. J. Stone, *Classification and Regression Trees*, Wadsworth & Brooks/Cole, 1984.
3. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825-2830, 2011.
4. C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On Calibration of Modern Neural Networks," *Proceedings of Machine Learning Research*, vol. 70, pp. 1321-1330, 2017.
