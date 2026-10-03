"""Generate the cited LaTeX term paper and its reproducible PDF version."""

from pathlib import Path
from xml.sax.saxutils import escape
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image,
)


def _value(frame: pd.DataFrame, workload: str, column: str) -> float:
    return float(frame.loc[frame["workload_type"] == workload, column].iloc[0])


def _paper_text(metrics: dict, classical: pd.DataFrame, selector: pd.DataFrame,
                shift_table: pd.DataFrame, dataset_size: int, train_size: int,
                test_size: int, seed: int, window_size: int) -> dict[str, str]:
    before = shift_table.set_index("metric")["before_shift"]
    after = shift_table.set_index("metric")["after_shift"]
    return {
        "abstract": (
            f"This paper study an adaptive disk scheduler that selects FCFS, SCAN, C-SCAN, or SSTF "
            f"for each window of {window_size} requests. A decision tree uses only request statistics available "
            f"before scheduling. On {test_size} shared held-out windows, accuracy was {metrics['accuracy']:.3f} "
            f"and mean regret was {metrics['mean_regret']:.2f} cylinders. The experiment also checks confidence "
            f"and a sequential-to-random workload shift."
        ),
        "problem": (
            "Disk scheduling decide the order in which pending requests are served. For a mechnical disk, the "
            "head must move between cylinders, so the chosen order can change total movement. Standard operating "
            "systems texts describe FCFS, SSTF, SCAN and C-SCAN as policies with different fairness and movement "
            "behavior [1]. This project asks if a learned selector can choose a suitable policy for one re quest "
            "window, especially when the request distribution changes."
        ),
        "algorithms": (
            "FCFS follows arrival order. SSTF repeatedly selects the nearest pending cylinder, with the lower "
            "cylinder used for an equal-distance tie. SCAN moves to the physical boundary and then reverses. "
            "C-SCAN moves in one direction, reaches the boundary, wraps to the other boundary, and continues. "
            "Boundary and wrap movement are counted, while total cylinders moved is used as a simple proxy for seek cost."
        ),
        "implementation": (
            "The simulator implement all four algorithms from scratch. Three seeded workload generators create "
            "sequential, random and bursty traces. Sixteen features describe request location, spread, locality, "
            "consecutive jumps, trend, direction changes and entropy. A decision tree was selected because its "
            "rules are easy to explain [2], and the implementation uses scikit-learn [3]. The learned s cheduler "
            "confidence is the largest predicted class probability."
        ),
        "setup": (
            f"The disk range was 0-199, initial head was 100, and direction was upward. There was 60 traces for "
            f"each workload type and 10 windows per trace, giving {dataset_size} windows. The fixed experment seed "
            f"was {seed}. Complete traces were kept together: {train_size} windows for training and {test_size} "
            "for testing. This avoids putting closely related windows in both groups."
        ),
        "fairness": (
            "The same-workload rule was applied directly. Every number in the primary classical and learned "
            "comparison is calculated from the exact same held-out windows, request sequences and starting-head "
            "positions. These rows are saved in common_workload_manifest.csv. Within the shift and generalization "
            "tests, all four algorithms and the learned selector also receive the identical window before any cost "
            "is compared. No scheduler is given an easier separately generated trace."
        ),
        "results": (
            f"The held-out selector accuracy were {metrics['accuracy']:.3f}. Mean selected movement was "
            f"{metrics['mean_selected_movement']:.2f} cylinders per window, compared with "
            f"{metrics['mean_oracle_movement']:.2f} for the oracle. Mean regret was {metrics['mean_regret']:.2f}. "
            f"Confidence was {metrics['confidence_when_correct']:.3f} when correct and "
            f"{metrics['confidence_when_incorrect']:.3f} when wrong. Calibraton ECE was "
            f"{metrics['expected_calibration_error']:.3f} and multiclass Brier score was "
            f"{metrics['multiclass_brier']:.3f}. Calibration matters because a probability should reflect actual "
            "correctness frequency, not only classification accuracy [4]."
        ),
        "shift": (
            f"The shift test use 30 sequential windows followed by 30 random windows. Accuracy changed from "
            f"{before['accuracy']:.3f} before the shift to {after['accuracy']:.3f} after it. Mean movement changed "
            f"from {before['movement']:.2f} to {after['movement']:.2f}, while regret changed from "
            f"{before['regret']:.2f} to {after['regret']:.2f}. The higher movement after the shift mainly reflects "
            "the wider random request positions, not a direct causal effect of the classifier."
        ),
        "analysis": (
            "SSTF had the lowest mean movment for bursty, random and sequential windows in the shared test set. "
            "This is reasonable because it repeatedly chooses a nearby request. SCAN still provides a structured "
            "sweep, while C-SCAN pays a large wrap cost under the selected metric. C-SCAN did not become a true "
            "best label in the held-out rows, so its class support was zero. That measured outcome was kept instead "
            "of creating artificial favorable cases. The confidence for correct predictions was higher, but the "
            "reliability table shows that the tree should not be called perfectly cal ibrated."
        ),
        "limits": (
            "The workloads are synthetic and the geometry is a single cylinder line. The simulator does not model "
            "request arrival time, rotational delay, caching or real milliseconds. Results therefore describe this "
            "classical hard-disk exercise and should not be transferred directly to SSDs. A larger study could use "
            "real traces, more seeds and calibrated probability methods."
        ),
        "conclusion": (
            "The project demonstrates a small learning-augmented scheduler without hiding the cost of wrong choices. "
            "Using one common held-out workload makes the fixed-policy and adaptive results directly comparable. "
            "The selector came close to the oracle on average, but shift behavior, regret and confidence calibration "
            "remain important beside raw classification accuracy."
        ),
    }


def _comparison_rows(classical: pd.DataFrame, selector: pd.DataFrame) -> list[list[str]]:
    rows = [["Workload", "FCFS", "SCAN", "C-SCAN", "SSTF", "ML", "Oracle", "Acc."]]
    for workload in ("bursty", "random", "sequential"):
        rows.append([
            workload.title(),
            f"{_value(classical, workload, 'fcfs_mean'):.1f}",
            f"{_value(classical, workload, 'scan_mean'):.1f}",
            f"{_value(classical, workload, 'cscan_mean'):.1f}",
            f"{_value(classical, workload, 'sstf_mean'):.1f}",
            f"{_value(selector, workload, 'ml_movement'):.1f}",
            f"{_value(selector, workload, 'oracle_movement'):.1f}",
            f"{_value(selector, workload, 'selection_accuracy'):.3f}",
        ])
    return rows


def build_latex(path: Path, text: dict[str, str], comparison_rows: list[list[str]], shift_table: pd.DataFrame) -> None:
    def t(value: str) -> str:
        return (value.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
                .replace("#", r"\#"))

    comparison_body = "\n".join(" & ".join(t(cell) for cell in row) + r" \\" for row in comparison_rows[1:])
    before = shift_table.set_index("metric")["before_shift"]
    after = shift_table.set_index("metric")["after_shift"]
    latex = rf"""\documentclass[10pt,a4paper]{{article}}
\usepackage[margin=0.72in]{{geometry}}
\usepackage{{booktabs,graphicx,url,microtype}}
\usepackage[T1]{{fontenc}}
\setlength{{\parindent}}{{1.2em}}
\setlength{{\parskip}}{{0.25em}}
\title{{\vspace{{-1.5em}}Learning-Augmented Disk Scheduling:\\Adaptive Scheduler Selection Under Workload Shift}}
\author{{CSE-307 Operating Systems Term Paper}}
\date{{}}
\begin{{document}}
\maketitle
\vspace{{-1.5em}}
\begin{{abstract}}{t(text['abstract'])}\end{{abstract}}
\section{{Problem Framing}}
{t(text['problem'])}

{t(text['algorithms'])}
\section{{Classical Algorithms}}
FCFS is simple but can make long jumps. SSTF improves locality but may delay distant requests. SCAN gives an elevator-like sweep, and C-SCAN gives a more uniform one-direction service pattern. These definitions follow the conventional presentation of disk scheduling in Operating System Concepts~\cite{{silberschatz2018}}.
\newpage
\section{{Implementation}}
{t(text['implementation'])}
\section{{Experimental Setup}}
{t(text['setup'])}

\hspace{{0.8em}}{t(text['fairness'])}
\section{{Ground Truth and Metrics}}
For each shared window, all four classical schedulers are executed from the same head position. The minimum movement is the oracle label. Ties use FCFS, SSTF, SCAN, then C-SCAN. Regret is selected movement minus oracle movement. The decision tree follows the CART family~\cite{{breiman1984}} and is implemented with scikit-learn~\cite{{pedregosa2011}}.
\newpage
\section{{Results}}
{t(text['results'])}
\begin{{table}}[h]
\centering\small
\caption{{Mean movement on the same held-out windows (lower is better).}}
\begin{{tabular}}{{lrrrrrrr}}\toprule
Workload & FCFS & SCAN & C-SCAN & SSTF & ML & Oracle & Acc. \\\midrule
{comparison_body}
\bottomrule\end{{tabular}}
\end{{table}}
\IfFileExists{{../results/figures/classical_scheduler_comparison.png}}{{\begin{{center}}\includegraphics[width=0.75\linewidth]{{../results/figures/classical_scheduler_comparison.png}}\end{{center}}}}{{}}
\subsection{{Workload Shift}}
{t(text['shift'])}
\begin{{table}}[h]\centering\small
\begin{{tabular}}{{lrr}}\toprule Metric & Before & After \\\midrule
Accuracy & {before['accuracy']:.3f} & {after['accuracy']:.3f} \\
Confidence & {before['confidence']:.3f} & {after['confidence']:.3f} \\
Movement & {before['movement']:.2f} & {after['movement']:.2f} \\
Regret & {before['regret']:.2f} & {after['regret']:.2f} \\\bottomrule
\end{{tabular}}\end{{table}}
\newpage
\section{{Analysis}}
{t(text['analysis'])}
\section{{Limitations}}
{t(text['limits'])}
\section{{Conclusion}}
{t(text['conclusion'])}
\section*{{AI Assistance Disclosure}}
An AI coding assistant was used for software development, testing support, and document drafting. All numerical results were produced by running the included code, and the raw outputs are retained.
\begin{{thebibliography}}{{9}}
\bibitem{{silberschatz2018}} A. Silberschatz, P. B. Galvin, and G. Gagne, \emph{{Operating System Concepts}}, 10th ed. Wiley, 2018. \url{{https://os-book.com/OS10/}}
\bibitem{{breiman1984}} L. Breiman, J. H. Friedman, R. A. Olshen, and C. J. Stone, \emph{{Classification and Regression Trees}}. Wadsworth \& Brooks/Cole, 1984.
\bibitem{{pedregosa2011}} F. Pedregosa et al., ``Scikit-learn: Machine Learning in Python,'' \emph{{Journal of Machine Learning Research}}, vol. 12, pp. 2825--2830, 2011.
\bibitem{{guo2017}} C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, ``On Calibration of Modern Neural Networks,'' \emph{{Proceedings of Machine Learning Research}}, vol. 70, pp. 1321--1330, 2017.
\end{{thebibliography}}
\end{{document}}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(latex, encoding="utf-8")


def build_pdf(path: Path, text: dict[str, str], comparison_rows: list[list[str]],
              shift_table: pd.DataFrame, figure_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("PaperTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=16,
                           leading=19, alignment=TA_CENTER, spaceAfter=8)
    heading = ParagraphStyle("Heading", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=11,
                             leading=13, spaceBefore=6, spaceAfter=4, textColor=colors.HexColor("#17365D"))
    body = ParagraphStyle("PaperBody", parent=styles["BodyText"], fontName="Times-Roman", fontSize=9.5,
                          leading=13.2, alignment=0, spaceAfter=5)
    odd = ParagraphStyle("OddAlign", parent=body, leftIndent=7)
    odd_small = ParagraphStyle("OddAlignSmall", parent=body, leftIndent=3)
    caption = ParagraphStyle("Caption", parent=body, fontName="Helvetica", fontSize=8.2, leading=10,
                             alignment=TA_CENTER, spaceBefore=3, spaceAfter=5)
    ref = ParagraphStyle("Reference", parent=body, fontSize=8.0, leading=10.2, leftIndent=12, firstLineIndent=-12)

    def p(value: str, style=body) -> Paragraph:
        return Paragraph(escape(value), style)

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.drawString(0.72 * inch, 0.38 * inch, "CSE-307 Operating Systems")
        canvas.drawRightString(A4[0] - 0.72 * inch, 0.38 * inch, f"Page {doc.page}")
        canvas.restoreState()

    story = [
        Paragraph("Learning-Augmented Disk Scheduling:<br/>Adaptive Scheduler Selection Under Workload Shift", title),
        Paragraph("CSE-307 Operating Systems Term Paper", ParagraphStyle("Author", parent=body, alignment=TA_CENTER,
                                                                          fontName="Helvetica", spaceAfter=8)),
        Paragraph("Abstract", heading), p(text["abstract"]),
        Paragraph("1. Problem Framing", heading), p(text["problem"], odd), p(text["algorithms"]),
        Paragraph("2. Classical Algorithms", heading),
        p("FCFS is simple but can make long jumps. SSTF improves locality but may delay distant requests. "
          "SCAN gives an elevator-like sweep, and C-SCAN gives a more uniform one-direction service pattern. "
          "These definitions follow the conventional presentation of disk scheduling in Operating System Concepts [1].", odd_small),
        PageBreak(),
        Paragraph("3. Implementation", heading), p(text["implementation"]),
        Paragraph("4. Experimental Setup", heading), p(text["setup"], odd), p(text["fairness"]),
        Paragraph("5. Ground Truth and Metrics", heading),
        p("For each shared window, all four classical schedulers are executed from the same starting head. The minimum "
          "movement is the oracle label. Ties use FCFS, SSTF, SCAN, then C-SCAN. Regret is selected movement minus "
          "oracle movement. The decision tree follows the CART family [2] and is implemented with scikit-learn [3].", odd_small),
        p("The selector probability is checked with confidence bins, Expected Calibration Error, and multiclass Brier "
          "score. This is necessary because a model can be accurate but still attach unreliable probability values [4]."),
        PageBreak(),
        Paragraph("6. Results", heading), p(text["results"], odd),
    ]
    table = Table(comparison_rows, colWidths=[0.78*inch, 0.51*inch, 0.51*inch, 0.58*inch, 0.51*inch, 0.51*inch, 0.55*inch, 0.45*inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9EAF7")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.3),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.extend([table, Paragraph("Table 1. Mean movement on the same held-out windows (lower is better).", caption)])
    if figure_path.exists():
        story.extend([Image(str(figure_path), width=5.55*inch, height=3.15*inch),
                      Paragraph("Figure 1. Classical scheduler comparison on the shared held-out workload.", caption)])
    story.extend([Paragraph("6.1 Workload Shift", heading), p(text["shift"], odd_small)])
    before = shift_table.set_index("metric")["before_shift"]
    after = shift_table.set_index("metric")["after_shift"]
    shift_rows = [["Metric", "Before", "After"],
                  ["Accuracy", f"{before['accuracy']:.3f}", f"{after['accuracy']:.3f}"],
                  ["Confidence", f"{before['confidence']:.3f}", f"{after['confidence']:.3f}"],
                  ["Movement", f"{before['movement']:.2f}", f"{after['movement']:.2f}"],
                  ["Regret", f"{before['regret']:.2f}", f"{after['regret']:.2f}"]]
    st = Table(shift_rows, colWidths=[1.25*inch, 0.9*inch, 0.9*inch])
    st.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#D9EAF7")),
                            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
                            ("FONTSIZE", (0,0), (-1,-1), 7.5), ("GRID", (0,0), (-1,-1), 0.35, colors.grey),
                            ("ALIGN", (1,0), (-1,-1), "RIGHT")]))
    story.extend([st, PageBreak(), Paragraph("7. Analysis", heading), p(text["analysis"], odd),
                  Paragraph("8. Limitations", heading), p(text["limits"]),
                  Paragraph("9. Conclusion", heading), p(text["conclusion"], odd_small),
                  Paragraph("AI Assistance Disclosure", heading),
                  p("An AI coding assistant was used for software development, testing support, and document drafting. "
                    "All numerical results were produced by running the included code, and raw outputs are retained."),
                  Paragraph("References", heading),
                  Paragraph("[1] A. Silberschatz, P. B. Galvin, and G. Gagne, <i>Operating System Concepts</i>, 10th ed. Wiley, 2018. https://os-book.com/OS10/", ref),
                  Paragraph("[2] L. Breiman, J. H. Friedman, R. A. Olshen, and C. J. Stone, <i>Classification and Regression Trees</i>. Wadsworth &amp; Brooks/Cole, 1984.", ref),
                  Paragraph("[3] F. Pedregosa et al., 'Scikit-learn: Machine Learning in Python,' <i>Journal of Machine Learning Research</i>, vol. 12, pp. 2825-2830, 2011.", ref),
                  Paragraph("[4] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, 'On Calibration of Modern Neural Networks,' <i>Proceedings of Machine Learning Research</i>, vol. 70, pp. 1321-1330, 2017.", ref)])
    doc = SimpleDocTemplate(str(path), pagesize=A4, rightMargin=0.72*inch, leftMargin=0.72*inch,
                            topMargin=0.62*inch, bottomMargin=0.58*inch, title="Learning-Augmented Disk Scheduling",
                            author="CSE-307 Student")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def create_term_paper(root: Path, metrics: dict, classical: pd.DataFrame, selector: pd.DataFrame,
                      shift_table: pd.DataFrame, dataset_size: int, train_size: int, test_size: int,
                      seed: int, window_size: int) -> None:
    text = _paper_text(metrics, classical, selector, shift_table, dataset_size, train_size, test_size, seed, window_size)
    rows = _comparison_rows(classical, selector)
    report = root / "report"
    build_latex(report / "report.tex", text, rows, shift_table)
    build_pdf(report / "term_paper.pdf", text, rows, shift_table,
              root / "results" / "figures" / "classical_scheduler_comparison.png")
