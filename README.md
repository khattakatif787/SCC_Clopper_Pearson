# SCC Clopper-Pearson

Implementation of Clopper-Pearson-bounded Self-Checking Classifiers (SCCs) and ordered Self-Checking Classifier Ensembles (SCCEs).

## Overview

This repository provides Python implementations for constructing, evaluating, and comparing statistically bounded SCCs and ordered SCCEs.

## Python Scripts

| Script | Description |
|---|---|
| construct_cp_bounded_scc.py | Construct SCCs using one-sided Clopper-Pearson bounds. |
| evaluate_scc_performance.py | Evaluate correct, misclassified, and rejected predictions. |
| optimize_ordered_scce.py | Optimize ordered SCC ensembles of sizes 2 and 3. |
| summarize_scc_scce_results.py | Generate Excel summaries of SCC and SCCE performance. |
| analyze_validation_size_sensitivity.py | Analyze the effects of validation dataset size. |
| analyze_scce_gain_drop.py | Analyze gains and additional misclassifications in ensembles. |

## Requirements

Python 3.10 or later.

Install dependencies:

    python -m pip install numpy pandas scipy openpyxl

## Experimental Settings

Individual SCC target ALRs: 0.01, 0.001, 0.0001.

Confidence levels: 90%, 95%, 99%, 99.9%.

Ordered SCCE sizes: 2 and 3.

The ensemble accepts the first non-rejected prediction and rejects only when every constituent SCC rejects.

## Execution

Run the main scripts in order:

    python construct_cp_bounded_scc.py
    python evaluate_scc_performance.py
    python optimize_ordered_scce.py
    python summarize_scc_scce_results.py

Additional analyses:

    python analyze_validation_size_sensitivity.py --preflight
    python analyze_validation_size_sensitivity.py
    python analyze_scce_gain_drop.py

## Experimental Results

Selected experimental results are stored in the results directory.

- validation_classifier_scc_ensemble_comparison.xlsx
- single_scc_vs_scce_cp_bound_matched.xlsx

### Color Interpretation

- Blue: No qualified single SCC exists, but a qualified SCCE exists.
- Green: SCCE has fewer rejections than the single SCC under the matched CP-bound setting.
- White with numeric value: Single SCC has fewer rejections than the SCCE.
- White with empty value: No qualified SCC or SCCE is available.
- Equal rejection rates indicate a tie.

## Dataset and Path Configuration

Most scripts use:

    PROJECT_ROOT = Path.home() / "SAC_Extension_Clopper_Upper_Bound_2026-09-04"

Update the project path before running the scripts in a different environment.

The validation-size sensitivity script uses its own directory and expects the required datasets and base SCC script to be available there.

## Relationship to the Research Paper

This repository provides the implementation and supplementary experimental results for Clopper-Pearson-bounded Self-Checking Classifiers (SCCs) and ordered Self-Checking Classifier Ensembles (SCCEs).

The research paper reports experimental results obtained at a **95% confidence level**. To provide additional experimental evidence and support further analysis, this repository also includes evaluations at **90%, 99%, and 99.9% confidence levels**.

The experiments consider individual  ALRs of 0.01, 0.001, and 0.0001, as well as ordered SCC ensembles containing two or three SCCs.

The additional results allow readers to examine the effects of confidence level and ALR on SCC qualification, rejection behavior, and SCCE performance.

### Scope of Reported Results

- **Research paper:** Experimental results reported at 95% confidence.
- **GitHub repository:** Experimental results covering 90%, 95%, 99%, and 99.9% confidence.


## Related Repository: Empirical SCC Generation and Ensemble Evaluation

The empirical implementation of Self-Checking Classifiers (SCCs) and ordered Self-Checking Classifier Ensembles (SCCEs) is available in our related repository:

**[Empirical SCC Generation and Ensemble Evaluation](https://github.com/khattakatif787/scc-generation-and-ensembles/blob/main/README.md)**

The related repository focuses on SCC construction and ensemble evaluation using empirical acceptable ALR constraints.

The current repository extends this approach by incorporating **one-sided Clopper-Pearson upper confidence bounds**, providing statistical control of misclassification risk at specified confidence levels.

While the associated research paper reports results at **95% confidence**, this repository additionally provides supplementary experiments at **90%, 99%, and 99.9% confidence**.


## Related Repository: Empirical SCC Generation and Ensemble Evaluation

The empirical implementation of Self-Checking Classifiers (SCCs) and ordered Self-Checking Classifier Ensembles (SCCEs) is available in our related repository:

**[Empirical SCC Generation and Ensemble Evaluation](https://github.com/khattakatif787/scc-generation-and-ensembles/blob/main/README.md)**

The related repository focuses on SCC construction and ensemble evaluation using empirical acceptable loss-rate (ALR) constraints.

The current repository extends this approach by incorporating **one-sided Clopper-Pearson upper confidence bounds**, providing statistical control of misclassification risk at specified confidence levels.

While the associated research paper reports results at **95% confidence**, this repository additionally provides supplementary experiments at **90%, 99%, and 99.9% confidence**.
