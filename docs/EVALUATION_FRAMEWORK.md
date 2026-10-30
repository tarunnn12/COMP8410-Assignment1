# Evaluation framework (shared by all four pairs)

Every pair is evaluated with the same structure so that the results stay comparable. The framework was written before the runs started. The choices that belong to one pair (data, metrics, settings) are in its README and in the report.

## 1. Principles

- The AI models and techniques are evaluated, not general hardware or software engineering.
- Both models in a pair do the same task on the same inputs with the same seeds and settings.
- Every number in the report traces to a file in `pairs/pN_*/results/` that was produced by a script or notebook in `pairs/pN_*/src/`.
- Qualitative and subjective statements need a defined rule or a recorded observation. A statement without evidence is not made.
- Limits are stated with each result: sample size, data source, hardware, and what was not tested.
- Business figures that rely on assumptions (prices, labour time) are labelled as assumptions.

## 2. Model profile

A profile is written for each model and the two are compared side by side.

| Item | Question answered |
|---|---|
| Category | Which category of models does it belong to, and why this one? |
| AI objectives | Technical goal and business goal in the retailer setting |
| AI tasks | Task performed and why this task |
| AI functions | Functions the model implements (inputs, outputs, controls) |
| AI framework | Architecture, workflow, processing stages |
| AI algorithms | Core algorithms of the model |
| AI governance | Safety filters, licence terms, data provenance, privacy, misuse risk |
| AI advantages | Where it outperforms competitors, backed by a cited source or an own result |

## 3. Evaluation dimensions

| # | Dimension | Evidence |
|---|---|---|
| D1 | Role and appropriateness against technical trends | Literature and timeline |
| D2 | Technical performance (quantified) | Metrics, tables and figures |
| D3 | Business performance (quantified) | Cost, throughput and success rate per task |
| D4 | Competitiveness and uniqueness against other suitable models | Baseline comparison, cited benchmarks |
| D5 | Social, economic and environmental benefits | Energy use per output, cost, with the assumptions listed |
| D6 | Feedback | Own structured account of each pair (section 6) |
| D7 | System gaps and disadvantages | Failure cases with examples |
| D8 | Governance and security | Licence review, filter behaviour, data provenance, injection probe where relevant |
| D9 | Risk of safety, privacy and ethics | Risks with likelihood and impact |
| D10 | Fairness and trustworthiness | Probe sets and consistency across seeds where relevant |
| D11 | Interpretation of the difference between the two models | Analysis of why A and B differ |
| D12 | Recommendations for improvement | Ranked actions with justification |

## 4. Results standard

Each pair documents:
1. Settings: hardware, software versions, precision, resolution, steps and seeds.
2. Baselines or benchmarks: the competing model, and published reference numbers where they exist (cited).
3. Metrics: definition, direction (higher or lower is better) and the reason for the choice.
4. Data: source, size, licence, selection method, and a script to rebuild it.
5. Dimensions covered: which of D1 to D12.
6. Presentation: tables, figures with labelled axes and side-by-side qualitative samples.

Statistics: intervals (Wilson for proportions, bootstrap for differences) and paired tests (exact McNemar for yes or no outcomes, Wilcoxon signed rank for scores). With very small samples this is stated and no significance is claimed. Every pair is run twice (a second seed or a second random sample) and the two runs are pooled.

Shared measurements where they apply: latency per item, peak GPU memory, model size on disk, GPU energy and failure rate (crash, out of memory, invalid output).

## 5. Reflection standard

- Applicability to the retailer scenario and the conditions under which the model would fail.
- Gaps and open issues found in testing.
- Alternatives the organisation could consider.
- Future directions, and what would be tested with more time or hardware.

## 6. Feedback protocol

Feedback is an own, structured account of each pair:

- What was done with each model (set-up effort, failures met, restarts, time spent waiting) and what each output looked like in use.
- The failure cases that were examined one by one (the failure files in `results/`), with the reason where one was found.
- Where a reading of the outputs disagreed with an automatic metric, and what was done about it.
- What a person who relies on the output would need (for example a review step), and what would be changed.

Every statement links to a file in `results/`, to the work log or to a figure. Feedback from one person's evaluation is not generalised to other users.

## 7. Structure of each pair section

1. Scenario and objectives
2. Models: profile table and category
3. Evaluation design
4. Quantitative results
5. Failure analysis
6. Feedback, governance, risk, fairness and trust
7. Business view, benefits and competitiveness
8. Reflection and recommendations
9. Reproduction steps (the pair README)
