# Evidence-Verified Hybrid Program Synthesis for ARC-AGI-2

> Paper Track draft. Keep the final Kaggle Writeup within the competition limit and replace every bracketed placeholder only with measured results from the submitted notebook.

## Subtitle
Object-centric abstractions, compositional DSL search, learned program priors, and safe neural-symbolic proposals under deterministic demonstration verification.

## Abstract
We study ARC-AGI-2 as test-time program induction. Instead of mapping grids directly to outputs with a single opaque predictor, our system separates proposal, execution, verification, and ranking. A compact domain-specific language contains geometric, color, object-centric, counting, and symmetry operations. The solver generates task-derived transformations and bounded two-step compositions, executes them on every demonstration, deduplicates behaviorally equivalent programs, and ranks hypotheses by exact demonstration consistency, partial agreement, learned program priors, and description complexity. ARC's two-attempt interface is used for genuine hypothesis diversity. We additionally define a safe neural-symbolic bridge in which an LLM or learned proposer may emit only approved DSL operation names; arbitrary code execution is forbidden and every proposed program must reproduce all demonstrations before it can influence a test prediction. On the submitted ARC-AGI-2 notebook this system achieves [LEADERBOARD_SCORE], submission ID [SUBMISSION_ID].

## 1. Motivation
ARC tasks provide a small number of input-output demonstrations and require exact generalization to new inputs. We model a task as latent program induction. Given demonstrations D={(x_i,y_i)}, the objective is to identify a compact program p from a hypothesis space H such that p(x_i)=y_i for all observed pairs and then apply p to the unseen test input.

The design hypothesis is that ARC generalization benefits from separating two roles:

1. **Proposal:** generate plausible abstractions and programs.
2. **Verification:** reject programs that contradict observed evidence.

This separation allows heuristics, learned priors, retrieval systems, or language models to improve search without being trusted blindly.

## 2. Representation
A grid is represented as a rectangular matrix of color IDs. The solver also constructs object-centric representations from connected components. Each object stores its color, cells, size, and bounding box. This supports operations that refer to the largest or smallest object, isolate one component, fill its bounding box, count components, and form structural signatures.

The initial DSL includes identity, rotations, reflections, transpose variants, background-aware crop, scaling, tiling, color substitution, object selection, foreground filtering, symmetry completion, outlining, count rendering, and simple object ordering.

## 3. Task-Derived Programs
Some ARC transformations contain parameters that cannot be fixed globally. The system infers parameters from demonstrations, including color maps, foreground recolor targets, scaling factors, tile factors, and count-rendering orientation/color. A derived program is retained only when it can be executed safely and its behavior is evaluated against every demonstration.

## 4. Compositional Search
Single primitives are insufficient for tasks such as "extract an object and then reflect it." We therefore perform bounded two-step composition. Candidate programs are behaviorally deduplicated on the demonstrations to reduce equivalent search paths.

For candidate p, each demonstration contributes an exact or partial agreement score. Exact consistency receives highest priority. Partial agreement is used only when no program completely explains the demonstrations and provides a more informative fallback than unconditional identity.

The ranking order is:

1. exact consistency across demonstrations;
2. mean demonstration agreement;
3. optional learned symbolic prior;
4. lower program complexity;
5. deterministic tie-breaking.

This is an Occam-style prior: when multiple programs explain the same evidence, prefer the shorter one unless training-derived evidence supports another candidate.

## 5. Learned Program Priors
The public ARC training challenges are used to estimate empirical priors over symbolic programs. For each training task, exact-fitting programs receive fractional credit. These frequencies are later used only as a small ranking tie-breaker.

The learned prior never overrides a candidate with stronger task-local demonstration fit. Consequently, the learned component proposes a preference over abstractions while the current task remains the final authority.

## 6. Neural-Symbolic Proposal Interface
The architecture contains an optional `SymbolicProposal` interface for external learned models or LLMs. A proposer may output a sequence such as:

`crop_largest_object -> flip_h`

The sequence is compiled only from an allow-listed operation registry. No generated Python or shell code is executed. The compiled program must then reproduce all demonstrations before it is accepted as an exact candidate.

This creates a controlled path for future neural proposal models while retaining deterministic symbolic verification and notebook reproducibility.

## 7. Two-Attempt Test-Time Reasoning
ARC-AGI-2 permits two output attempts. We treat these as two distinct hypotheses rather than duplicated outputs. Candidate programs are ranked once per task; test outputs are then deduplicated by predicted grid so `attempt_2` provides alternative coverage whenever observationally equivalent training programs diverge on the unseen input.

The solver records a trace for every prediction containing the selected program names, demonstration scores, exact-fit flags, and candidate count. These traces support failure analysis and make the final paper auditable.

## 8. Experimental Protocol
Report the following ablations using the same public evaluation split and exact-output metric:

| Version | Added capability | Exact-output accuracy | Fully solved tasks |
|---|---|---:|---:|
| v0 | single-step geometric/color baseline | [V0_ACC] | [V0_TASKS] |
| v1 | + object-centric reasoning | [V1_ACC] | [V1_TASKS] |
| v2 | + symmetry/counting/relational primitives | [V2_ACC] | [V2_TASKS] |
| v3 | + two-step compositional search | [V3_ACC] | [V3_TASKS] |
| v4 | + learned symbolic priors | [V4_ACC] | [V4_TASKS] |
| v5 | + externally proposed verified DSL programs, if used | [V5_ACC] | [V5_TASKS] |

Final ARC-AGI-2 public leaderboard score: **[LEADERBOARD_SCORE]**  
Kaggle submission ID: **[SUBMISSION_ID]**  
Local evaluation exact-output accuracy: **[LOCAL_ACCURACY]**  
Fully solved evaluation tasks: **[SOLVED_TASKS] / [TOTAL_TASKS]**

Only report values generated by the actual submitted code.

## 9. Why the Approach May Generalize
The system does not memorize task IDs or hidden outputs. It searches reusable transformations and validates them against each new task's demonstrations. The abstraction layer is modular: richer topology, path reasoning, scene graphs, iterative simulators, or learned proposal distributions can be added without changing the evidence-verification contract.

This modularity is the main universality claim. The current DSL is ARC-oriented, but the proposal-verification architecture applies more broadly to few-shot structured reasoning problems where candidate programs can be executed and checked.

## 10. Limitations
The solver remains intentionally bounded. It will fail tasks requiring concepts outside the DSL, deep compositions, iterative dynamics, sophisticated topology, long-range object correspondence, or latent semantics not captured by connected components and simple relations. Exact consistency on only a few demonstrations can also leave multiple programs observationally equivalent, and a short-program prior can prefer the wrong extrapolation.

The learned prior is lightweight rather than a large neural model. This improves reproducibility and offline Kaggle execution, but it cannot discover a missing abstraction by itself.

## 11. Novelty Position
Symbolic program synthesis and object-centric ARC solving are established ideas. The contribution should therefore be claimed narrowly: a reproducible evidence-verified hybrid architecture combining object abstractions, bounded compositional search, demonstration-sensitive fallback scoring, learned symbolic priors, genuine two-attempt diversity, prediction traces, and a safe allow-listed neural-symbolic proposal interface.

The final novelty discussion should compare measured behavior and engineering choices with relevant public ARC research and 2026 competition solutions rather than claiming that the underlying paradigm is new.

## 12. Conclusion
ARC-AGI-2 tests adaptation when the rule changes from task to task. Our system treats that adaptation as executable hypothesis search constrained by evidence. The current implementation expands a simple symbolic baseline into an object-centric, compositional, partially learned, and auditable hybrid solver while preserving deterministic verification. The remaining research problem is empirical: identify recurring failure classes, add reusable abstractions rather than task-specific patches, and measure whether each addition improves unseen-task generalization.
