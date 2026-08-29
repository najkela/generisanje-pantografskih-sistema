# Pantograph Synthesis — Design Notes

## Problem Definition

The goal is to find a pantograph linkage system — defined by a topology (number of links, which nodes are fixed, which is the crank) and a geometry (link lengths, pivot positions) — whose tracer node approximates a given target curve as closely as possible.

Before any optimisation can be attempted, the problem space must be clearly bounded. This means specifying:

- A **maximum node count**, which caps the complexity of the mechanism and the size of the search space
- The **degrees of freedom** treated as variable: link lengths, fixed pivot positions, and crank pivot position
- A **target curve** as a finite set of ordered sample points, with a fixed correspondence between sample index and crank angle
- A **loss function** — typically the mean squared distance between the tracer path and the target curve at corresponding angles

This problem belongs to a well-studied family known as **path synthesis** problems. Sleesongsom & Bureerat (2018) give a representative formulation: the objective is to minimise the sum of squared errors between the target path and the actual path of a tracer point on the mechanism, subject to constraints on the design variables [1]. Without these definitions fixed upfront, neither the topology search nor the geometry optimisation has a well-posed objective to work toward.

---

## Why a Combined Approach Is Necessary

The full synthesis problem splits into two distinct subproblems: choosing a **topology** (which nodes exist, how they are connected, which are fixed) and optimising a **geometry** (the actual lengths and positions for a given topology). These require fundamentally different methods, and neither alone is sufficient.

A genetic algorithm applied directly to continuous geometry parameters — link lengths and pivot positions — is a poor fit. Continuous real-valued search spaces have no natural notion of meaningful crossover, and GAs lack any mechanism for learning correlations between parameters. In a kinematic system, parameters are heavily coupled: changing one link length requires compensating adjustments elsewhere to preserve a valid mechanism. A GA will explore this space slowly and wastefully. This two-level character — discrete topology choice at the outer level, continuous geometry optimisation at the inner level — is an instance of **bilevel optimisation**, a structure that appears broadly in topology and structural design [2].

CMA-ES (Covariance Matrix Adaptation Evolution Strategy) is the right tool for this inner problem. Introduced by Hansen & Ostermeier (2001), CMA-ES maintains an explicit probabilistic model of the search space — a multivariate Gaussian distribution — and updates it each generation to reflect the shape and orientation of the fitness landscape [3]. If two parameters need to move together, the covariance matrix learns this and begins sampling along that correlation. This makes CMA-ES well suited to exactly the kind of tightly coupled, continuous, non-convex landscape that linkage geometry optimisation produces. It has since become one of the standard benchmarks in continuous black-box optimisation, and has been applied to real engineering design problems including lightweight automotive structural optimisation.

---

## The Two-Level Structure

### Outer Problem — Topology Search (Genetic Algorithm)

The genetic algorithm is responsible for searching over the discrete space of possible mechanism topologies. Each individual in the GA population represents a candidate graph: a set of nodes, an edge set, a designated crank node, and a set of fixed nodes. The GA proposes topologies; the inner optimiser evaluates them.

Valid mechanism topologies must satisfy several constraints — the graph must be connected, the kinematic chain must be solvable in a consistent dependency order (every floating node must be resolvable from known nodes outward), and the structure must be generically rigid. The GA must respect these constraints, which makes the design of variation operators non-trivial.

**Crossover is particularly problematic.** Splicing two parent graphs together rarely produces a child that satisfies all validity constraints, and repairing invalid graphs after crossover adds significant complexity. This difficulty is noted across the mechanism synthesis literature — for example, Lin (2010) reports that hybrid GA-based approaches for path synthesis often replace or heavily modify the standard crossover operation precisely because it does not transfer well to structured continuous or graph-valued domains [1]. For this reason it may be preferable to rely on **mutation only** — local perturbations such as adding or removing an edge, adding or removing a node, or reassigning the fixed/crank labels.

A cleaner alternative is **grammatical evolution**, in which the search operates over strings in a grammar that is defined to generate only valid linkage topologies. This eliminates the need for validity checks or repair entirely, since invalid graphs are unreachable by construction. The grammar encodes the structural rules of the mechanism, and the GA evolves grammar strings rather than graphs directly. This approach has been used effectively in analogous discrete-topology synthesis problems: Nasuf, Bhaskar & Keane (2013) apply grammatical evolution to structural shape optimisation, using a shape grammar to automatically constrain the search to geometrically valid designs [4], and Kunaver et al. demonstrate a similar approach for automated analog circuit synthesis, where the grammar guarantees that only valid circuit topologies are produced [5]. The graph grammar framework studied by Luerssen et al. (2007) extends this idea further, directly evolving the grammar itself to optimise graph-structured designs across domains including circuit design and symbolic regression [6].

### Inner Problem — Geometry Optimisation (CMA-ES)

Given a fixed topology from the outer search, CMA-ES optimises the continuous geometry parameters — link lengths and pivot coordinates — to minimise the tracing error against the target curve. For each crank angle in the discretised target, the kinematic solver propagates the crank position through the linkage using circle-circle intersection, and the loss is accumulated over all angles.

CMA-ES is run with the topology held fixed. It samples candidate geometries from its learned Gaussian distribution, evaluates each via the kinematic solver and loss function, and updates its covariance model accordingly. Nomura, Akimoto & Ono (2024) show that CMA-ES with learning rate adaptation handles multimodal and noisy objective functions — both characteristics of linkage geometry optimisation — robustly without expensive hyperparameter tuning [7].

In the context of the outer topology search, **it is not necessary to run CMA-ES to convergence for every candidate topology**. A short, fixed-budget CMA-ES run — enough to get a reasonable lower bound on the geometry loss for a given topology — is sufficient to score and rank topologies in the GA. Full convergence runs are reserved for the most promising topologies at the end of the search, as a final refinement step. This pattern, where an expensive inner optimiser is run cheaply during search and fully only at the end, is closely related to surrogate-assisted optimisation: Farhadi Khouzani et al. note that integrating surrogate or budget-limited inner evaluations with CMA-ES is a standard strategy when full function evaluations are costly [8].

---

## Summary of Responsibilities

| Component | Method | Searches Over |
|---|---|---|
| Topology search | Genetic algorithm (mutation-based or grammatical evolution) | Graph structure — nodes, edges, fixed/crank labels |
| Geometry evaluation | CMA-ES (short run) | Continuous parameters for a fixed topology |
| Final refinement | CMA-ES (full convergence) | Continuous parameters for selected topologies |

---

## References

[1] Sleesongsom, S. & Bureerat, S. (2018). Optimal Synthesis of Four-Bar Linkage Path Generation through Evolutionary Computation with a Novel Constraint Handling Technique. *Computational Intelligence and Neuroscience*. https://doi.org/10.1155/2018/5462563

[2] Pan, Z., Gao, X. & Wu, K. (2022). First-Order Bilevel Topology Optimization for Fast Mechanical Design. *arXiv preprint*. https://arxiv.org/abs/2204.06204

[3] Hansen, N. & Ostermeier, A. (2001). Completely Derandomized Self-Adaptation in Evolution Strategies. *Evolutionary Computation*, 9(2), 159–195. https://doi.org/10.1162/106365601750190398

[4] Nasuf, A., Bhaskar, A. & Keane, A.J. (2013). Grammatical evolution of shape and its application to structural shape optimisation. *Structural and Multidisciplinary Optimization*, 48, 187–199. https://doi.org/10.1007/s00158-013-0890-0

[5] Kunaver, M. et al. (2022). Grammatical Evolution-based Analog Circuit Synthesis. *Informacije MIDEM*. http://ojs.midem-drustvo.si/index.php/InfMIDEM/article/view/841

[6] Luerssen, M. et al. (2007). Graph design by graph grammar evolution. *Proceedings of the IEEE Congress on Evolutionary Computation*. https://www.researchgate.net/publication/221006517

[7] Nomura, M., Akimoto, Y. & Ono, I. (2024). CMA-ES with Learning Rate Adaptation. *ACM Transactions on Evolutionary Learning and Optimization*, 1(1). https://doi.org/10.1145/3698203

[8] Farhadi Khouzani, F. et al. (2025). CMA-ES with Radial Basis Function Surrogate for Black-Box Optimization. *arXiv preprint*. https://arxiv.org/abs/2505.16127
