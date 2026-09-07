# Examples

Four worked examples live in the repository's
[`examples/`](https://github.com/ai-vnv/DORASolvers.jl/tree/main/examples)
directory. Run them from the repository root with

```bash
julia --project=examples -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=examples examples/01_gridworld_quickstart.jl
```

## 1. Gridworld quickstart

[`01_gridworld_quickstart.jl`](https://github.com/ai-vnv/DORASolvers.jl/blob/main/examples/01_gridworld_quickstart.jl)
solves a `SimpleGridWorld` from POMDPModels.jl with the solver defaults,
queries `action` and `value`, runs the planner under
`POMDPTools.RolloutSimulator`, and compares the planner's policy cost to the
optimal cost of the tabularized model.

The gap reported there is the relative excess cost
``(J_\pi - J^*) / J^*`` at the start state, with both terms computed exactly
by dynamic programming on the tabularized model. The test suite asserts that
this gap stays **below 1%** (`test/runtests.jl`); the example prints the value
actually attained, which on this instance is smaller than the asserted bound.

## 2. Online cost learning

[`02_online_learning.jl`](https://github.com/ai-vnv/DORASolvers.jl/blob/main/examples/02_online_learning.jl)
turns the known-costs seeding off and learns traversal costs from noisy
observations, in both modes: self-simulated training inside `solve` (plus
`train!` to continue), and deployment-time learning through `observe!` with
lazy replanning.

## 3. Warehouse navigation benchmark

[`03_warehouse_navigation.jl`](https://github.com/ai-vnv/DORASolvers.jl/blob/main/examples/03_warehouse_navigation.jl)
builds the paper's warehouse domain — a shelf grid with unknown terrain
costs, lateral actuation slip, and a hazard band of dynamic obstacles — and
drives the learner with the episode-level API (`plan!`, `run_episode!`),
tracking regret and planner work. It also demonstrates the
chance-constrained variant `RiskDORA`, whose risk multiplier is updated by
projected dual ascent on the realized contact indicator.

## 4. Custom MDP with explicit SSP structure

[`04_custom_mdp.jl`](https://github.com/ai-vnv/DORASolvers.jl/blob/main/examples/04_custom_mdp.jl)
defines a small river-crossing MDP and shows how to pass `classify`, `cost`,
`start`, and the penalty scales explicitly when the reward-based defaults do
not fit — for example, action-dependent costs where a careful move is slow
but safe and a long jump is fast but risky.

## 5. Duckietown: an online SSP over a continuous driving MDP

[`05_duckietown.jl`](https://github.com/ai-vnv/DORASolvers.jl/blob/main/examples/05_duckietown.jl)
drives [Duckietown.jl](https://github.com/ai-vnv/Duckietown.jl) — a
lane-following MDP with a stop sign and a crossing duck, validated
decision-by-decision against its Python reference — with DORA under a
receding horizon. Three ideas bridge the continuous world to a tabular SSP:
a measured-deterministic transition (fixed per-decision RNG), macro actions
that hold one command for eight physics decisions, and a state key that
pairs discretized pose with lap progress, so the loop becomes monotone
progress toward an absorbing goal. The solver call is the same explicit
`start` / `classify` / `cost` / `key` pattern as example 4. The full case
study — the determinism measurement, the timescale ablation, and a rendered
lap — lives in the Duckietown.jl repository as
[`notebooks/DORA_on_Duckietown.jl`](https://github.com/ai-vnv/Duckietown.jl/blob/main/notebooks/DORA_on_Duckietown.jl).
