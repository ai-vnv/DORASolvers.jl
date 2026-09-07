# DORA driving a Duckietown robot: an online SSP over a continuous MDP.
#
# Duckietown.jl (https://github.com/ai-vnv/Duckietown.jl) is a lane-following
# MDP with a stop sign and a crossing duck, validated decision-by-decision
# against its Python reference. It is not in the General registry yet, so run
# this example from a shared scratch project:
#
#     julia -e 'using Pkg; Pkg.activate("duckie-dora"; shared=true);
#               Pkg.add(url="https://github.com/ai-vnv/DORASolvers.jl");
#               Pkg.add(url="https://github.com/ai-vnv/Duckietown.jl");
#               Pkg.add(["POMDPs", "POMDPTools"])'
#     julia --project=@duckie-dora examples/05_duckietown.jl
#
# The bridge below is the one the case-study notebook
# (Duckietown.jl/notebooks/DORA_on_Duckietown.jl) builds step by step, with
# every design choice measured there; this file is its distilled, script-form
# twin. Three ideas make a continuous driving world fit a tabular SSP:
#
# 1. MEASURED determinism. With a fixed per-decision RNG the transition is a
#    function of (state, action) — the duck's trigger draw is the only
#    stochastic element and the scenario pins it. So `transition` can honestly
#    return `Deterministic(...)`.
# 2. MACRO ACTIONS. One SSP action holds a command for K = 8 physics
#    decisions (1.6 s). Measured in the notebook: shorter macros collapse the
#    reachable graph (K = 2 explores 22 states); K = 8 branches properly.
# 3. PROGRESS IN THE KEY. States hash by (ring progress, discretized pose),
#    so driving around the loop is monotone progress toward an absorbing goal
#    — the shape DORA's stochastic-shortest-path formulation wants.
#
# The solver call itself is exactly the documented custom-MDP pattern from
# 04_custom_mdp.jl: explicit `start`, `classify`, `cost`, `key`.

using DORASolvers
using Duckietown
using POMDPs
using POMDPTools: Deterministic
using Random
using Printf

# ---------------------------------------------------------------------------
# The world: a stop sign and a duck that crosses; reward already encodes
# stopping and yielding, so the cheapest route IS the compliant one.
# ---------------------------------------------------------------------------
const CFG = scenario_config(:stop_and_duck_safe)
const BASE = DuckietownMDP(CFG; action_space = :discrete)
const TR = BASE.transition
const SCFG = TR.state_cfg
const ACTS = collect(POMDPs.actions(BASE))

# Ring of drivable tiles, ordered by angle around the loop; lap progress is
# counted as forward hops along this ring.
const RING = let m = initial_map(CFG)
    t = collect(drivable_tiles(m))
    cx = sum(first.(t)) / length(t)
    cy = sum(last.(t)) / length(t)
    sort(t; by = p -> atan(p[2] - cy, p[1] - cx))
end
const RIX = Dict(t => i for (i, t) in enumerate(RING))
const NRING = length(RING)

tile_of(s) = get_grid_coords(s.map, collect(s.ego.pos))
raw_of(s) = first(get_raw_state(s, SCFG))

function advance(prog, prev, new)
    new == prev && return prog
    a, b = get(RIX, prev, 0), get(RIX, new, 0)
    (a == 0 || b == 0) && return prog
    return mod(b - a, NRING) == 1 ? prog + 1 : prog
end

# ---------------------------------------------------------------------------
# Macro step and the SSP. Cost is DORA's documented default shape,
# max(c_min, 1 - reward): the model's shaped reward supplies the safety
# semantics, nothing safety-related is hand-coded here.
# ---------------------------------------------------------------------------
const K = 8
const C_MIN = 0.05
const GOAL = 2          # ring tiles to complete; set NRING for a full lap
step_cost(r) = max(C_MIN, 1.0 - r.reward.total)

struct LapState
    s::DuckieWorldState
    prog::Int
end

function macro_step(ls::LapState, a)
    s = ls.s; prog = ls.prog; tile = tile_of(s); c = 0.0
    for _ in 1:K
        r = simulate_decision(TR, s, a, MersenneTwister(1))
        c += step_cost(r)
        prog = advance(prog, tile, tile_of(r.sp))
        tile = tile_of(r.sp); s = r.sp
        (r.terminated || r.truncated || prog >= GOAL) &&
            return (LapState(s, prog), c)
    end
    return (LapState(s, prog), c)
end

struct LapMDP <: MDP{LapState,MacroAction} end
POMDPs.actions(::LapMDP) = ACTS
POMDPs.discount(::LapMDP) = 1.0
POMDPs.isterminal(::LapMDP, ls) = POMDPs.isterminal(BASE, ls.s)
POMDPs.transition(::LapMDP, ls, a) = Deterministic(first(macro_step(ls, a)))

lapkey(ls) = (min(ls.prog, GOAL), discretize(raw_of(ls.s)))
classify(ls) = POMDPs.isterminal(BASE, ls.s) ? :crash :
               ls.prog >= GOAL ? :goal : :normal

plan_from(from) = solve(DORASolver(
    start = from, classify = classify,
    cost = (ls, a, lsp) -> macro_step(ls, a)[2],
    key = lapkey,
    c_min = K * C_MIN, c_to = 2000.0, c_crash = 1000.0, horizon = 60,
), LapMDP())

# ---------------------------------------------------------------------------
# Receding horizon: re-plan every macro decision on the state the world
# actually reached, then hold the chosen command for K decisions.
# ---------------------------------------------------------------------------
function drive(; seed = 1001, max_dec = 400)
    rng = MersenneTwister(seed)
    s0 = rand(MersenneTwister(seed), initialstate(BASE))
    ls = LapState(s0, 0)
    total, dec, plans = 0.0, 0, 0
    while dec < max_dec
        POMDPs.isterminal(BASE, ls.s) && return (:crash, total, dec, plans)
        ls.prog >= GOAL && return (:goal, total, dec, plans)
        t0 = time()
        pl = plan_from(ls)
        plans += 1
        a = action(pl, ls)
        @printf("plan %2d: %4d states (%.1f s) -> %s\n",
            plans, pl.tab.S, time() - t0, string(a))
        s = ls.s; prog = ls.prog; tile = tile_of(s)
        for _ in 1:K
            dec += 1
            r = simulate_decision(TR, s, a, rng)
            total += step_cost(r)
            prog = advance(prog, tile, tile_of(r.sp))
            tile = tile_of(r.sp); s = r.sp
            (r.terminated || r.truncated || prog >= GOAL) && break
        end
        ls = LapState(s, prog)
    end
    return (:timeout, total, dec, plans)
end

outcome, total, dec, plans = drive()
@printf("\noutcome %s after %d decisions (%d plans), accumulated cost %.2f\n",
    outcome, dec, plans, total)
@assert outcome === :goal
println("DORA reached $(GOAL) ring tiles of :stop_and_duck_safe — ",
    "the notebook drives the full $(NRING)-tile lap and renders it.")
