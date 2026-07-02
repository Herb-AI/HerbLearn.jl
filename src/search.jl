# =============================================================================
# Search drivers: plain functions that run a guided search.
#
# Each driver fixes one combination of decision space (what the guidance says)
# and schedule (when the search listens); the guidance itself is just an
# argument, so its shape is checked by the compiler:
#
#   guided_bottom_up_search   weights::Vector    consulted once per task
#   value_guided_search       score::Function    consulted at every step
#   repl_beam_search          score::Function    consulted at every step,
#                                                beam instead of enumeration
#
# The searches are HerbSearch's bottom-up iterators: the vanilla CostBUSIterator
# for rule weights, and the two custom iterators in iterators.jl. All drivers
# take a HerbSpecification.Problem and return `(program, enumerated)` with
# `program === nothing` when the budget runs out.
# =============================================================================

"""
    Evaluator(grammar, spec; mod=Main)

Callable helper that executes programs on a spec's inputs.
`ev(program)` returns the vector of outputs (with `nothing` where execution
failed); `n_correct(ev, outputs)` counts satisfied examples.
`mod` is the module whose functions the grammar's rules call.
"""
struct Evaluator{G<:AbstractGrammar,S}
    grammar::G
    spec::S
    symtable::Dict{Symbol,Any}
end

Evaluator(grammar::AbstractGrammar, spec::AbstractVector{<:IOExample}; mod::Module=Main) =
    Evaluator(grammar, spec, grammar2symboltable(grammar, mod))

function (ev::Evaluator)(program::AbstractRuleNode)
    expr = rulenode2expr(program, ev.grammar)
    return Any[
        try
            execute_on_input(ev.symtable, expr, ex.in)
        catch
            nothing
        end
        for ex in ev.spec
    ]
end

n_correct(ev::Evaluator, outputs) =
    count(i -> outputs[i] !== nothing && outputs[i] == ev.spec[i].out, eachindex(ev.spec))

is_solution(ev::Evaluator, outputs) = n_correct(ev, outputs) == length(ev.spec)

"""
    weights_to_costs(weights; min_cost=1, scale=10) -> Vector{Int}

Map rule weights (any nonnegative numbers, higher = more likely) to the
positive integer costs the bottom-up search enumerates by: the highest weight
gets `min_cost`, a zero weight gets `min_cost + scale`. Wraps
[`scores_to_costs`](@ref) after normalizing by the maximum.
"""
function weights_to_costs(weights::AbstractVector{<:Real}; min_cost::Integer=1, scale::Integer=10)
    m = maximum(weights)
    scores = m > 0 ? weights ./ m : fill(1.0, length(weights))
    return scores_to_costs(scores; min_cost=min_cost, scale=scale)
end

# Pull programs from `iter` until one satisfies every example or the budget
# runs out. The iterators yield only start-type programs.
function _first_solution(iter, ev::Evaluator, max_programs::Integer)
    enumerated = 0
    for program in iter
        enumerated += 1
        is_solution(ev, ev(program)) && return program, enumerated
        enumerated >= max_programs && break
    end
    return nothing, enumerated
end

# ── driver 1: rule weights, consulted once per task ─────────────────────────

"""
    guided_bottom_up_search(grammar, start::Symbol, problem::Problem,
                            weights::AbstractVector{<:Real};
                            mod=Main, max_programs=100_000, max_cost=typemax(Int),
                            min_cost=1, scale=10) -> (program, enumerated)

Complete bottom-up search over `grammar`, enumerating by increasing cost,
where each rule's integer cost comes from its weight (see
[`weights_to_costs`](@ref)). This is the driver for every static, rule-weight
method: DeepCoder-style presence predictions, corpus priors, distilled LLM
priors. Pass uniform weights (`ones(length(grammar.rules))`) for the unguided
baseline.

Runs HerbSearch's `CostBUSIterator` with observational-equivalence pruning and
returns the first program satisfying every example of `problem`, or `nothing`
within the budget.
"""
function guided_bottom_up_search(grammar::AbstractGrammar, start::Symbol, problem::Problem,
                                 weights::AbstractVector{<:Real};
                                 mod::Module=Main, max_programs::Integer=100_000,
                                 max_cost::Integer=typemax(Int),
                                 min_cost::Integer=1, scale::Integer=10)
    ev = Evaluator(grammar, problem.spec; mod=mod)
    costs = weights_to_costs(weights; min_cost=min_cost, scale=scale)
    iter = CostBUSIterator(grammar, start, Int(max_cost), costs, p -> ev(p))
    return _first_solution(iter, ev, max_programs)
end

# ── driver 2: value scores, consulted at every step ─────────────────────────

"""
    value_guided_search(grammar, start::Symbol, problem::Problem, score;
                        mod=Main, max_programs=100_000, max_cost=typemax(Int),
                        penalty_scale=5) -> (program, enumerated)

Bottom-up search in which every newly built candidate is scored by
`score(program, outputs) -> Float64` in `[0, 1]` and banked at its structural
cost plus a penalty of `penalty_scale * (1 - score)` levels -- the BUSTLE
recipe: a low score *delays* a value, it never discards it, so the search
stays complete. Because a child's banked cost already includes its own
penalty, penalties compound through a program's subterms.

Runs [`BustleBUSIterator`](@ref), the vanilla cost-based BUS with only its
bank cost overridden.
"""
function value_guided_search(grammar::AbstractGrammar, start::Symbol, problem::Problem,
                             score::F;
                             mod::Module=Main, max_programs::Integer=100_000,
                             max_cost::Integer=typemax(Int), penalty_scale::Integer=5) where {F}
    ev = Evaluator(grammar, problem.spec; mod=mod)
    penalty(prog) = round(Int, penalty_scale * (1 - clamp(score(prog, ev(prog)), 0, 1)))
    iter = BustleBUSIterator(grammar, start;
                             penalty=penalty, max_cost=Int(max_cost),
                             program_to_outputs=(p -> ev(p)))
    return _first_solution(iter, ev, max_programs)
end

# ── driver 3: action scores, beam search over REPL states ───────────────────

"""
    repl_beam_search(grammar, start::Symbol, problem::Problem, score;
                     mod=Main, max_programs=100_000, beam_width=16,
                     max_steps=12) -> (program, enumerated)

The write-execute-assess loop, run by [`ReplBeamIterator`](@ref): a state is a
set of executed values; each step applies every applicable rule to values in
the state (write), runs the result (execute), scores it with
`score(program, outputs) -> Float64` (assess), and keeps the `beam_width` best
successor states. `max_steps` bounds how many values a state may build on top
of the terminals.

Incomplete by design: values outside the beam are discarded, not delayed. A
failure means "the scorer never liked the right fragments", not "no program
exists". Compare with [`value_guided_search`](@ref), which uses the same score
function but stays complete.
"""
function repl_beam_search(grammar::AbstractGrammar, start::Symbol, problem::Problem,
                          score::F;
                          mod::Module=Main, max_programs::Integer=100_000,
                          beam_width::Integer=16, max_steps::Integer=12) where {F}
    ev = Evaluator(grammar, problem.spec; mod=mod)
    iter = ReplBeamIterator(grammar, start;
                            score=score, program_to_outputs=(p -> ev(p)),
                            beam_width=Int(beam_width), max_steps=Int(max_steps))
    enumerated = 0
    for program in iter
        enumerated += 1
        # the beam yields every executed candidate, of any type; a solution
        # must be of the start type
        if grammar.types[program.ind] == start && is_solution(ev, ev(program))
            return program, enumerated
        end
        enumerated >= max_programs && break
    end
    return nothing, enumerated
end
