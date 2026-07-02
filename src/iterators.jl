# =============================================================================
# Custom search iterators on top of HerbSearch's bottom-up search.
#
# The vanilla `CostBUSIterator` already covers every rule-weight method
# (DeepCoder, corpus priors, HySynth). The two iterators here change as little
# as possible for their method:
#
#   BustleBUSIterator   the cost-based BUS with one override, `bank_cost`:
#                       a learned penalty delays a value in the bank (BUSTLE).
#   ReplBeamIterator    its own `Base.iterate`: beam search over REPL states,
#                       keeping every candidate's execution results in the
#                       state (write-execute-assess).
# =============================================================================

"""
    BustleBUSIterator(grammar, start; penalty, rule_costs=Int[], max_cost=typemax(Int),
                      program_to_outputs=nothing)

HerbSearch's cost-based bottom-up iterator with one changed function: a newly
built program is banked at its structural cost plus `penalty(program)::Int`,
so the search combines promising values earlier. A penalty only *delays* a
value, it never discards it -- the search stays complete. Level-wise growth,
observational-equivalence pruning, and cost-ordered yielding are all inherited.

`rule_costs` defaults to cost 1 per rule (empty vector means uniform).
"""
@programiterator BustleBUSIterator(
    penalty::Function,
    rule_costs::Vector{Int}=Int[],
    max_cost::Int=typemax(Int),
    program_to_outputs::Union{Nothing,Function}=nothing,
) <: HerbSearch.AbstractBUSIterator

HerbSearch.node_cost(it::BustleBUSIterator, op::Int) =
    isempty(it.rule_costs) ? 1 : it.rule_costs[op]
HerbSearch.node_cost(it::BustleBUSIterator, prog::RuleNode) =
    HerbSearch.node_cost(it, prog.ind)
HerbSearch.cost_bound(it::BustleBUSIterator) = it.max_cost

HerbSearch.bank_cost(it::BustleBUSIterator, prog::RuleNode, ::Symbol, cost::Int) =
    cost + it.penalty(prog)

"""
    ReplBeamIterator(grammar, start; score, program_to_outputs, beam_width=16,
                     max_steps=12)

Beam search over REPL states, for write-execute-assess. A state is a set of
executed values; each step applies every applicable rule to values in the
state (write), runs the result through `program_to_outputs` (execute), scores
it with `score(program, outputs)` (assess), and keeps the `beam_width`
best-scoring successor states. `max_steps` bounds how many values a state may
build on top of the terminals.

Yields every program it executes, in step order, so a driver can count and
check candidates. Incomplete by design: values outside the beam are discarded,
not delayed (compare [`BustleBUSIterator`](@ref)).
"""
@programiterator ReplBeamIterator(
    score::Function,
    program_to_outputs::Function,
    beam_width::Int=16,
    max_steps::Int=12,
) <: ProgramIterator

Base.IteratorSize(::Type{<:ReplBeamIterator}) = Base.SizeUnknown()

# One executed value in a REPL state.
struct ReplValue
    program::RuleNode
    outputs::Vector{Any}
    type::Symbol
end

mutable struct ReplBeamState
    beam::Vector{Vector{ReplValue}}
    to_yield::Vector{RuleNode}   # programs executed in the current step
    yi::Int                      # next index into to_yield
    steps::Int
end

function Base.iterate(iter::ReplBeamIterator)
    grammar = get_grammar(iter)
    # Seed: one state holding every terminal value (deduplicated by outputs);
    # the terminals themselves are the first step's yields.
    seed = ReplValue[]
    to_yield = RuleNode[]
    for r in eachindex(grammar.isterminal)
        grammar.isterminal[r] || continue
        prog = RuleNode(r)
        outs = iter.program_to_outputs(prog)
        push!(to_yield, prog)
        any(v -> v.outputs == outs, seed) ||
            push!(seed, ReplValue(prog, outs, grammar.types[r]))
    end
    return Base.iterate(iter, ReplBeamState([seed], to_yield, 1, 0))
end

function Base.iterate(iter::ReplBeamIterator, state::ReplBeamState)
    while state.yi > length(state.to_yield)
        _beam_step!(iter, state) || return nothing
    end
    prog = state.to_yield[state.yi]
    state.yi += 1
    return prog, state
end

# Advance the beam by one step: expand every state, score every new value,
# keep the beam_width best successor states. Returns false when done.
function _beam_step!(iter::ReplBeamIterator, state::ReplBeamState)
    state.steps >= iter.max_steps && return false
    grammar = get_grammar(iter)

    successors = Tuple{Float64,Int,ReplValue}[]   # (score, parent state, value)
    to_yield = RuleNode[]
    for (si, repl) in enumerate(state.beam)
        by_type = Dict{Symbol,Vector{ReplValue}}()
        for v in repl
            push!(get!(by_type, v.type, ReplValue[]), v)
        end
        for op in findall(.!grammar.isterminal)
            slots = [get(by_type, t, ReplValue[]) for t in grammar.childtypes[op]]
            any(isempty, slots) && continue
            for children in Iterators.product(slots...)
                prog = RuleNode(op, [c.program for c in children])
                outs = iter.program_to_outputs(prog)
                push!(to_yield, prog)
                # a value the state already has adds nothing to the REPL
                any(v -> v.outputs == outs, repl) && continue
                s = iter.score(prog, outs)
                push!(successors, (s, si, ReplValue(prog, outs, grammar.types[op])))
            end
        end
    end
    isempty(successors) && isempty(to_yield) && return false

    sort!(successors; by=first, rev=true)
    state.beam = [vcat(state.beam[si], v)
                  for (_, si, v) in successors[1:min(iter.beam_width, end)]]
    state.to_yield = to_yield
    state.yi = 1
    state.steps += 1
    return !isempty(to_yield)
end
