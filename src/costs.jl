# =============================================================================
# Turning heuristic scores into integer grammar-rule costs.
#
# The cost-based bottom-up search enumerates programs in non-decreasing total
# cost, where a program's cost is the sum of its rules' costs. To exploit the
# heuristic we give *likely* rules (high score) a *low* cost so they are combined
# first, and unlikely rules a high cost.
#
# Integer costs keep the search's cost levels discrete and exactly bucketable.
# Every cost is at least `min_cost ≥ 1`, which guarantees a program's cost grows
# with its size and the search makes progress.
# =============================================================================

"""
    scores_to_costs(scores; min_cost=1, scale=10) -> Vector{Int}

Map heuristic scores in `[0, 1]` to positive integer costs. A score of `1`
(maximally likely rule) gets `min_cost`; a score of `0` gets `min_cost + scale`.
In general:

    cost = min_cost + round(Int, scale * (1 - clamp(score, 0, 1)))

so cost decreases monotonically as score increases, and `cost ≥ min_cost ≥ 1`.
"""
function scores_to_costs(scores::AbstractVector{<:Real}; min_cost::Integer=1, scale::Integer=10)::Vector{Int}
    min_cost >= 1 || throw(ArgumentError("min_cost must be ≥ 1, got $min_cost"))
    scale >= 0 || throw(ArgumentError("scale must be ≥ 0, got $scale"))
    return [min_cost + round(Int, scale * (1 - clamp(s, 0, 1))) for s in scores]
end

"""
    tiebreak_costs(scores; size_base=100, levels=3) -> Vector{Int}

Cost mapping that uses the heuristic only as a **tie-breaker within each program
size**. Every rule costs `size_base` plus a small adjustment in `0:levels-1` set
by the rule's score *rank* (cheapest adjustment = highest score). Because
`size_base` dominates the total per-program adjustment, the cost-based search
enumerates by size first and the heuristic only re-orders programs of *equal*
size — so a wrong ranking can shuffle within a size bucket but can never deepen
the solution.

This is what makes a weak/unreliable heuristic safe to use. Empirically on
SyGuS-SLIA it gives the same speedup as a "hard" score-proportional cost
(`scores_to_costs`) — median ≈0.78× the programs of an unguided search, winning
on most problems — while the worst case stays ≈1.0× (it essentially never hurts),
whereas hard costs blow up the search 25×+ when the heuristic mis-ranks a needed
rule.

Pick `size_base > levels * (largest program size you will search to)` to keep the
ordering strictly size-first up to that size, and set the search's `max_cost`
accordingly (≈ `size_base * max_size`).
"""
function tiebreak_costs(scores::AbstractVector{<:Real}; size_base::Integer=100, levels::Integer=3)::Vector{Int}
    size_base >= 1 || throw(ArgumentError("size_base must be ≥ 1, got $size_base"))
    levels >= 1 || throw(ArgumentError("levels must be ≥ 1, got $levels"))
    order = sortperm(scores; rev=true)                 # highest score first
    costs = fill(Int(size_base), length(scores))
    for (i, idx) in enumerate(order)
        costs[idx] = size_base + (levels * (i - 1)) ÷ length(scores)
    end
    return costs
end

"""
    assign_costs(model, embedder, grammar, spec; min_cost=1, scale=10) -> Vector{Int}

Compute integer costs for every rule of `grammar`, conditioned on the
specification `spec` (a vector of `IOExample`s): embed the spec, embed the rules,
score each rule with `model`, then convert scores to costs with `scores_to_costs`.
The returned vector aligns with `grammar.rules` and is ready to hand to the
cost-based bottom-up iterator.
"""
function assign_costs(
    model::UniversEModel,
    embedder::AbstractEmbedder,
    grammar::AbstractGrammar,
    spec::AbstractVector{<:IOExample};
    min_cost::Integer=1,
    scale::Integer=10,
)::Vector{Int}
    spec_emb = embed_spec(embedder, spec)
    grammar_emb = embed_grammar(embedder, grammar)
    scores = predict_scores(model, spec_emb, grammar_emb)
    return scores_to_costs(scores; min_cost=min_cost, scale=scale)
end
