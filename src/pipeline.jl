# =============================================================================
# End-to-end UniversE pipeline.
#
# Ties the stages together:
#
#   fit_universe :  grammar + inputs  ->  trained heuristic  (embed, generate,
#                                                              train)
#   guided_costs :  heuristic + spec  ->  integer rule costs
#   search_with_costs : grammar + spec + costs -> solution program, via the
#                       integer-cost bottom-up search.
#
# `solve` runs the whole thing for a single problem.
# =============================================================================

"""
    UniversE(embedder, model)

A trained heuristic: the (offline) `embedder` plus the learned `UniversEModel`.
Use `guided_costs` to turn it into integer grammar-rule costs for a given spec.
"""
struct UniversE
    embedder::AbstractEmbedder
    model::UniversEModel
end

"""
    fit_universe(grammar, inputs; kwargs...) -> UniversE

Train a heuristic for `grammar` from a set of `inputs` (a vector of argument
`Dict`s, e.g. the inputs of a real problem). Steps: sample programs and execute
them on the inputs (`generate_examples`), embed the resulting specs, and train
the model to rank used rules above unused ones.

Keyword arguments (with defaults):
- `embedder=HashEmbedder(dim=64)` — the universal embedding function.
- `start=:Start`, `max_depth=5`, `min_depth=1` — program-sampling controls.
- `num_programs=200` — number of programs to generate.
- `hidden=[64, 64]`, `epochs=200`, `lr=1e-3` — model/training controls.
- `seed=nothing`, `verbose=false`.
"""
function fit_universe(
    grammar::AbstractGrammar,
    inputs::AbstractVector{<:AbstractDict};
    embedder::AbstractEmbedder=HashEmbedder(dim=64),
    start::Symbol=:Start,
    min_depth::Integer=1,
    max_depth::Integer=5,
    num_programs::Integer=200,
    hidden::AbstractVector{<:Integer}=[64, 64],
    epochs::Integer=200,
    lr::Real=1e-3,
    lossfn=contrastive_loss,
    seed::Union{Nothing,Integer}=nothing,
    verbose::Bool=false,
)::UniversE
    generated = generate_examples(grammar, inputs, num_programs;
                                  start=start, min_depth=min_depth, max_depth=max_depth, seed=seed)
    isempty(generated) && error("no training programs could be generated; try a larger max_depth or different inputs")

    data = make_training_data(embedder, generated)
    grammar_emb = embed_grammar(embedder, grammar)
    model = UniversEModel(embed_dim(embedder); hidden=hidden)
    train!(model, data, grammar_emb; lossfn=lossfn, epochs=epochs, lr=lr, seed=seed, verbose=verbose)

    return UniversE(embedder, model)
end

"""
    guided_costs(u::UniversE, grammar, spec; min_cost=1, scale=10) -> Vector{Int}

Integer per-rule costs for `grammar` conditioned on `spec`, from the trained
heuristic `u` (see `assign_costs`).
"""
guided_costs(u::UniversE, grammar::AbstractGrammar, spec::AbstractVector{<:IOExample}; kwargs...) =
    assign_costs(u.model, u.embedder, grammar, spec; kwargs...)

"""
    search_with_costs(grammar, spec, costs; start=:Start, max_cost=typemax(Int),
                      max_programs=100_000) -> (program, enumerated)

Run the integer-cost bottom-up search with the given per-rule `costs` and return
the first program satisfying every example in `spec`, along with how many
programs were enumerated before it was found. `program` is `nothing` if no
solution was found within `max_cost` / `max_programs`.
"""
function search_with_costs(
    grammar::AbstractGrammar,
    spec::AbstractVector{<:IOExample},
    costs::AbstractVector{<:Integer};
    start::Symbol=:Start,
    max_cost::Integer=typemax(Int),
    max_programs::Integer=100_000,
    observational_equivalence::Bool=true,
    mod::Module=Main,
)
    symtable = grammar2symboltable(grammar, mod)
    run_on(input, expr) = try
        execute_on_input(symtable, expr, input)
    catch
        nothing
    end

    # outputs over all inputs, used both for observational-equivalence pruning
    # (passed to the iterator) and for the solution check below.
    prog_outputs(p) = (e = rulenode2expr(p, grammar); Any[run_on(ex.in, e) for ex in spec])

    # CostBUSIterator (HerbSearch's queue-free, horizon-free cost-based search):
    # step the cost level and enumerate every program of that cost. Passing
    # `prog_outputs` enables observational-equivalence pruning; `nothing` disables it.
    oe_fn = observational_equivalence ? prog_outputs : nothing
    iter = CostBUSIterator(grammar, start, Int(max_cost), collect(Int, costs), oe_fn)

    enumerated = 0
    for program in iter
        enumerated += 1
        outs = prog_outputs(program)
        if all(outs[i] == spec[i].out for i in eachindex(spec))
            return (program=program, enumerated=enumerated)
        end
        enumerated >= max_programs && break
    end
    return (program=nothing, enumerated=enumerated)
end

"""
    solve(grammar, spec; uniform=false, search_max_cost, search_max_programs,
          min_cost, scale, fit_kwargs...) -> (program, enumerated, costs, universe)

Convenience end-to-end driver for one problem: train a heuristic on the spec's
inputs, derive integer costs, and search. Pass `uniform=true` to ignore the
heuristic and search with all-ones costs (the unguided baseline). Returns the
solution program, the number of programs enumerated, the costs used, and the
trained `UniversE` (or `nothing` when `uniform=true`).
"""
function solve(
    grammar::AbstractGrammar,
    spec::AbstractVector{<:IOExample};
    uniform::Bool=false,
    start::Symbol=:Start,
    search_max_cost::Integer=typemax(Int),
    search_max_programs::Integer=100_000,
    observational_equivalence::Bool=true,
    min_cost::Integer=1,
    scale::Integer=10,
    fit_kwargs...,
)
    inputs = [ex.in for ex in spec]
    if uniform
        costs = ones(Int, length(grammar.rules))
        universe = nothing
    else
        universe = fit_universe(grammar, inputs; start=start, fit_kwargs...)
        costs = guided_costs(universe, grammar, spec; min_cost=min_cost, scale=scale)
    end
    res = search_with_costs(grammar, spec, costs;
                            start=start, max_cost=search_max_cost, max_programs=search_max_programs,
                            observational_equivalence=observational_equivalence)
    return (program=res.program, enumerated=res.enumerated, costs=costs, universe=universe)
end
