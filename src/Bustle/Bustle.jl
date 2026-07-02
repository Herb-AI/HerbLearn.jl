module Bustle

# BUSTLE [Odena et al., ICLR 2021, ref.bib]: during bottom-up search, score
# every newly built value against the examples and delay low-scoring values in
# the bank. The scorer is the shared HerbLearn.ValueScorer; this module only
# connects it to the value-guided driver.

using HerbCore: AbstractGrammar
using HerbSpecification: Problem
using ..HerbLearn: ValueScorer, value_score, value_guided_search

export bustle

"""
    bustle(grammar, start::Symbol, problem::Problem; scorer::ValueScorer,
           kwargs...) -> (program, enumerated)

Synthesize a program for `problem`, BUSTLE-style: run the complete cost-based
bottom-up search, re-prioritizing at every step -- each new value is banked at
its structural cost plus a penalty from `scorer`, so promising fragments are
combined first and a wrong score costs time, never the solution. Train the
scorer with `train_scorer!` on `scorer_training_data`. Keyword arguments
(`penalty_scale`, budgets, `mod`) are forwarded to
[`value_guided_search`](@ref HerbLearn.value_guided_search).
"""
bustle(grammar::AbstractGrammar, start::Symbol, problem::Problem;
       scorer::ValueScorer, kwargs...) =
    value_guided_search(grammar, start, problem,
                        (prog, outs) -> value_score(scorer, problem.spec, outs);
                        kwargs...)

end # module Bustle
