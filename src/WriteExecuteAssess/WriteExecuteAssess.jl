module WriteExecuteAssess

# Write, execute, assess [Ellis et al., NeurIPS 2019, ref.bib]: keep the search
# in states where everything is executed. A state is a set of values; an
# action builds one more value from them (write), runs it (execute), and a
# learned scorer judges it (assess); beam search keeps the best states. The
# scorer is the shared HerbLearn.ValueScorer -- the same object BUSTLE uses --
# so this module only connects it to the beam driver.
#
# Simplifications w.r.t. the original: one scorer plays both the policy and
# the value network, and the search is a plain beam instead of beam + SMC.

using HerbCore: AbstractGrammar
using HerbSpecification: Problem
using ..HerbLearn: ValueScorer, value_score, repl_beam_search

export write_execute_assess

"""
    write_execute_assess(grammar, start::Symbol, problem::Problem;
                         scorer::ValueScorer, beam_width=16, max_steps=12,
                         kwargs...) -> (program, enumerated)

Synthesize a program for `problem` with the write-execute-assess loop: beam
search over REPL states (sets of executed values), where each candidate action
is executed and assessed by `scorer`. Incomplete by design -- values outside
the beam are discarded, not delayed -- in exchange the search only grows
states the scorer likes. Keyword arguments are forwarded to
[`repl_beam_search`](@ref HerbLearn.repl_beam_search).
"""
write_execute_assess(grammar::AbstractGrammar, start::Symbol, problem::Problem;
                     scorer::ValueScorer, kwargs...) =
    repl_beam_search(grammar, start, problem,
                     (prog, outs) -> value_score(scorer, problem.spec, outs);
                     kwargs...)

end # module WriteExecuteAssess
