module HySynth

# HYSYNTH [Barke et al., NeurIPS 2024, ref.bib]: ask an LLM for candidate
# programs, but do not trust them -- count which grammar rules the parseable
# proposals use, fit a smoothed PCFG (HerbLearn.fit_pcfg), and run the
# ordinary complete bottom-up search under those weights. The model's
# unreliability then costs only time, never correctness.
#
# This module is deliberately boilerplate to build variants on: change the
# prompt, change the parsing, change the fitting. Li, Parsert & Polgreen
# (CAV 2024, ref.bib) show the feedback-loop variant.

using HerbCore: AbstractGrammar, RuleNode
using HerbGrammar: expr2rulenode
using HerbSpecification: Problem, IOExample
using ..HerbLearn: fit_pcfg, guided_bottom_up_search, input_string

export LLM, complete, MockLLM, default_prompt, parse_proposals, hysynth

"""
    LLM

Supertype of language-model backends. A concrete backend implements one
function: `complete(llm, prompt::AbstractString)::Vector{String}`, returning
one or more completions.
"""
abstract type LLM end

"""
    complete(llm::LLM, prompt::AbstractString) -> Vector{String}

Ask the model for completions of `prompt`.
"""
function complete end

"""
    MockLLM(responses::Vector{String})

An offline stand-in that returns the canned `responses` for every prompt. Use
it to test an LLM-guided pipeline deterministically, or to replay cached
completions of a real model.
"""
struct MockLLM <: LLM
    responses::Vector{String}
end

complete(llm::MockLLM, ::AbstractString) = llm.responses

"""
    default_prompt(grammar, spec) -> String

A minimal prompt: the grammar's rules, the examples, and a request for
programs, one per line. Deliberately plain -- a starting point to edit, not a
tuned artifact.
"""
function default_prompt(grammar::AbstractGrammar, spec::AbstractVector{<:IOExample})
    io = IOBuffer()
    println(io, "You write programs in a small language with these rules:")
    for i in eachindex(grammar.rules)
        println(io, "  ", grammar.types[i], " = ", grammar.rules[i])
    end
    println(io, "\nWrite a program that maps each input to its output:")
    for ex in spec
        println(io, "  ", input_string(ex.in), "  ->  ", ex.out)
    end
    println(io, "\nAnswer with candidate programs only, one per line.")
    return String(take!(io))
end

"""
    parse_proposals(proposals, grammar) -> Vector{RuleNode}

Parse each proposed line into a program of `grammar`, silently dropping
everything that does not parse (chatter, ill-formed or out-of-grammar
expressions). What survives is enough to fit rule weights from.
"""
function parse_proposals(proposals::AbstractVector{<:AbstractString}, grammar::AbstractGrammar)
    programs = RuleNode[]
    for line in proposals
        s = strip(line)
        isempty(s) && continue
        program = try
            expr = Meta.parse(s)
            expr2rulenode(expr, grammar)
        catch
            nothing
        end
        program isa RuleNode && push!(programs, program)
    end
    return programs
end

"""
    hysynth(grammar, start::Symbol, problem::Problem; llm::LLM,
            smoothing=1.0, prompt=default_prompt, kwargs...)
        -> (program, enumerated)

Synthesize a program for `problem`, HYSYNTH-style: prompt `llm` once, parse
whatever parses, fit a smoothed PCFG to the parsed rules, and run the complete
cost-based bottom-up search under those weights. `prompt` is a function
`(grammar, spec) -> String`. Keyword arguments are forwarded to
[`guided_bottom_up_search`](@ref HerbLearn.guided_bottom_up_search).
"""
function hysynth(grammar::AbstractGrammar, start::Symbol, problem::Problem;
                 llm::LLM, smoothing::Real=1.0, prompt=default_prompt, kwargs...)
    proposals = complete(llm, prompt(grammar, problem.spec))
    programs = parse_proposals(proposals, grammar)
    weights = fit_pcfg(programs, grammar; smoothing=smoothing)
    return guided_bottom_up_search(grammar, start, problem, weights; kwargs...)
end

end # module HySynth
