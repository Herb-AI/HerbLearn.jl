# =============================================================================
# Training-data generation.
#
# Following UniversE/DeepCoder/DreamCoder we learn from *generated* data: take
# the inputs of a problem, sample random programs from its grammar, execute them
# on the inputs to obtain outputs, and record which rules the program used. Each
# such (spec, program, rules-used) triple is one training example. The learned
# heuristic then predicts the used rules from the spec embedding.
# =============================================================================

"""
    GeneratedExample

One generated training datapoint:

- `spec`      : the input/output examples (inputs taken from the source problem,
                outputs obtained by executing `program` on them).
- `program`   : the sampled `RuleNode` that produced the outputs.
- `rule_mask` : `BitVector` of length `length(grammar.rules)`; `true` at every
                rule index used somewhere in `program` (the learning target).
"""
struct GeneratedExample
    spec::Vector{IOExample}
    program::RuleNode
    rule_mask::BitVector
end

"""
    rule_mask(program, n_rules)::BitVector

A `BitVector` of length `n_rules` marking every rule index appearing in
`program` (the multiset of rules collapsed to presence/absence).
"""
function rule_mask(program::AbstractRuleNode, n_rules::Integer)::BitVector
    mask = falses(n_rules)
    _mark_rules!(mask, program)
    return mask
end

_mark_rules!(mask::BitVector, rn::RuleNode) = begin
    mask[rn.ind] = true
    for c in rn.children
        _mark_rules!(mask, c)
    end
end

"""
    generate_examples(grammar, inputs, num_programs; kwargs...) -> Vector{GeneratedExample}

Sample up to `num_programs` distinct programs from `grammar` and turn each into a
`GeneratedExample` by executing it on every input in `inputs` (a vector of
argument `Dict`s, typically reused from a real problem's spec).

Keyword arguments:
- `start::Symbol=:Start` — start symbol to sample from.
- `min_depth::Int=1`, `max_depth::Int=5` — depth bounds on sampled programs.
- `exclude::Set{RuleNode}=Set{RuleNode}()` — programs never to emit (e.g. the
  known solution, so generated data does not contain it).
- `max_attempts::Int=100*num_programs` — give up after this many samples.
- `seed::Union{Nothing,Int}=nothing` — if given, seeds the global RNG first so
  the sample is reproducible (HerbGrammar's sampler draws from the global RNG).
- `mod::Module=Main` — module whose functions the grammar calls (pass a benchmark
  module like `PBE_SLIA_Track_2019` so its `*_cvc` string functions resolve).

Programs that error on any input (or produce `nothing`) are skipped.
"""
function generate_examples(
    grammar::AbstractGrammar,
    inputs::AbstractVector{<:AbstractDict},
    num_programs::Integer;
    start::Symbol=:Start,
    min_depth::Integer=1,
    max_depth::Integer=5,
    exclude::Set{RuleNode}=Set{RuleNode}(),
    max_attempts::Integer=100 * num_programs,
    seed::Union{Nothing,Integer}=nothing,
    mod::Module=Main,
)::Vector{GeneratedExample}

    seed === nothing || Random.seed!(seed)

    n_rules = length(grammar.rules)
    # `mod` is the module whose functions the grammar calls (e.g. a benchmark
    # module like PBE_SLIA_Track_2019, which defines `concat_cvc`, `substr_cvc`, …).
    symtable = grammar2symboltable(grammar, mod)

    out = GeneratedExample[]
    seen = Set{RuleNode}()
    attempts = 0

    while length(out) < num_programs && attempts < max_attempts
        attempts += 1

        program = rand(RuleNode, grammar, start, max_depth)
        depth(program) < min_depth && continue
        program in seen && continue
        program in exclude && continue

        expr = rulenode2expr(program, grammar)

        examples = IOExample[]
        ok = true
        for input in inputs
            output = try
                execute_on_input(symtable, expr, input)
            catch
                nothing
            end
            if output === nothing
                ok = false
                break
            end
            push!(examples, IOExample(input, output))
        end
        ok || continue

        push!(seen, program)
        push!(out, GeneratedExample(examples, program, rule_mask(program, n_rules)))
    end

    return out
end
