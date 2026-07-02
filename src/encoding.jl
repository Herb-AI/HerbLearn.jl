# =============================================================================
# Encoding specifications and grammar rules with an embedder.
#
#   - an input/output pair  (i, o)  ->  f_e(i) ⊕ f_e(o)   ∈ ℝ^{2n}
#   - a set of examples E (a spec)  ->  mean over pairs    ∈ ℝ^{2n}   (deep set,
#                                       so the encoding is order-invariant)
#   - a grammar rule r              ->  f_e(str(r))        ∈ ℝ^{n}
#
# The rule embeddings (embed_rule / embed_grammar) are the starting point for
# heuristics that read the grammar as an input instead of hard-wiring it into
# their output layer -- an open direction, kept here for experiments.
# =============================================================================

"""
    input_string(input)::String

Canonical string representation of an example's input used for embedding.
For a `Dict` of arguments the values are joined in a key-sorted, deterministic
order so the embedding does not depend on Julia's `Dict` iteration order.
"""
input_string(input::AbstractDict) =
    join((string(input[k]) for k in sort(collect(keys(input)); by=string)), " ")
input_string(input) = string(input)

"""
    embed_example(e, ex::IOExample)::Vector{Float32}

Embed a single input/output example as the concatenation `f_e(i) ⊕ f_e(o)`,
a vector of length `2 * embed_dim(e)`.
"""
function embed_example(e::AbstractEmbedder, ex::IOExample)::Vector{Float32}
    return vcat(embed(e, input_string(ex.in)), embed(e, string(ex.out)))
end

"""
    embed_spec(e, examples::AbstractVector{<:IOExample})::Vector{Float32}

Embed a whole specification (a set of examples) as the mean of the per-example
embeddings -- a deep set, hence invariant to the order of the examples. Returns
a vector of length `2 * embed_dim(e)`. An empty spec yields the zero vector.
"""
function embed_spec(e::AbstractEmbedder, examples::AbstractVector{<:IOExample})::Vector{Float32}
    isempty(examples) && return zeros(Float32, 2 * embed_dim(e))
    acc = embed_example(e, examples[1])
    @inbounds for k in 2:length(examples)
        acc .+= embed_example(e, examples[k])
    end
    acc ./= length(examples)
    return acc
end

"""
    embed_rule(e, grammar, rule_idx::Int)::Vector{Float32}

Embed a single grammar derivation rule via its string representation, a vector
of length `embed_dim(e)`.
"""
embed_rule(e::AbstractEmbedder, grammar::AbstractGrammar, rule_idx::Integer)::Vector{Float32} =
    embed(e, string(grammar.rules[rule_idx]))

"""
    embedding_strings(grammar, specs)::Vector{String}

Every distinct string the pipeline will need an embedding for: each rule's string
representation, plus each example's input string and output string across all the
given `specs` (a collection of `Vector{IOExample}`). Use this to know exactly
which strings to embed offline with an LLM backend before training, so the
resulting `PrecomputedEmbedder` never misses.
"""
function embedding_strings(grammar::AbstractGrammar, specs)::Vector{String}
    s = Set{String}()
    for i in eachindex(grammar.rules)
        push!(s, string(grammar.rules[i]))
    end
    for spec in specs, ex in spec
        push!(s, input_string(ex.in))
        push!(s, string(ex.out))
    end
    return collect(s)
end

"""
    embed_grammar(e, grammar)::Matrix{Float32}

Embed every rule of `grammar`, returning an `embed_dim(e) × length(rules)`
matrix whose column `i` is `embed_rule(e, grammar, i)`.
"""
function embed_grammar(e::AbstractEmbedder, grammar::AbstractGrammar)::Matrix{Float32}
    n = length(grammar.rules)
    M = Matrix{Float32}(undef, embed_dim(e), n)
    @inbounds for i in 1:n
        M[:, i] = embed_rule(e, grammar, i)
    end
    return M
end
