# =============================================================================
# The UniversE model.
#
# The only learnable component is an MLP  f_MLP : ℝ^{2n} -> ℝ^{n}  that maps a
# spec embedding (an input/output pair embedding, dimension 2n) into the rule
# embedding space (dimension n). The heuristic score of a rule r given a spec E
# is the (rescaled) cosine similarity between f_MLP(f_e(E)) and the rule's own
# embedding f_e(r):
#
#     h(E, r) = ( cos( f_MLP(f_e(E)), f_e(r) ) + 1 ) / 2   ∈ [0, 1]
#
# This formulation is independent of the number of rules, so the same trained
# model scores rules of *any* grammar — including rules it never saw in training.
# =============================================================================

"""
    UniversEModel(n; hidden=[64, 64])

Build the learnable heuristic for a rule-embedding dimension `n`. The wrapped
MLP maps the spec space `ℝ^{2n}` to the rule space `ℝ^{n}` through the given
`hidden` layer sizes (ReLU activations). Callable: `model(spec)` returns the
projection `f_MLP(spec)`.
"""
struct UniversEModel
    mlp::Flux.Chain
    in_dim::Int      # 2n
    out_dim::Int     # n
end

Flux.@layer UniversEModel trainable=(mlp,)

function UniversEModel(n::Integer; hidden::AbstractVector{<:Integer}=[64, 64])
    in_dim = 2n
    out_dim = n
    layers = Any[]
    d = in_dim
    for h in hidden
        push!(layers, Flux.Dense(d => h, Flux.relu))
        d = h
    end
    push!(layers, Flux.Dense(d => out_dim))   # linear projection into rule space
    return UniversEModel(Flux.Chain(layers...), in_dim, out_dim)
end

# Forward pass: project a spec embedding (vector) or a batch (2n × B matrix).
(m::UniversEModel)(spec) = m.mlp(spec)

"""
    cosine01(u, v) -> Float

Cosine similarity rescaled from `[-1, 1]` to `[0, 1]`, where `1` is maximally
similar. Matches the `sim` used by UniversE's heuristic.
"""
function cosine01(u::AbstractVector, v::AbstractVector)
    d = norm(u) * norm(v)
    d == 0 && return 0.5f0
    return (dot(u, v) / d + 1) / 2
end

# L2-normalise each column of a matrix (differentiable; safe at zero).
_l2norm_cols(X::AbstractMatrix) = X ./ sqrt.(sum(abs2, X; dims=1) .+ 1.0f-12)

"""
    predict_scores(model, spec_emb, grammar_emb) -> scores

Heuristic scores in `[0, 1]` of every grammar rule for one or more specs.

- `grammar_emb` is the `n × R` matrix of rule embeddings (see `embed_grammar`).
- If `spec_emb` is a length-`2n` vector, returns a length-`R` vector of scores.
- If `spec_emb` is a `2n × B` matrix (a batch), returns an `R × B` matrix.

This is the differentiable core used both for training and for assigning costs.
"""
function predict_scores(model::UniversEModel, spec_emb::AbstractVector, grammar_emb::AbstractMatrix)
    return vec(predict_scores(model, reshape(spec_emb, :, 1), grammar_emb))
end

function predict_scores(model::UniversEModel, spec_emb::AbstractMatrix, grammar_emb::AbstractMatrix)
    P = _l2norm_cols(model(spec_emb))      # n × B  (projected, normalised)
    G = _l2norm_cols(grammar_emb)          # n × R  (rule embeddings, normalised)
    sims = transpose(G) * P                # R × B  cosine similarities in [-1, 1]
    return (sims .+ 1) ./ 2                 # rescale to [0, 1]
end
