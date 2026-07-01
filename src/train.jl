# =============================================================================
# Training the heuristic.
#
# We learn f_MLP so that, for a given spec, the rules that actually occur in the
# generating program score *higher* than the rules that do not. This is the
# "learn a likely direction" objective of UniversE, realised as a pairwise
# ranking loss over (present-rule, absent-rule) pairs.
# =============================================================================

"""
    TrainingDatum(spec_emb, rule_mask)

A ready-to-train datapoint: the spec embedding `spec_emb` (length `2n`) and the
`rule_mask` (`BitVector` over the training grammar's rules) it should predict.
"""
struct TrainingDatum
    spec_emb::Vector{Float32}
    rule_mask::BitVector
end

"""
    make_training_data(embedder, generated) -> Vector{TrainingDatum}

Turn `GeneratedExample`s into `TrainingDatum`s by embedding each spec with
`embedder` (the rule mask is carried over unchanged).
"""
make_training_data(embedder::AbstractEmbedder, generated::AbstractVector{GeneratedExample}) =
    [TrainingDatum(embed_spec(embedder, g.spec), g.rule_mask) for g in generated]

"""
    pairwise_loss(model, spec_emb, mask, grammar_emb)

Mean softplus pairwise ranking loss for one spec: over every (present, absent)
rule pair it penalises `softplus(score_absent - score_present)`, which is small
exactly when present rules outscore absent rules. Returns `0` if the mask is all
present or all absent (no pairs to rank).
"""
function pairwise_loss(model::UniversEModel, spec_emb::AbstractVector, mask::BitVector, grammar_emb::AbstractMatrix)
    scores = predict_scores(model, spec_emb, grammar_emb)   # length R, in [0,1]
    pos = scores[mask]
    neg = scores[.!mask]
    (isempty(pos) || isempty(neg)) && return zero(eltype(scores))
    # diff[i,j] = score_absent_j - score_present_i  ; want this negative
    diff = reshape(neg, 1, :) .- reshape(pos, :, 1)
    return mean(Flux.softplus.(diff))
end

# numerically-stable log-sum-exp (Zygote-differentiable), vector and column-wise.
_logsumexp(x::AbstractVector) = (m = maximum(x); m + log(sum(exp, x .- m)))
_logsumexp(x::AbstractMatrix; dims) = (m = maximum(x; dims); m .+ log.(sum(exp.(x .- m); dims)))

"""
    contrastive_loss(model, spec_emb, mask, grammar_emb; temperature=0.1)

InfoNCE-style contrastive loss. The model maps the spec to a direction in rule
space; this pulls that direction toward the embeddings of the rules that *do*
occur in the program (positives) and pushes it away from all the others
(negatives), via a softmax over every rule:

    L = -mean_{r ∈ positives} log softmax(cos(dir, rule_r) / temperature)_r

Because the softmax is normalised over the whole grammar, each spec's loss is a
*relative* ranking of its rules — which suppresses the global rule-frequency
prior that a plain presence/absence objective tends to learn, and is what makes
the heuristic actually discriminate useful rules from rubbish ones. Returns `0`
if the mask is all-present or all-absent (no contrast to form).
"""
function contrastive_loss(model::UniversEModel, spec_emb::AbstractVector, mask::BitVector,
                          grammar_emb::AbstractMatrix; temperature::Real=0.1f0)
    npos = sum(mask)
    (npos == 0 || npos == length(mask)) && return zero(eltype(spec_emb))

    dir = model(spec_emb)                                        # predicted direction, ℝⁿ
    dir = dir ./ (sqrt(sum(abs2, dir)) + 1.0f-12)
    G = grammar_emb ./ sqrt.(sum(abs2, grammar_emb; dims=1) .+ 1.0f-12)  # n × R, unit cols
    sims = (transpose(G) * dir) ./ eltype(spec_emb)(temperature)         # R cosine sims / τ

    logprob = sims .- _logsumexp(sims)                           # log-softmax over rules
    return -sum(logprob[mask]) / npos
end

"""
    _contrastive_batch(model, S, M, grammar_emb; temperature=0.1)

Vectorised [`contrastive_loss`](@ref) over a whole batch in one forward pass:
`S` is the `2n × B` matrix of spec embeddings and `M` the `R × B` rule-mask
matrix. Doing the batch as a single matmul (instead of summing `B` separate
per-datum graphs) keeps the Zygote tape shallow — the per-datum sum overflows the
stack for wide models / large batches — and is much faster.
"""
function _contrastive_batch(model::UniversEModel, S::AbstractMatrix, M::AbstractMatrix,
                            grammar_emb::AbstractMatrix; temperature::Real=0.1f0)
    P  = model(S)                                                       # n × B
    Pn = P ./ (sqrt.(sum(abs2, P; dims=1)) .+ 1.0f-12)
    Gn = grammar_emb ./ (sqrt.(sum(abs2, grammar_emb; dims=1)) .+ 1.0f-12)  # n × R
    sims = (transpose(Gn) * Pn) ./ eltype(S)(temperature)              # R × B
    logprob = sims .- _logsumexp(sims; dims=1)                         # R × B log-softmax
    Mf   = eltype(S).(M)
    npos = sum(Mf; dims=1)                                             # 1 × B
    percol = vec(-sum(logprob .* Mf; dims=1)) ./ vec(max.(npos, 1.0f0))  # B  (mean over positives)
    valid  = vec(npos) .> 0                                            # skip degenerate specs
    return sum(percol .* valid) / max(count(valid), 1)
end

# Mean loss over a batch. The contrastive loss uses the vectorised path; any
# other per-spec lossfn falls back to summing per-datum.
function _batch_loss(model, batch::AbstractVector{TrainingDatum}, grammar_emb, lossfn)
    if lossfn === contrastive_loss
        S = reduce(hcat, (d.spec_emb for d in batch))
        M = reduce(hcat, (d.rule_mask for d in batch))
        return _contrastive_batch(model, S, M, grammar_emb)
    end
    return sum(d -> lossfn(model, d.spec_emb, d.rule_mask, grammar_emb), batch) / length(batch)
end

"""
    train!(model, data, grammar_emb; lossfn=contrastive_loss, epochs=200, lr=1e-3,
           batchsize=length(data), seed=nothing, verbose=false) -> Vector{Float32}

Train `model` in place with Adam against `grammar_emb` (the `n × R` rule-embedding
matrix of the training grammar). `lossfn(model, spec_emb, mask, grammar_emb)` is
the per-spec loss; it defaults to [`contrastive_loss`](@ref) (which discriminates
useful rules from rubbish ones better than [`pairwise_loss`](@ref)). Returns the
per-epoch mean loss history.
"""
function train!(
    model::UniversEModel,
    data::AbstractVector{TrainingDatum},
    grammar_emb::AbstractMatrix;
    lossfn=contrastive_loss,
    epochs::Integer=200,
    lr::Real=1e-3,
    batchsize::Integer=min(32, length(data)),  # mini-batch: a full-batch Zygote
                                                # graph over a large dataset blows the stack
    seed::Union{Nothing,Integer}=nothing,
    verbose::Bool=false,
)
    @assert !isempty(data) "no training data"
    seed === nothing || Random.seed!(seed)

    opt_state = Flux.setup(Flux.Adam(lr), model)
    history = Float32[]

    for epoch in 1:epochs
        idx = shuffle(1:length(data))
        epoch_loss = 0.0f0
        nbatches = 0
        for start in 1:batchsize:length(data)
            batch = data[idx[start:min(start + batchsize - 1, end)]]
            loss, grads = Flux.withgradient(m -> _batch_loss(m, batch, grammar_emb, lossfn), model)
            Flux.update!(opt_state, model, grads[1])
            epoch_loss += loss
            nbatches += 1
        end
        epoch_loss /= nbatches
        push!(history, epoch_loss)
        verbose && (epoch % max(1, epochs ÷ 10) == 0) &&
            println("epoch $epoch / $epochs\tloss $(round(epoch_loss; digits=5))")
    end

    return history
end
