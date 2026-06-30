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

# Mean loss over a batch of data (one fixed grammar embedding).
_batch_loss(model, batch::AbstractVector{TrainingDatum}, grammar_emb) =
    sum(d -> pairwise_loss(model, d.spec_emb, d.rule_mask, grammar_emb), batch) / length(batch)

"""
    train!(model, data, grammar_emb; epochs=200, lr=1e-3, batchsize=length(data),
           seed=nothing, verbose=false) -> Vector{Float32}

Train `model` in place with Adam to minimise the pairwise ranking loss against
`grammar_emb` (the `n × R` rule-embedding matrix of the training grammar).
Returns the per-epoch mean loss history.
"""
function train!(
    model::UniversEModel,
    data::AbstractVector{TrainingDatum},
    grammar_emb::AbstractMatrix;
    epochs::Integer=200,
    lr::Real=1e-3,
    batchsize::Integer=length(data),
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
            loss, grads = Flux.withgradient(m -> _batch_loss(m, batch, grammar_emb), model)
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
