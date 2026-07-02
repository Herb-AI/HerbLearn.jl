# =============================================================================
# The value scorer: how useful does this value look for this task?
#
# A small MLP over property signatures that judges one executed value against
# the specification. It is a shared building block, deliberately: BUSTLE
# (Bustle/) plugs it into a complete cost-ordered search, write-execute-assess
# (WriteExecuteAssess/) plugs the *same* trained scorer into a beam over REPL
# states. Whatever differs between those two synthesizers is the search, not
# the network.
# =============================================================================

"""
    ValueScorer(encoder::PropertySignatureEncoder=PropertySignatureEncoder(); hidden=[32])

An MLP mapping the concatenation of the specification's signature and a
candidate value's signature (each `encoding_dim(encoder)` long) to a
usefulness score in `[0, 1]`. Train with [`train_scorer!`](@ref) on data from
[`scorer_training_data`](@ref); evaluate with [`value_score`](@ref).
"""
struct ValueScorer
    encoder::PropertySignatureEncoder
    net::Flux.Chain
end

function ValueScorer(encoder::PropertySignatureEncoder=PropertySignatureEncoder();
                     hidden::AbstractVector{<:Integer}=[32])
    layers = Any[]
    d = 2 * encoding_dim(encoder)
    for h in hidden
        push!(layers, Flux.Dense(d => h, Flux.relu))
        d = h
    end
    push!(layers, Flux.Dense(d => 1, Flux.sigmoid))
    return ValueScorer(encoder, Flux.Chain(layers...))
end

# The features of one candidate: the spec's signature next to the signature of
# (candidate outputs, target outputs), example by example. A failed execution
# (nothing) makes properties error, which the signature counts as "does not
# apply".
_value_features(s::ValueScorer, spec, outputs) = vcat(
    encode(s.encoder, spec),
    signature(s.encoder, ((outputs[i], spec[i].out) for i in eachindex(spec))),
)

"""
    value_score(s::ValueScorer, spec, outputs) -> Float64

Score in `[0, 1]` for a candidate whose execution on the spec's inputs gave
`outputs` (with `nothing` marking failed executions). `1` means "looks useful
for this task".
"""
value_score(s::ValueScorer, spec::AbstractVector{<:IOExample}, outputs) =
    Float64(only(s.net(_value_features(s, spec, outputs))))

"""
    subprograms(program::RuleNode) -> Vector{RuleNode}

All subtrees of `program`, including itself -- the values a bottom-up search
would build on the way to it.
"""
function subprograms(program::RuleNode)
    out = RuleNode[]
    _collect_subprograms!(out, program)
    return out
end

function _collect_subprograms!(out, node::RuleNode)
    push!(out, node)
    for c in node.children
        _collect_subprograms!(out, c)
    end
end

"""
    scorer_training_data(s::ValueScorer, grammar, generated; mod=Main, seed=nothing)
        -> Vector{@NamedTuple{features::Vector{Float32}, label::Float32}}

Labelled (spec, value) pairs from generated programs: for each generated
(spec, program), every subprogram's execution result is a *positive* (label 1
-- the search should keep it), and the execution results of a different
generated program's subprograms on the same inputs are *negatives* (label 0).
This is the BUSTLE supervision signal: does this value occur on the way to the
solution?
"""
function scorer_training_data(s::ValueScorer, grammar::AbstractGrammar,
                              generated::AbstractVector{GeneratedExample};
                              mod::Module=Main, seed::Union{Nothing,Integer}=nothing)
    seed === nothing || Random.seed!(seed)
    data = @NamedTuple{features::Vector{Float32}, label::Float32}[]
    length(generated) < 2 && return data

    for (i, g) in enumerate(generated)
        ev = Evaluator(grammar, g.spec; mod=mod)
        # positives: values on the path to this spec's own program
        for sub in subprograms(g.program)
            push!(data, (features=_value_features(s, g.spec, ev(sub)), label=1.0f0))
        end
        # negatives: values from another program, evaluated on this spec's inputs
        other = generated[i == length(generated) ? 1 : i + 1]
        for sub in subprograms(other.program)
            push!(data, (features=_value_features(s, g.spec, ev(sub)), label=0.0f0))
        end
    end
    return data
end

"""
    train_scorer!(s::ValueScorer, data; epochs=100, lr=1e-3, batchsize=32,
                  seed=nothing, verbose=false) -> Vector{Float32}

Train the scorer with binary cross-entropy on the output of
[`scorer_training_data`](@ref). Returns the per-epoch loss history.
"""
function train_scorer!(s::ValueScorer,
                       data::AbstractVector{@NamedTuple{features::Vector{Float32}, label::Float32}};
                       epochs::Integer=100, lr::Real=1e-3, batchsize::Integer=32,
                       seed::Union{Nothing,Integer}=nothing, verbose::Bool=false)
    @assert !isempty(data) "no training data"
    seed === nothing || Random.seed!(seed)

    X = reduce(hcat, (d.features for d in data))                # 2dim × N
    Y = reshape([d.label for d in data], 1, :)                  # 1 × N

    opt_state = Flux.setup(Flux.Adam(lr), s.net)
    history = Float32[]
    n = size(X, 2)
    for epoch in 1:epochs
        idx = shuffle(1:n)
        epoch_loss = 0.0f0
        nbatches = 0
        for lo in 1:batchsize:n
            batch = idx[lo:min(lo + batchsize - 1, n)]
            loss, grads = Flux.withgradient(net -> Flux.binarycrossentropy(net(X[:, batch]), Y[:, batch]), s.net)
            Flux.update!(opt_state, s.net, grads[1])
            epoch_loss += loss
            nbatches += 1
        end
        push!(history, epoch_loss / nbatches)
        verbose && (epoch % max(1, epochs ÷ 10) == 0) &&
            println("epoch $epoch / $epochs\tloss $(round(history[end]; digits=5))")
    end
    return history
end
