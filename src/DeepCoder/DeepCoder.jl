module DeepCoder

# DeepCoder [Balog et al., ICLR 2017, ref.bib]: predict which rules the
# solution uses, once, from the specification; search the grammar reweighted
# by those predictions. The original runs DFS / sort-and-add; the cost-based
# bottom-up driver plays the same role here.

import Flux
using Random: shuffle
import Random

using HerbCore: AbstractGrammar
using HerbSpecification: Problem, IOExample
using ..HerbLearn: SpecEncoder, ValueEncoder, HashEmbedder, encode, encoding_dim,
    GeneratedExample, guided_bottom_up_search

export DeepCoderModel, train_deepcoder!, predict_rule_weights, deepcoder

"""
    DeepCoderModel(encoder::SpecEncoder, grammar; hidden=[64])

An MLP from the spec encoding to a presence probability per rule of `grammar`.
The output layer has one unit per rule, so the model is tied to this grammar.
"""
struct DeepCoderModel{E<:SpecEncoder}
    encoder::E
    net::Flux.Chain
end

function DeepCoderModel(encoder::SpecEncoder, grammar::AbstractGrammar;
                        hidden::AbstractVector{<:Integer}=[64])
    layers = Any[]
    d = encoding_dim(encoder)
    for h in hidden
        push!(layers, Flux.Dense(d => h, Flux.relu))
        d = h
    end
    push!(layers, Flux.Dense(d => length(grammar.rules), Flux.sigmoid))
    return DeepCoderModel(encoder, Flux.Chain(layers...))
end

"""
    predict_rule_weights(model::DeepCoderModel, spec) -> Vector{Float64}

The model's presence probability for every rule, given the examples.
"""
predict_rule_weights(model::DeepCoderModel, spec::AbstractVector{<:IOExample}) =
    Float64.(model.net(encode(model.encoder, spec)))

"""
    train_deepcoder!(model, generated::AbstractVector{GeneratedExample};
                     epochs=100, lr=1e-3, batchsize=32, seed=nothing,
                     verbose=false) -> Vector{Float32}

Train the presence predictor with binary cross-entropy: for each generated
(spec, program) pair, the target is the program's rule mask. Returns the
per-epoch loss history.
"""
function train_deepcoder!(model::DeepCoderModel, generated::AbstractVector{GeneratedExample};
                          epochs::Integer=100, lr::Real=1e-3, batchsize::Integer=32,
                          seed::Union{Nothing,Integer}=nothing, verbose::Bool=false)
    @assert !isempty(generated) "no training data"
    seed === nothing || Random.seed!(seed)

    X = reduce(hcat, (encode(model.encoder, g.spec) for g in generated))   # d × N
    Y = reduce(hcat, (Float32.(g.rule_mask) for g in generated))           # R × N

    opt_state = Flux.setup(Flux.Adam(lr), model.net)
    history = Float32[]
    n = size(X, 2)
    for epoch in 1:epochs
        idx = shuffle(1:n)
        epoch_loss = 0.0f0
        nbatches = 0
        for lo in 1:batchsize:n
            b = idx[lo:min(lo + batchsize - 1, n)]
            loss, grads = Flux.withgradient(net -> Flux.binarycrossentropy(net(X[:, b]), Y[:, b]), model.net)
            Flux.update!(opt_state, model.net, grads[1])
            epoch_loss += loss
            nbatches += 1
        end
        push!(history, epoch_loss / nbatches)
        verbose && (epoch % max(1, epochs ÷ 10) == 0) &&
            println("epoch $epoch / $epochs\tloss $(round(history[end]; digits=5))")
    end
    return history
end

"""
    deepcoder(grammar, start::Symbol, problem::Problem; model::DeepCoderModel,
              kwargs...) -> (program, enumerated)

Synthesize a program for `problem`, DeepCoder-style: predict a weight per rule
from the examples, once, and run the complete cost-based bottom-up search
under those weights. Keyword arguments are forwarded to
[`guided_bottom_up_search`](@ref HerbLearn.guided_bottom_up_search).
"""
deepcoder(grammar::AbstractGrammar, start::Symbol, problem::Problem;
          model::DeepCoderModel, kwargs...) =
    guided_bottom_up_search(grammar, start, problem,
                            predict_rule_weights(model, problem.spec); kwargs...)

end # module DeepCoder
