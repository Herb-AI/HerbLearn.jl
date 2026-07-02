# =============================================================================
# Fitting rule weights from a corpus of programs.
#
# Count which rules the programs use, smooth, normalize within each return
# type. This is the estimator behind corpus priors (Menon et al., ICML 2013;
# Euphony, PLDI 2018) and behind HySynth's LLM distillation -- the corpus is
# just whatever programs you have: solutions, partial solutions, or parsed
# LLM proposals.
# =============================================================================

"""
    fit_pcfg(programs, grammar; smoothing=1.0) -> Vector{Float64}

Rule probabilities fitted by counting rule occurrences over `programs`
(`RuleNode`s), with Laplace `smoothing`, normalized within each return type
(so the rules of each non-terminal form a distribution). Pass the result as
the weight vector of [`guided_bottom_up_search`](@ref).
"""
function fit_pcfg(programs, grammar::AbstractGrammar; smoothing::Real=1.0)
    counts = zeros(Float64, length(grammar.rules))
    for p in programs
        _count_rules!(counts, p)
    end
    return _normalize_by_type(counts .+ smoothing, grammar)
end

function _count_rules!(counts::Vector{Float64}, node::RuleNode)
    counts[node.ind] += 1
    for c in node.children
        _count_rules!(counts, c)
    end
end

function _normalize_by_type(weights::Vector{Float64}, grammar::AbstractGrammar)
    probs = similar(weights)
    for (_, idxs) in grammar.bytype
        total = sum(weights[idxs])
        probs[idxs] .= total > 0 ? weights[idxs] ./ total : 1.0 / length(idxs)
    end
    return probs
end
