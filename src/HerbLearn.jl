module HerbLearn

# --- standard library ---
using Random
using Serialization
using Statistics
using LinearAlgebra: norm, dot

# --- ML ---
using Flux

# --- Herb ecosystem ---
using HerbCore
using HerbGrammar
using HerbConstraints
using HerbInterpret
using HerbSearch
using HerbSpecification

# =============================================================================
# UniversE — a clean, modular reimplementation of the neurally-guided
# synthesis pipeline:
#
#   embed (universal LLM-style embeddings, offline + cached)
#     -> generate training data (sample programs, execute on inputs)
#     -> learn a heuristic in Flux (predict likely rules from the spec)
#     -> turn the heuristic into integer grammar-rule costs
#     -> cost-based bottom-up search guided by those costs
#
# Each stage lives in its own file and is independently testable.
# =============================================================================

include("embedding.jl")
include("encoding.jl")
include("data.jl")
include("model.jl")
include("train.jl")
include("costs.jl")
include("pipeline.jl")

export
    # embedding
    AbstractEmbedder,
    HashEmbedder,
    CachedEmbedder,
    PrecomputedEmbedder,
    embed,
    embed_dim,
    save_cache,

    # encoding
    input_string,
    embedding_strings,
    embed_example,
    embed_spec,
    embed_rule,
    embed_grammar,

    # data generation
    GeneratedExample,
    rule_mask,
    generate_examples,

    # model
    UniversEModel,
    cosine01,
    predict_scores,

    # training
    TrainingDatum,
    make_training_data,
    pairwise_loss,
    contrastive_loss,
    train!,

    # costs
    scores_to_costs,
    tiebreak_costs,
    assign_costs,

    # pipeline
    UniversE,
    fit_universe,
    guided_costs,
    search_with_costs,
    solve

end # module HerbLearn
