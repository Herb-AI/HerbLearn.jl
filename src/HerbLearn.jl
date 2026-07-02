"""
    HerbLearn

Learning from and about programs: building blocks for neurally-guided program
synthesis in the Herb ecosystem.

Every guidance method in the literature trains a heuristic that steers a
search, and the methods share the same subcomponents. HerbLearn provides those
subcomponents, and each synthesizer is a short file that combines them:

- **What the heuristic sees** -- specification encoders (`ValueEncoder`,
  `PropertySignatureEncoder` in `see.jl`) over pluggable string embedders
  (`embedding.jl`), and the shared value scorer (`scorer.jl`).
- **What it says, and when the search listens** -- the search drivers in
  `search.jl`, plain functions taking either a weight vector
  ([`guided_bottom_up_search`](@ref)) or a score function
  ([`value_guided_search`](@ref), [`repl_beam_search`](@ref)).
- **Training data** -- program sampling in `data.jl`, PCFG fitting in `pcfg.jl`.

The synthesizers live in one folder each (`method.jl` + `README.md` +
`ref.bib`, following Garden.jl): `DeepCoder/`, `Bustle/`,
`WriteExecuteAssess/`, `HySynth/`. Call them as
`HerbLearn.DeepCoder.deepcoder(grammar, start, problem; model)` and so on.
"""
module HerbLearn

# --- standard library ---
using Random
using Serialization
using Statistics
using LinearAlgebra: norm

# --- ML ---
# `import` (not `using`): all Flux references in this package are qualified.
import Flux

# --- Herb ecosystem ---
using HerbCore
using HerbGrammar
using HerbConstraints
using HerbInterpret
using HerbSearch
using HerbSpecification

# shared building blocks
include("embedding.jl")
include("encoding.jl")
include("see.jl")
include("data.jl")
include("costs.jl")
include("pcfg.jl")
include("scorer.jl")
include("iterators.jl")
include("search.jl")

# synthesizers, one folder each (Garden.jl layout)
include("DeepCoder/DeepCoder.jl")
include("Bustle/Bustle.jl")
include("WriteExecuteAssess/WriteExecuteAssess.jl")
include("HySynth/HySynth.jl")

export
    # string embedders
    AbstractEmbedder, HashEmbedder, CachedEmbedder, PrecomputedEmbedder,
    embed, embed_dim, save_cache,
    input_string, embedding_strings, embed_example, embed_spec,
    embed_rule, embed_grammar,

    # specification encoders
    SpecEncoder, encode, encoding_dim,
    ValueEncoder,
    PropertySignatureEncoder, signature, standard_properties,

    # value scorer (shared by Bustle and WriteExecuteAssess)
    ValueScorer, value_score, scorer_training_data, train_scorer!,
    subprograms,

    # training data
    GeneratedExample, rule_mask, generate_examples,
    fit_pcfg,

    # search drivers and their iterators
    guided_bottom_up_search, value_guided_search, repl_beam_search,
    BustleBUSIterator, ReplBeamIterator,
    Evaluator, n_correct, is_solution,
    weights_to_costs, scores_to_costs, tiebreak_costs

end # module HerbLearn
