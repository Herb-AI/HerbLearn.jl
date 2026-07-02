using HerbLearn
using HerbGrammar, HerbCore, HerbSpecification
using Test

include("initials.jl")   # shared fixture: the initials task

# Shared building blocks first, then one file per synthesizer (Garden.jl
# convention). Each file is a self-contained testset.
@testset "HerbLearn" begin
    include("test_embedding.jl")
    include("test_encoding.jl")
    include("test_see.jl")
    include("test_data.jl")
    include("test_costs.jl")
    include("test_scorer.jl")
    include("test_iterators.jl")
    include("test_search.jl")

    include("test_deepcoder.jl")
    include("test_bustle.jl")
    include("test_wea.jl")
    include("test_hysynth.jl")
end
