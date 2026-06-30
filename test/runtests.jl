using HerbLearn
using Test

# Each file is a self-contained, simple testset so individual stages can be
# debugged in isolation. Add new stages here as they are built.
@testset "HerbLearn" begin
    include("test_embedding.jl")
    include("test_encoding.jl")
    include("test_data.jl")
    include("test_model.jl")
    include("test_train.jl")
    include("test_costs.jl")
    include("test_pipeline.jl")
end
