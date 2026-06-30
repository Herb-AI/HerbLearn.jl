using HerbLearn
using HerbGrammar, HerbSpecification
using Test

@testset "costs" begin
    @testset "scores_to_costs monotone & integer" begin
        costs = scores_to_costs([0.0, 0.5, 1.0]; min_cost=1, scale=10)
        @test costs isa Vector{Int}
        @test costs == [11, 6, 1]                      # score 1 -> min_cost, 0 -> min+scale
        # strictly decreasing in score
        s = [0.1, 0.3, 0.9]
        c = scores_to_costs(s; scale=20)
        @test issorted(c; rev=true)
        @test all(c .>= 1)
    end

    @testset "scores_to_costs clamps & validates" begin
        @test scores_to_costs([-1.0, 2.0]; min_cost=2, scale=4) == [6, 2]  # clamped to [0,1]
        @test_throws ArgumentError scores_to_costs([0.5]; min_cost=0)
    end

    @testset "assign_costs over a grammar" begin
        e = HashEmbedder(dim=16)
        g = @csgrammar begin
            Start = "x"
            Start = s
            Start = Start * Start
        end
        model = UniversEModel(embed_dim(e); hidden=[8])
        spec = [IOExample(Dict(:s => "a"), "ax")]
        costs = assign_costs(model, e, g, spec; min_cost=1, scale=10)
        @test length(costs) == length(g.rules)
        @test costs isa Vector{Int}
        @test all(costs .>= 1)
    end
end
