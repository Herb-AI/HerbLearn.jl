@testset verbose=true "DataGeneration" begin
    @testset "test data generation" begin
        g₁ = @cfgrammar begin
            Real = |(0:3)
            Real = x
            Real = Real + Real
        end

        problem = Herb.HerbData.Problem([Herb.HerbData.IOExample(Dict(:x => x), 3x) for x ∈ 1:5], "example")

        io_data = Herb.HerbLearn.generate_data(g₁, problem, 4, 3, Herb.HerbSearch.get_bfs_enumerator, :Real)

        @test length(io_data) == 15
    end
end
