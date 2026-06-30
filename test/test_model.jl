using HerbLearn
using Flux
using LinearAlgebra: norm
using Test

@testset "model" begin
    n = 16
    R = 5                                   # number of rules
    model = UniversEModel(n; hidden=[8])

    @testset "dims" begin
        @test model.in_dim == 2n
        @test model.out_dim == n
        @test length(model(rand(Float32, 2n))) == n          # vector forward
        @test size(model(rand(Float32, 2n, 4))) == (n, 4)    # batch forward
    end

    @testset "cosine01 range & extremes" begin
        u = rand(Float32, n)
        @test cosine01(u, u) ≈ 1                              # identical -> 1
        @test isapprox(cosine01(u, -u), 0; atol=1e-5)         # opposite  -> 0
        @test 0 <= cosine01(rand(Float32, n), rand(Float32, n)) <= 1
    end

    @testset "predict_scores shapes & range" begin
        G = rand(Float32, n, R)
        s_vec = predict_scores(model, rand(Float32, 2n), G)
        @test length(s_vec) == R
        @test all(0 .<= s_vec .<= 1)

        S = predict_scores(model, rand(Float32, 2n, 3), G)
        @test size(S) == (R, 3)
        @test all(0 .<= S .<= 1)
        # batched columns match the per-spec vector calls
        spec = rand(Float32, 2n)
        @test predict_scores(model, reshape(spec, :, 1), G)[:, 1] ≈ predict_scores(model, spec, G)
    end

    @testset "differentiable through Flux" begin
        G = rand(Float32, n, R)
        spec = rand(Float32, 2n)
        target = Float32[1, 0, 0, 1, 0]
        loss(m) = sum(abs2, predict_scores(m, spec, G) .- target)
        grads = Flux.gradient(loss, model)
        @test grads[1] !== nothing
        # the first dense layer's weights receive a nonzero gradient
        @test any(!iszero, grads[1].mlp.layers[1].weight)
    end
end
