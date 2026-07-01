using HerbLearn
using HerbGrammar, HerbSpecification
using LinearAlgebra: normalize, dot
using Test

@testset "training" begin
    n = 16
    R = 4
    G = rand(Float32, n, R)

    @testset "loss edge cases (need both positives and negatives)" begin
        model = UniversEModel(n; hidden=[8])
        spec = rand(Float32, 2n)
        for lossfn in (pairwise_loss, contrastive_loss)
            @test lossfn(model, spec, falses(R), G) == 0          # no positives
            @test lossfn(model, spec, trues(R), G) == 0           # no negatives
            @test lossfn(model, spec, BitVector([1, 0, 1, 0]), G) >= 0
        end
    end

    @testset "make_training_data from generated examples" begin
        e = HashEmbedder(dim=n)
        g = @csgrammar begin
            Start = "x"
            Start = s
            Start = Start * Start
        end
        gen = generate_examples(g, [Dict(:s => "a")], 3; start=:Start, max_depth=3, seed=1)
        td = make_training_data(e, gen)
        @test length(td) == length(gen)
        @test all(d -> length(d.spec_emb) == 2n, td)
        @test all(d -> length(d.rule_mask) == length(g.rules), td)
    end

    @testset "loss decreases & ranks correctly ($lossfn)" for lossfn in (contrastive_loss, pairwise_loss)
        # Two distinct specs, each mapped to a different rule pattern. The model
        # should learn to rank the right rules up for each spec.
        n2 = 12
        # two orthogonal rule directions so the groups {1,2} and {3,4} are
        # actually separable in rule-embedding space
        u = normalize(randn(Float32, n2))
        w = normalize(randn(Float32, n2))
        w = normalize(w .- dot(w, u) .* u)
        G2 = hcat(u, u, w, w)                      # rules 1,2 ~ u ; rules 3,4 ~ w
        sA = randn(Float32, 2 * n2)
        sB = randn(Float32, 2 * n2)
        data = [
            TrainingDatum(sA, BitVector([1, 1, 0, 0])),
            TrainingDatum(sB, BitVector([0, 0, 1, 1])),
        ]
        model = UniversEModel(n2; hidden=[16, 16])
        history = train!(model, data, G2; lossfn=lossfn, epochs=300, lr=1e-2, seed=0)

        @test length(history) == 300
        @test history[end] < history[1]            # learned something

        # after training, the intended rules outrank the others for each spec
        scoresA = predict_scores(model, sA, G2)
        @test minimum(scoresA[[1, 2]]) > maximum(scoresA[[3, 4]])
        scoresB = predict_scores(model, sB, G2)
        @test minimum(scoresB[[3, 4]]) > maximum(scoresB[[1, 2]])
    end
end
