using HerbLearn
using HerbLearn.DeepCoder: DeepCoderModel, train_deepcoder!, predict_rule_weights, deepcoder
using HerbGrammar, HerbCore, HerbSpecification
using Test

@testset "DeepCoder" begin
    g = INITIALS_GRAMMAR
    generated = generate_examples(g, INITIALS_INPUTS, 60;
                                  start=:S, max_depth=5, mod=Initials, seed=1)

    model = DeepCoderModel(ValueEncoder(HashEmbedder(dim=32)), g; hidden=[32])
    history = train_deepcoder!(model, generated; epochs=20, seed=1)
    @test history[end] <= history[1]                     # loss went down

    w = predict_rule_weights(model, INITIALS_PROBLEM.spec)
    @test length(w) == length(g.rules) && all(0 .<= w .<= 1)

    # a weak model must not break the search, only reorder it
    p, n = deepcoder(g, :S, INITIALS_PROBLEM; model=model,
                     mod=Initials, scale=4, max_programs=100_000)
    @test p !== nothing
    @test solves(p, INITIALS_PROBLEM)
end
