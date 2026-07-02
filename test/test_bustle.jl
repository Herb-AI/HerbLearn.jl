using HerbLearn
using HerbLearn.Bustle: bustle
using HerbGrammar, HerbSpecification
using Test

@testset "Bustle" begin
    g = INITIALS_GRAMMAR
    generated = generate_examples(g, INITIALS_INPUTS, 60;
                                  start=:S, max_depth=5, mod=Initials, seed=1)

    scorer = ValueScorer()
    train_scorer!(scorer, scorer_training_data(scorer, g, generated; mod=Initials, seed=1);
                  epochs=15, seed=1)

    p, n = bustle(g, :S, INITIALS_PROBLEM; scorer=scorer, mod=Initials, max_programs=100_000)
    @test p !== nothing
    @test solves(p, INITIALS_PROBLEM)
end
