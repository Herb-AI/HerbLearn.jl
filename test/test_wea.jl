using HerbLearn
using HerbLearn.WriteExecuteAssess: write_execute_assess
using HerbGrammar, HerbSpecification
using Test

@testset "WriteExecuteAssess" begin
    g = INITIALS_GRAMMAR
    generated = generate_examples(g, INITIALS_INPUTS, 60;
                                  start=:S, max_depth=5, mod=Initials, seed=1)

    # the same scorer type Bustle uses; only the search differs
    scorer = ValueScorer()
    train_scorer!(scorer, scorer_training_data(scorer, g, generated; mod=Initials, seed=1);
                  epochs=15, seed=1)

    # incomplete search: check the mechanics on the easy target, where the
    # solution is a few actions away from the terminals
    p, n = write_execute_assess(g, :S, FIRST_INITIAL_PROBLEM; scorer=scorer,
                                mod=Initials, beam_width=16, max_steps=10,
                                max_programs=100_000)
    @test p !== nothing
    @test solves(p, FIRST_INITIAL_PROBLEM)
end
