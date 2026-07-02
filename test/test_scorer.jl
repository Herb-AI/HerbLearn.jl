using HerbLearn
using HerbGrammar, HerbCore, HerbSpecification
using Test

@testset "value scorer" begin
    g = INITIALS_GRAMMAR
    spec = INITIALS_PROBLEM.spec
    generated = generate_examples(g, INITIALS_INPUTS, 40;
                                  start=:S, max_depth=5, mod=Initials, seed=1)

    scorer = ValueScorer()
    data = scorer_training_data(scorer, g, generated; mod=Initials, seed=1)
    @test !isempty(data)
    @test all(d -> d.label in (0.0f0, 1.0f0), data)

    history = train_scorer!(scorer, data; epochs=15, seed=1)
    @test history[end] <= history[1]    # loss went down

    prog = RuleNode(R_FIRST, [RuleNode(R_X)])
    ev = Evaluator(g, spec; mod=Initials)
    @test 0.0 <= value_score(scorer, spec, ev(prog)) <= 1.0

    @testset "subprograms" begin
        p = RuleNode(R_CONCAT, [RuleNode(R_FIRST, [RuleNode(R_X)]), RuleNode(R_DOT)])
        @test length(subprograms(p)) == 4   # one per node
    end
end
