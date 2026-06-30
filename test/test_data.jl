using HerbLearn
using HerbGrammar, HerbCore, HerbSpecification
using Test

@testset "data generation" begin
    # tiny string grammar: build strings from a constant, the input, and concat
    g = @csgrammar begin
        Start = "x"
        Start = s
        Start = Start * Start
    end

    @testset "rule_mask marks used rules" begin
        # program: concat(s, "x")  -> rule 3 over (rule 2, rule 1)
        prog = RuleNode(3, [RuleNode(2), RuleNode(1)])
        m = rule_mask(prog, length(g.rules))
        @test m == BitVector([true, true, true])

        prog2 = RuleNode(1)               # just "x"
        @test rule_mask(prog2, length(g.rules)) == BitVector([true, false, false])
    end

    @testset "generate_examples basic contract" begin
        inputs = [Dict(:s => "a"), Dict(:s => "bb")]
        data = generate_examples(g, inputs, 5;
                                 start=:Start, min_depth=1, max_depth=4, seed=1)

        @test !isempty(data)
        @test length(data) <= 5
        for d in data
            @test d isa GeneratedExample
            @test length(d.spec) == length(inputs)            # one example per input
            @test length(d.rule_mask) == length(g.rules)
            @test any(d.rule_mask)                            # at least one rule used
            # outputs are produced by executing the program on the given inputs
            for (ex, input) in zip(d.spec, inputs)
                @test ex.in == input
                @test ex.out isa AbstractString
            end
        end

        # programs are distinct
        progs = [d.program for d in data]
        @test length(unique(progs)) == length(progs)
    end

    @testset "exclude keeps a program out" begin
        inputs = [Dict(:s => "a")]
        target = RuleNode(2)                                  # the program `s`
        data = generate_examples(g, inputs, 20;
                                 start=:Start, min_depth=1, max_depth=3,
                                 exclude=Set([target]), seed=2)
        @test all(d -> d.program != target, data)
    end

    @testset "reproducible with seeded rng" begin
        inputs = [Dict(:s => "a")]
        d1 = generate_examples(g, inputs, 5; start=:Start, max_depth=3, seed=7)
        d2 = generate_examples(g, inputs, 5; start=:Start, max_depth=3, seed=7)
        @test [d.program for d in d1] == [d.program for d in d2]
    end
end
