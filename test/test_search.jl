using HerbLearn
using HerbGrammar, HerbCore, HerbSpecification
using Test

@testset "search drivers" begin
    g, problem = INITIALS_GRAMMAR, INITIALS_PROBLEM

    @testset "weights_to_costs" begin
        costs = weights_to_costs([1.0, 0.5, 0.0]; min_cost=1, scale=10)
        @test costs == [1, 6, 11]                     # best weight -> min cost
        @test weights_to_costs([0.0, 0.0]) == [1, 1]  # all-zero weights: uniform
    end

    @testset "guided_bottom_up_search, uniform weights (the baseline)" begin
        p, n = guided_bottom_up_search(g, :S, problem, ones(length(g.rules));
                                       mod=Initials, max_programs=50_000)
        @test p !== nothing
        @test solves(p, problem)
        @test n > 0
    end

    @testset "guided_bottom_up_search follows the weights" begin
        # make the needed rules cheap and the (useless here) word/index rules
        # expensive: the search should not get slower than uniform
        w = ones(length(g.rules))
        w[R_WORD] = w[R_TWO] = 0.1
        pu, nu = guided_bottom_up_search(g, :S, FIRST_INITIAL_PROBLEM, ones(length(g.rules));
                                         mod=Initials, max_programs=50_000, scale=4)
        pw, nw = guided_bottom_up_search(g, :S, FIRST_INITIAL_PROBLEM, w;
                                         mod=Initials, max_programs=50_000, scale=4)
        @test pu !== nothing && pw !== nothing
        @test nw <= nu
    end

    @testset "value_guided_search with a constant score is complete" begin
        p, n = value_guided_search(g, :S, problem, (prog, outs) -> 0.5;
                                   mod=Initials, max_programs=50_000)
        @test p !== nothing
        @test solves(p, problem)
    end

    @testset "repl_beam_search finds a one-action solution" begin
        # target "A." is terminals + two concat steps away; a neutral score
        # keeps everything in the beam long enough on this tiny grammar
        p, n = repl_beam_search(g, :S, FIRST_INITIAL_PROBLEM, (prog, outs) -> 0.5;
                                mod=Initials, beam_width=32, max_steps=6,
                                max_programs=50_000)
        @test p !== nothing
        @test solves(p, FIRST_INITIAL_PROBLEM)
    end

    @testset "budget is respected" begin
        p, n = guided_bottom_up_search(g, :S, problem, ones(length(g.rules));
                                       mod=Initials, max_programs=10)
        @test p === nothing
        @test n <= 10
    end
end

@testset "fit_pcfg" begin
    g = INITIALS_GRAMMAR
    # concat(first_char(x), ".") uses rules concat, first_char, x, "."
    p = RuleNode(R_CONCAT, [RuleNode(R_FIRST, [RuleNode(R_X)]), RuleNode(R_DOT)])
    w = fit_pcfg([p], g; smoothing=1.0)
    @test sum(w[g.bytype[:S]]) ≈ 1.0
    @test sum(w[g.bytype[:N]]) ≈ 1.0
    @test w[R_CONCAT] > w[R_WORD]        # used rule outweighs unused rule
    @test w[R_ONE] ≈ w[R_TWO]            # smoothing only: N rules stay equal
end
