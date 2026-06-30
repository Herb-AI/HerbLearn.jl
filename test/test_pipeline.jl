using HerbLearn
using HerbGrammar, HerbCore, HerbSpecification, HerbInterpret
using Test

@testset "pipeline (end-to-end)" begin
    # tiny string-transformation problem: append "!" to the input
    g = @csgrammar begin
        Start = s
        Start = "!"
        Start = "?"
        Start = Start * Start
    end
    spec = [
        IOExample(Dict(:s => "a"), "a!"),
        IOExample(Dict(:s => "bc"), "bc!"),
    ]
    # the intended solution program is  s * "!"  == RuleNode(4, [RuleNode(1), RuleNode(2)])
    solution = RuleNode(4, [RuleNode(1), RuleNode(2)])
    symtable = grammar2symboltable(g)
    solves(p) = all(execute_on_input(symtable, rulenode2expr(p, g), ex.in) == ex.out for ex in spec)
    @test solves(solution)   # sanity check the test's own target

    @testset "search_with_costs finds the solution (uniform)" begin
        costs = ones(Int, length(g.rules))
        res = search_with_costs(g, spec, costs; max_cost=6, max_programs=50_000)
        @test res.program !== nothing
        @test solves(res.program)
    end

    @testset "cost guidance reduces enumeration" begin
        # make the useless constant "?" (rule 3) expensive: solution never needs it
        uniform = ones(Int, length(g.rules))
        biased = [1, 1, 20, 1]
        ru = search_with_costs(g, spec, uniform; max_cost=6, max_programs=50_000)
        rb = search_with_costs(g, spec, biased; max_cost=6, max_programs=50_000)
        @test ru.program !== nothing && rb.program !== nothing
        @test solves(ru.program) && solves(rb.program)
        # deferring "?"-programs means fewer programs enumerated before the solution
        @test rb.enumerated <= ru.enumerated
    end

    @testset "full fit -> costs -> search" begin
        u = fit_universe(g, [ex.in for ex in spec];
                         embedder=HashEmbedder(dim=32),
                         num_programs=80, max_depth=4,
                         hidden=[32], epochs=60, seed=1)
        costs = guided_costs(u, g, spec; min_cost=1, scale=4)
        @test length(costs) == length(g.rules)
        @test all(costs .>= 1)

        res = search_with_costs(g, spec, costs; max_cost=18, max_programs=50_000)
        @test res.program !== nothing
        @test solves(res.program)
    end

    @testset "solve convenience driver" begin
        # unguided baseline
        ru = solve(g, spec; uniform=true, search_max_cost=6, search_max_programs=50_000)
        @test ru.program !== nothing && solves(ru.program)
        @test ru.universe === nothing

        # guided
        rg = solve(g, spec; embedder=HashEmbedder(dim=32), num_programs=80, max_depth=4,
                   hidden=[32], epochs=60, seed=1, scale=4,
                   search_max_cost=18, search_max_programs=50_000)
        @test rg.program !== nothing && solves(rg.program)
        @test rg.universe isa UniversE
    end
end
