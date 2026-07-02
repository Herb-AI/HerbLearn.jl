using HerbLearn
using HerbGrammar, HerbCore, HerbSpecification
using HerbSearch: CostBUSIterator
using Test

@testset "custom iterators" begin
    g = INITIALS_GRAMMAR
    ev = Evaluator(g, INITIALS_PROBLEM.spec; mod=Initials)

    @testset "BustleBUSIterator" begin
        @testset "zero penalty equals the vanilla cost iterator" begin
            vanilla = CostBUSIterator(g, :S, 7, ones(Int, length(g.rules)))
            hooked = BustleBUSIterator(g, :S; penalty=p -> 0, max_cost=7)
            @test collect(vanilla) == collect(hooked)
        end

        @testset "penalty delays, never discards" begin
            # penalize everything containing `word`: those programs appear
            # later, but they still appear
            uses_word(p) = p.ind == R_WORD || any(uses_word, p.children)
            hooked = BustleBUSIterator(g, :S; penalty=p -> uses_word(p) ? 3 : 0, max_cost=10)
            progs = collect(hooked)
            plain = collect(CostBUSIterator(g, :S, 7, ones(Int, length(g.rules))))
            @test issubset(Set(plain), Set(progs))
            first_word = findfirst(uses_word, progs)
            @test first_word !== nothing
            # some equally-sized program without `word` comes before it
            @test any(!uses_word(p) && length(p) >= length(progs[first_word])
                      for p in progs[1:first_word-1])
        end
    end

    @testset "ReplBeamIterator" begin
        score_all(prog, outs) = 0.5
        make(score; kwargs...) = ReplBeamIterator(g, :S;
            score=score, program_to_outputs=(p -> ev(p)), kwargs...)

        @testset "terminals come first, then composites" begin
            progs = collect(Iterators.take(make(score_all), 30))
            n_terminals = count(p -> length(p) == 1, progs)
            @test n_terminals == count(g.isterminal)
            @test all(length(p) == 1 for p in progs[1:n_terminals])
            @test all(length(p) > 1 for p in progs[n_terminals+1:end])
        end

        @testset "beam width bounds the states, max_steps ends the search" begin
            # width 1, 2 steps: the second step expands exactly one state
            progs = collect(make(score_all; beam_width=1, max_steps=2))
            @test !isempty(progs)
            @test maximum(length, progs) <= 5   # ≤ 2 growth steps from terminals
        end

        @testset "the score steers what survives" begin
            # score favors values whose outputs start like the target outputs;
            # the beam should reach first_char(word(x, 1)) territory
            target_like(prog, outs) =
                all(outs[i] isa String && !isempty(outs[i]) &&
                    startswith(INITIALS_PROBLEM.spec[i].out, first(outs[i]))
                    for i in eachindex(outs)) ? 1.0 : 0.0
            progs = collect(make(target_like; beam_width=8, max_steps=4))
            wanted = RuleNode(R_FIRST, [RuleNode(R_WORD, [RuleNode(R_X), RuleNode(R_ONE)])])
            @test wanted in progs
        end
    end
end
