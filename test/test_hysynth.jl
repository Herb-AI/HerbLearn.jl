using HerbLearn
using HerbLearn.HySynth: hysynth, MockLLM, complete, default_prompt, parse_proposals
using HerbGrammar, HerbSpecification
using Test

@testset "HySynth" begin
    g = INITIALS_GRAMMAR

    llm = MockLLM([
        "concat(first_char(x), \".\")",
        "concat(first_char(word(x, 2)), \".\")",
        "sorry, I am a language model and this is not a program",
    ])

    @testset "parsing drops what does not parse" begin
        @test length(parse_proposals(complete(llm, "ignored"), g)) == 2
    end

    @testset "the prompt shows the grammar and the examples" begin
        prompt = default_prompt(g, INITIALS_PROBLEM.spec)
        @test occursin("Ada Lovelace", prompt) && occursin("concat", prompt)
    end

    @testset "fitted weights favor proposed rules" begin
        w = fit_pcfg(parse_proposals(complete(llm, ""), g), g)
        @test w[R_CONCAT] > w[R_WORD]   # proposals used concat more than word
    end

    @testset "end to end" begin
        p, n = hysynth(g, :S, INITIALS_PROBLEM; llm=llm, mod=Initials,
                       scale=4, max_programs=100_000)
        @test p !== nothing
        @test solves(p, INITIALS_PROBLEM)
    end
end
