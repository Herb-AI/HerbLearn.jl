using HerbLearn
using HerbGrammar, HerbSpecification
using Test

@testset "encoding" begin
    e = HashEmbedder(dim=16)
    n = embed_dim(e)

    g = @csgrammar begin
        S = "a"
        S = x
        S = S * S
    end

    @testset "input_string is deterministic & key-sorted" begin
        d = Dict(:b => "2", :a => "1")
        @test input_string(d) == "1 2"           # sorted by key a,b
        @test input_string("plain") == "plain"
    end

    @testset "embed_example shape = 2n" begin
        ex = IOExample(Dict(:x => "p"), "pa")
        v = embed_example(e, ex)
        @test length(v) == 2n
    end

    @testset "embed_spec is order-invariant deep set" begin
        ex1 = IOExample(Dict(:x => "p"), "pa")
        ex2 = IOExample(Dict(:x => "q"), "qa")
        s12 = embed_spec(e, [ex1, ex2])
        s21 = embed_spec(e, [ex2, ex1])
        @test length(s12) == 2n
        @test s12 ≈ s21                          # mean -> permutation invariant
        # equals the mean of the individual example embeddings
        @test s12 ≈ (embed_example(e, ex1) .+ embed_example(e, ex2)) ./ 2
    end

    @testset "empty spec -> zeros" begin
        @test embed_spec(e, IOExample[]) == zeros(Float32, 2n)
    end

    @testset "embed_rule / embed_grammar shapes" begin
        @test length(embed_rule(e, g, 1)) == n
        M = embed_grammar(e, g)
        @test size(M) == (n, length(g.rules))
        @test M[:, 2] == embed_rule(e, g, 2)
    end

    @testset "embedding_strings covers rules + I/O" begin
        spec = [IOExample(Dict(:x => "p"), "pa")]
        strs = Set(embedding_strings(g, [spec]))
        @test "S * S" in strs              # a rule string
        @test "p" in strs                  # the input string
        @test "pa" in strs                 # the output string
        # every rule's string representation is present
        @test all(string(r) in strs for r in g.rules)
    end
end
