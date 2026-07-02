using HerbLearn
using HerbSpecification
using Test

@testset "spec encoders (Q1)" begin
    @testset "property signatures aggregate to {0, 0.5, 1}" begin
        enc = PropertySignatureEncoder(Function[
            (a, b) -> a == b,                 # holds on all / none / some
            (a, b) -> error("does not apply") # always errors -> 0.5
        ])
        @test encoding_dim(enc) == 2

        @test signature(enc, [("a", "a"), ("b", "b")]) == Float32[1.0, 0.5]
        @test signature(enc, [("a", "x"), ("b", "y")]) == Float32[0.0, 0.5]
        @test signature(enc, [("a", "a"), ("b", "y")]) == Float32[0.5, 0.5]
    end

    @testset "standard properties encode a spec" begin
        enc = PropertySignatureEncoder()
        spec = [IOExample(Dict(:x => "Ada Lovelace"), "A.L."),
                IOExample(Dict(:x => "Alan Turing"), "A.T.")]
        sig = encode(enc, spec)
        @test length(sig) == encoding_dim(enc)
        @test all(s -> s in (0.0f0, 0.5f0, 1.0f0), sig)
        # "output shorter than input" holds on both examples
        shorter = findfirst(p -> p("abc", "xy") === true && p("a", "abc") === false &&
                                 p("abc", "a") === true,
                            enc.properties)
        @test sig[shorter] == 1.0f0
    end

    @testset "value encoder wraps an embedder" begin
        enc = ValueEncoder(HashEmbedder(dim=16))
        spec = [IOExample(Dict(:x => "abc"), "c")]
        v = encode(enc, spec)
        @test length(v) == encoding_dim(enc) == 32
        # order of examples must not matter (deep set)
        spec2 = [IOExample(Dict(:x => "abc"), "c"), IOExample(Dict(:x => "de"), "e")]
        @test encode(enc, spec2) ≈ encode(enc, reverse(spec2))
    end
end
