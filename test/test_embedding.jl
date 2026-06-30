using HerbLearn
using LinearAlgebra: norm, dot
using Serialization: serialize
using Test

cos_sim(a, b) = dot(a, b) / (norm(a) * norm(b))

@testset "embedding" begin

    @testset "HashEmbedder basics" begin
        e = HashEmbedder(dim=64)
        v = embed(e, "hello")

        @test v isa Vector{Float32}
        @test length(v) == embed_dim(e) == 64
        @test isapprox(norm(v), 1.0; atol=1e-5)        # L2-normalised
    end

    @testset "determinism" begin
        e = HashEmbedder(dim=32)
        @test embed(e, "concat") == embed(e, "concat")
        # a fresh embedder with the same params gives identical vectors
        @test embed(HashEmbedder(dim=32), "abc") == embed(HashEmbedder(dim=32), "abc")
    end

    @testset "empty string -> zero vector" begin
        e = HashEmbedder(dim=16)
        @test embed(e, "") == zeros(Float32, 16)
    end

    @testset "surface compositionality" begin
        # strings sharing substrings should be closer than unrelated strings
        e = HashEmbedder(dim=256, ngram=3)
        @test cos_sim(embed(e, "concat"), embed(e, "cat")) >
              cos_sim(embed(e, "concat"), embed(e, "xyzqrs"))
    end

    @testset "CachedEmbedder memoises" begin
        e = CachedEmbedder(HashEmbedder(dim=32))
        @test embed_dim(e) == 32
        v1 = embed(e, "foo")
        v2 = embed(e, "foo")
        @test v1 === v2                                 # same object on cache hit
        @test v1 == embed(HashEmbedder(dim=32), "foo")  # matches the inner embedder
        @test haskey(e.cache, "foo")
    end

    @testset "PrecomputedEmbedder lookup + fallback" begin
        table = Dict("a" => Float32[1, 0, 0], "b" => Float32[0, 1, 0])
        e = PrecomputedEmbedder(table)
        @test embed_dim(e) == 3
        @test embed(e, "a") == Float32[1, 0, 0]
        # miss without fallback errors
        @test_throws ErrorException embed(e, "missing")
        # miss with fallback uses the fallback embedder
        ef = PrecomputedEmbedder(table; fallback=HashEmbedder(dim=3))
        @test embed(ef, "missing") == embed(HashEmbedder(dim=3), "missing")
        # empty table + no fallback cannot infer dim
        @test_throws ArgumentError PrecomputedEmbedder(Dict{String,Vector{Float32}}())
    end

    @testset "PrecomputedEmbedder disk load" begin
        path = joinpath(mktempdir(), "table.jls")
        # produce a cache the way the offline LLM step would, then load it
        table = Dict("hello" => Float32[0.1, 0.2], "world" => Float32[0.3, 0.4])
        serialize(path, table)
        e = PrecomputedEmbedder(path)
        @test embed_dim(e) == 2
        @test embed(e, "world") == Float32[0.3, 0.4]
    end

    @testset "CachedEmbedder disk round-trip" begin
        path = joinpath(mktempdir(), "cache.jls")
        e = CachedEmbedder(HashEmbedder(dim=32); path=path)
        embed(e, "alpha"); embed(e, "beta")
        @test save_cache(e) == 2
        @test isfile(path)

        # a new embedder loads the warmed cache from disk
        e2 = CachedEmbedder(HashEmbedder(dim=32); path=path)
        @test haskey(e2.cache, "alpha")
        @test e2.cache["beta"] == e.cache["beta"]
    end
end
