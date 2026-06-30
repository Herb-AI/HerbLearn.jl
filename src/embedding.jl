# =============================================================================
# Embedders
#
# An embedder turns an arbitrary string into a fixed-size vector. This is the
# "universal embedding function" f_e from the UniversE paper. Everything
# downstream (spec/rule encoding, learning, costs) only ever sees the resulting
# `Vector{Float32}`, so the actual model that produces them is fully pluggable
# and is computed *offline* (and cached). That keeps any heavy LLM completely
# out of the learning/search hot path.
#
# Backends:
#   - `HashEmbedder`   : pure-Julia hashed character n-grams. Zero dependencies,
#                        deterministic, instant. The default, used for tests and
#                        fast iteration on a laptop.
#   - `CachedEmbedder` : wraps any embedder with an on-disk memoisation cache.
#   - (a real LLM backend, e.g. via Transformers.jl, plugs in here later by
#      subtyping `AbstractEmbedder` and implementing `embed`/`embed_dim`.)
# =============================================================================

"""
    AbstractEmbedder

Supertype for every embedding backend. A concrete embedder must implement:

- `embed(e, s::AbstractString)::Vector{Float32}` — embed a single string.
- `embed_dim(e)::Int` — the length of every vector `embed` returns.
"""
abstract type AbstractEmbedder end

"""
    embed(e::AbstractEmbedder, s::AbstractString)::Vector{Float32}

Embed a single string into a fixed-size vector of length `embed_dim(e)`.
"""
function embed end

"""
    embed_dim(e::AbstractEmbedder)::Int

Length of the vectors produced by `e`.
"""
function embed_dim end

"""
    embed(e, ss::AbstractVector{<:AbstractString})::Matrix{Float32}

Embed several strings at once, returning an `embed_dim(e) × length(ss)` matrix
(one column per string). Backends may override this for batching.
"""
function embed(e::AbstractEmbedder, ss::AbstractVector{<:AbstractString})
    isempty(ss) && return Matrix{Float32}(undef, embed_dim(e), 0)
    return reduce(hcat, (embed(e, s) for s in ss))
end

# -----------------------------------------------------------------------------
# HashEmbedder
# -----------------------------------------------------------------------------

"""
    HashEmbedder(; dim=64, ngram=3, seed=0x9e3779b97f4a7c15)

A pure-Julia embedder based on the *feature hashing* (hashing trick) of
character n-grams. For every n-gram (n = 1..`ngram`) of the input string we
hash it into one of `dim` buckets with a signed weight, then L2-normalise.

This is deterministic, has no dependencies, and is fast enough to run inline.
It captures *surface* compositional similarity for free — strings sharing
substrings (e.g. `"concat"` and `"cat"`) land closer than unrelated strings —
which is enough to develop and test the whole pipeline offline. It does **not**
capture deeper semantics (e.g. `"1" ≈ "0"`); a real LLM backend is needed for
that, and drops in behind the same interface.
"""
struct HashEmbedder <: AbstractEmbedder
    dim::Int
    ngram::Int
    seed::UInt64

    function HashEmbedder(dim::Int, ngram::Int, seed::UInt64)
        dim > 0 || throw(ArgumentError("dim must be positive, got $dim"))
        ngram > 0 || throw(ArgumentError("ngram must be positive, got $ngram"))
        new(dim, ngram, seed)
    end
end

HashEmbedder(; dim::Int=64, ngram::Int=3, seed=0x9e3779b97f4a7c15) =
    HashEmbedder(dim, ngram, UInt64(seed))

embed_dim(e::HashEmbedder) = e.dim

function embed(e::HashEmbedder, s::AbstractString)::Vector{Float32}
    v = zeros(Float32, e.dim)
    chars = collect(s)
    L = length(chars)
    L == 0 && return v  # empty string -> zero vector

    for n in 1:e.ngram
        L < n && break
        for i in 1:(L - n + 1)
            gram = String(chars[i:(i + n - 1)])
            h = hash((n, gram), e.seed)
            idx = Int(h % e.dim) + 1
            # a second hash bit decides the sign, halving collision cancellation bias
            sgn = iszero(h & 0x1) ? 1.0f0 : -1.0f0
            @inbounds v[idx] += sgn
        end
    end

    nrm = norm(v)
    nrm > 0 && (v ./= nrm)
    return v
end

# -----------------------------------------------------------------------------
# CachedEmbedder
# -----------------------------------------------------------------------------

"""
    CachedEmbedder(inner::AbstractEmbedder; path=nothing)

Wrap `inner` so each distinct string is embedded only once. Results are kept in
an in-memory `Dict` and, if `path` is given, loaded from / saved to disk with
`save_cache`. This is what makes an expensive offline LLM backend practical:
warm the cache once, then every later run just reads vectors.
"""
mutable struct CachedEmbedder{E<:AbstractEmbedder} <: AbstractEmbedder
    inner::E
    cache::Dict{String,Vector{Float32}}
    path::Union{Nothing,String}
end

function CachedEmbedder(inner::AbstractEmbedder; path::Union{Nothing,AbstractString}=nothing)
    cache = if path !== nothing && isfile(path)
        deserialize(path)::Dict{String,Vector{Float32}}
    else
        Dict{String,Vector{Float32}}()
    end
    return CachedEmbedder(inner, cache, path === nothing ? nothing : String(path))
end

embed_dim(e::CachedEmbedder) = embed_dim(e.inner)

function embed(e::CachedEmbedder, s::AbstractString)::Vector{Float32}
    return get!(e.cache, String(s)) do
        embed(e.inner, s)
    end
end

"""
    PrecomputedEmbedder(table::Dict{String,Vector{Float32}}; fallback=nothing)
    PrecomputedEmbedder(path::AbstractString; fallback=nothing)

An embedder that only *looks up* precomputed vectors — typically the cache file
produced offline by a heavy LLM backend running in its own environment (see
`examples/llm_env`). This is how an HF/Transformers.jl model is used without it
ever being a dependency of the main pipeline: embed everything once, serialize a
`String → Vector{Float32}` dict, then read it here.

On a cache miss it falls back to `fallback` (another `AbstractEmbedder`) if one
is given, otherwise it errors — which surfaces any string that was not embedded
ahead of time.
"""
struct PrecomputedEmbedder{F} <: AbstractEmbedder
    table::Dict{String,Vector{Float32}}
    dim::Int
    fallback::F
end

function PrecomputedEmbedder(table::Dict{String,Vector{Float32}}; fallback=nothing)
    dim = if !isempty(table)
        length(first(values(table)))
    elseif fallback !== nothing
        embed_dim(fallback)
    else
        throw(ArgumentError("empty table and no fallback: cannot determine embedding dimension"))
    end
    return PrecomputedEmbedder(table, dim, fallback)
end

PrecomputedEmbedder(path::AbstractString; fallback=nothing) =
    PrecomputedEmbedder(deserialize(path)::Dict{String,Vector{Float32}}; fallback=fallback)

embed_dim(e::PrecomputedEmbedder) = e.dim

function embed(e::PrecomputedEmbedder, s::AbstractString)::Vector{Float32}
    key = String(s)
    haskey(e.table, key) && return e.table[key]
    e.fallback === nothing &&
        error("no precomputed embedding for $(repr(key)); embed it offline first or pass a fallback embedder")
    return embed(e.fallback, s)
end

"""
    save_cache(e::CachedEmbedder)

Persist the cache to `e.path` (no-op if no path was configured). Returns the
number of cached entries.
"""
function save_cache(e::CachedEmbedder)
    e.path === nothing && return length(e.cache)
    mkpath(dirname(abspath(e.path)))
    serialize(e.path, e.cache)
    return length(e.cache)
end
