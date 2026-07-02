# =============================================================================
# Q1 -- what the heuristic sees: encoding specifications.
#
# A specification is a set of input/output examples; a network needs a vector.
# The two encoder families from the literature:
#
#   ValueEncoder             embed the values themselves (DeepCoder, RobustFill).
#                            Precise, but committed to the value types it was
#                            built for. Here: any string embedder from
#                            embedding.jl, mean-pooled over examples (a deep
#                            set, so the order of examples does not matter).
#
#   PropertySignatureEncoder evaluate a fixed list of cheap checks on each
#                            input/output pair (BUSTLE, CrossBeam). Type-agnostic
#                            and independent of the number of examples, at the
#                            price of abstraction: tasks with the same signature
#                            are indistinguishable.
#
# Both implement the same two-function interface, so every heuristic in
# methods/ is generic over the encoder.
# =============================================================================

"""
    SpecEncoder

Supertype of specification encoders. A concrete encoder implements:

- `encode(enc, spec::AbstractVector{<:IOExample})::Vector{Float32}`
- `encoding_dim(enc)::Int` -- the length of every vector `encode` returns.
"""
abstract type SpecEncoder end

"""
    encode(enc::SpecEncoder, spec::AbstractVector{<:IOExample})::Vector{Float32}

Encode a whole specification as a fixed-length vector.
"""
function encode end

"""
    encoding_dim(enc::SpecEncoder)::Int

Length of the vectors produced by `enc`.
"""
function encoding_dim end

# -----------------------------------------------------------------------------
# Value encoding
# -----------------------------------------------------------------------------

"""
    ValueEncoder(embedder::AbstractEmbedder)

Encode a spec by embedding each example's input and output strings with
`embedder` (concatenated, so one example is a `2 × embed_dim` vector) and
averaging over the examples. See `embed_spec` for the details.
"""
struct ValueEncoder{E<:AbstractEmbedder} <: SpecEncoder
    embedder::E
end

encoding_dim(enc::ValueEncoder) = 2 * embed_dim(enc.embedder)
encode(enc::ValueEncoder, spec::AbstractVector{<:IOExample}) = embed_spec(enc.embedder, spec)

# -----------------------------------------------------------------------------
# Property signatures
# -----------------------------------------------------------------------------

"""
    PropertySignatureEncoder(properties::Vector{Function})
    PropertySignatureEncoder()   # standard_properties()

Encode a pair of values `(a, b)` -- or a whole spec -- by evaluating each property
`p(a, b)::Bool` and aggregating over examples:

    1.0  if the property holds on every example,
    0.0  if it holds on none,
    0.5  if it holds on some (or errors / does not apply).

The resulting vector has one entry per property, regardless of value types and
of the number of examples [odena2020property]. `signature(enc, pairs)` computes
it for arbitrary value pairs, which is what lets the same encoder describe a
candidate value against the target outputs (BUSTLE) as well as a specification.
"""
struct PropertySignatureEncoder <: SpecEncoder
    properties::Vector{Function}
end

PropertySignatureEncoder() = PropertySignatureEncoder(standard_properties())

encoding_dim(enc::PropertySignatureEncoder) = length(enc.properties)

"""
    signature(enc::PropertySignatureEncoder, pairs) -> Vector{Float32}

The signature of a set of value pairs `(a, b)`. Each property is evaluated on
every pair; errors and non-Bool results count as "does not apply" and push the
entry to `0.5`.
"""
function signature(enc::PropertySignatureEncoder, pairs)::Vector{Float32}
    sig = Vector{Float32}(undef, length(enc.properties))
    for (k, p) in enumerate(enc.properties)
        n_true, n_false = 0, 0
        for (a, b) in pairs
            r = try
                p(a, b)
            catch
                missing
            end
            r === true && (n_true += 1)
            r === false && (n_false += 1)
        end
        n = n_true + n_false
        sig[k] = n == 0        ? 0.5f0 :
                 n_true == n   ? 1.0f0 :
                 n_false == n  ? 0.0f0 : 0.5f0
    end
    return sig
end

encode(enc::PropertySignatureEncoder, spec::AbstractVector{<:IOExample}) =
    signature(enc, ((_input_value(ex.in), ex.out) for ex in spec))

# The raw input value: unwrap single-argument Dicts so properties see the value,
# not the Dict. Multi-argument inputs are joined to a string.
_input_value(input::AbstractDict) =
    length(input) == 1 ? first(values(input)) : input_string(input)
_input_value(input) = input

"""
    standard_properties() -> Vector{Function}

A small, readable battery of properties of an `(a, b)` pair covering strings
and integers -- the kinds of checks in BUSTLE's signature set. Each takes the
pair and returns a `Bool` (or throws, counting as "does not apply"). Write your
own list for other domains; anything callable works.
"""
function standard_properties()
    return Function[
        (a, b) -> a == b,
        (a, b) -> string(a) == string(b),
        (a, b) -> occursin(string(b), string(a)),          # b appears in a
        (a, b) -> occursin(string(a), string(b)),          # a appears in b
        (a, b) -> startswith(string(b), string(a)),
        (a, b) -> endswith(string(b), string(a)),
        (a, b) -> length(string(b)) < length(string(a)),
        (a, b) -> length(string(b)) == length(string(a)),
        (a, b) -> lowercase(string(b)) == string(b),
        (a, b) -> uppercase(string(b)) == string(b),
        (a, b) -> b isa Integer,
        (a, b) -> b isa Integer && b == 0,
        (a, b) -> b isa Integer && b > 0,
        (a, b) -> a isa Integer && b isa Integer && b == a,
        (a, b) -> a isa Integer && b isa Integer && b > a,
        (a, b) -> a isa Integer && b isa Integer && abs(b) < abs(a),
    ]
end
