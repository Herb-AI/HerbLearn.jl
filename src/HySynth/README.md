# HySynth

Distill LLM proposals into rule weights for a complete search. From "HYSYNTH:
Context-Free LLM Approximation for Guiding Program Synthesis" (Barke et al.,
NeurIPS 2024; see `ref.bib`).

The LLM proposes whole programs; most are wrong, some do not even parse. That
is fine: instead of trusting them, count which grammar rules the parseable
proposals use, fit a smoothed PCFG, and run the ordinary complete bottom-up
search under those weights. Wrong proposals cost time, never correctness.

The LLM sits behind a one-function interface, `complete(llm, prompt) ->
Vector{String}`, so any backend works: an HTTP API, a local model, or
`MockLLM` (canned responses) in tests. The prompt and the parsing are plain
functions you can swap -- this module is deliberately boilerplate to build
variants on.

## Usage

```julia
using HerbLearn
using HerbLearn.HySynth: hysynth, MockLLM

llm = MockLLM(["concat(first_char(x), \".\")"])   # or any `LLM` backend
program, enumerated = hysynth(grammar, :S, problem; llm, mod=MyFunctions)
```
