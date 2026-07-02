# HerbLearn.jl

Learning from and about programs: building blocks for neurally-guided program
synthesis in the [Herb](https://github.com/Herb-AI) ecosystem.

Neurally-guided synthesizers share the same subcomponents: something encodes
the examples, something scores search decisions, something runs a guided
search. Published methods differ in a few choices, but each paper ships them
as one entangled system. HerbLearn breaks the pipeline into reusable pieces so
that a published method is a short file, and a new method is a recombination:

- **Encoders** (`see.jl`, `embedding.jl`) -- how a heuristic sees the examples:
  `ValueEncoder` (embed the values, pluggable string embedders) or
  `PropertySignatureEncoder` (evaluate cheap checks; type-agnostic).
- **The value scorer** (`scorer.jl`) -- a small network judging one executed
  value against the examples, shared by two synthesizers below.
- **Search drivers** (`search.jl`) -- plain functions; the guidance is an
  argument whose shape the compiler checks:

  | driver                    | guidance argument       | consulted    |
  |---------------------------|-------------------------|--------------|
  | `guided_bottom_up_search` | a weight per rule       | once per task |
  | `value_guided_search`     | `score(program, outputs)` | every step, complete search |
  | `repl_beam_search`        | `score(program, outputs)` | every step, beam |

- **Training data** (`data.jl`, `pcfg.jl`) -- sample programs from the grammar,
  execute them, learn from what they use and compute; fit rule weights from
  any corpus of programs.

## The synthesizers

One folder per method, following [Garden.jl](https://github.com/Herb-AI/Garden.jl):
`method.jl` with one entry function, a `README.md`, and the paper in `ref.bib`.

| folder | entry function | method |
|--------|----------------|--------|
| `src/DeepCoder/` | `deepcoder` | predict rule weights from the examples, once (Balog et al., ICLR 2017) |
| `src/Bustle/` | `bustle` | score every value during bottom-up search (Odena et al., ICLR 2021) |
| `src/WriteExecuteAssess/` | `write_execute_assess` | beam search over executed REPL states (Ellis et al., NeurIPS 2019) |
| `src/HySynth/` | `hysynth` | distill LLM proposals into rule weights (Barke et al., NeurIPS 2024) | TODO: we call that a pCFG

Because the building blocks carry the weight, each of these files is mostly
documentation. `Bustle` and `WriteExecuteAssess` are one function call each:
they share the same trained scorer, and only the driver differs -- the
complete search *delays* low-scoring values, the beam *discards* them.
Running both isolates exactly that trade-off. (For a learned-but-not-neural
synthesizer in the same style, see Garden.jl's Probe.)

## Installation

```julia
using Pkg
Pkg.develop(path="path/to/HerbSearch")   # dev HerbSearch with the cost-based BUS
Pkg.develop(path="path/to/HerbLearn.jl")
```

## A complete example

The task: abbreviate a name to initials. TODO: training is usually done over an entire domain, not just a problem

```julia
using HerbLearn, HerbGrammar, HerbSpecification

# The functions the grammar's rules call.
module Initials
first_char(s) = string(first(s))
word(s, i)    = String(split(s)[i])
concat(a, b)  = a * b
end

grammar = @csgrammar begin
    S = x
    S = "."
    S = concat(S, S)
    S = first_char(S)
    S = word(S, N)
    N = 1
    N = 2
end

problem = Problem([IOExample(Dict(:x => "Ada Lovelace"), "A.L."),
                   IOExample(Dict(:x => "Alan Turing"),  "A.T.")])

# Unguided baseline: the same driver every rule-weight method uses.
program, enumerated = guided_bottom_up_search(
    grammar, :S, problem, ones(length(grammar.rules)); mod=Initials)
rulenode2expr(program, grammar)
# concat(first_char(x), concat(".", concat(first_char(word(x, 2)), ".")))
```

Training and running the learned methods:

```julia
using HerbLearn.DeepCoder: DeepCoderModel, train_deepcoder!, deepcoder
using HerbLearn.Bustle: bustle
using HerbLearn.WriteExecuteAssess: write_execute_assess
using HerbLearn.HySynth: hysynth, MockLLM

inputs = [ex.in for ex in problem.spec]
data = generate_examples(grammar, inputs, 500; start=:S, mod=Initials)

# DeepCoder: rule weights from the examples, once.
model = DeepCoderModel(ValueEncoder(HashEmbedder(dim=32)), grammar)
train_deepcoder!(model, data)
deepcoder(grammar, :S, problem; model, mod=Initials)

# One scorer, two synthesizers.
scorer = ValueScorer()
train_scorer!(scorer, scorer_training_data(scorer, grammar, data; mod=Initials))
bustle(grammar, :S, problem; scorer, mod=Initials)
write_execute_assess(grammar, :S, problem; scorer, mod=Initials)

# LLM-guided: any backend implementing `complete(llm, prompt)`.
hysynth(grammar, :S, problem; llm=MockLLM(["concat(first_char(x), \".\")"]), mod=Initials)
```

## Writing your own synthesizer

Pick a driver, hand it your guidance. A corpus-prior synthesizer, from
scratch:

```julia
corpus_prior(grammar, start, problem, corpus; kwargs...) =
    guided_bottom_up_search(grammar, start, problem, fit_pcfg(corpus, grammar); kwargs...)
```

A new method usually means one new piece -- a different encoder, a different
score, a different way to get weights -- and reusing the rest. If it needs a
genuinely new search, add a driver: the three in `search.jl` are 60-90 lines
each and written to be read.  TODO: REAds like slop

## Tests

```julia
using Pkg; Pkg.test("HerbLearn")
```

The shared building blocks are tested first, then each synthesizer end to end
on the initials task, so the suite doubles as executable documentation.
