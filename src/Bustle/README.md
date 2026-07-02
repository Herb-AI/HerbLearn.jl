# BUSTLE

Score every intermediate value during bottom-up search and delay the
unpromising ones. From "BUSTLE: Bottom-Up Program Synthesis Through
Learning-Guided Exploration" (Odena et al., ICLR 2021; see `ref.bib`), built
on property signatures (Odena & Sutton, ICLR 2020).

The learned part is `HerbLearn.ValueScorer`, a shared building block; this
module is a single call that hands it to the value-guided search driver. The
search stays complete: a low score delays a value in the cost-ordered bank, it
never discards it.

## Usage

```julia
using HerbLearn
using HerbLearn.Bustle: bustle

data = generate_examples(grammar, inputs, 500; start=:S, mod=MyFunctions)
scorer = ValueScorer()
train_scorer!(scorer, scorer_training_data(scorer, grammar, data; mod=MyFunctions))

program, enumerated = bustle(grammar, :S, problem; scorer, mod=MyFunctions)
```

The same trained `scorer` also drives `WriteExecuteAssess/` -- comparing the
two isolates the search strategy (complete enumeration that delays vs. a beam
that discards).
