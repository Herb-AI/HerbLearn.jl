# Write, Execute, Assess

Program synthesis with a REPL, from Ellis et al. (NeurIPS 2019; see
`ref.bib`): keep the search in states where every partial result is executed.
Each step *writes* a new value from the values at hand, *executes* it, and a
learned network *assesses* how useful it looks; beam search keeps the most
promising states.

The learned part is `HerbLearn.ValueScorer` -- the same shared building block
that drives `Bustle/`. What this module changes is only the search: a beam
that *discards* values outside it, where BUSTLE's complete enumeration merely
*delays* them. Running both with one trained scorer isolates exactly that
trade-off.

Simplifications w.r.t. the paper: the policy and value networks collapse into
the one scorer, and the search is a plain beam (no sequential Monte Carlo).

## Usage

```julia
using HerbLearn
using HerbLearn.WriteExecuteAssess: write_execute_assess

data = generate_examples(grammar, inputs, 500; start=:S, mod=MyFunctions)
scorer = ValueScorer()
train_scorer!(scorer, scorer_training_data(scorer, grammar, data; mod=MyFunctions))

program, enumerated = write_execute_assess(grammar, :S, problem;
                                           scorer, beam_width=16, mod=MyFunctions)
```
