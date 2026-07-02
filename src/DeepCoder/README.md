# DeepCoder

Predict which grammar rules the solution uses, once, from the input--output
examples; then search the reweighted grammar. From "DeepCoder: Learning to
Write Programs" (Balog et al., ICLR 2017; see `ref.bib`). The original paper
guides a DFS / sort-and-add enumeration; here the weights drive the shared
cost-based bottom-up driver, which plays the same role.

## Usage

```julia
using HerbLearn
using HerbLearn.DeepCoder: DeepCoderModel, train_deepcoder!, deepcoder

# training data: programs sampled from the grammar, executed on real inputs
data = generate_examples(grammar, inputs, 500; start=:S, mod=MyFunctions)

model = DeepCoderModel(ValueEncoder(HashEmbedder(dim=32)), grammar)
train_deepcoder!(model, data; epochs=100)

program, enumerated = deepcoder(grammar, :S, problem; model, mod=MyFunctions)
```

The encoder is pluggable (`ValueEncoder` over any string embedder, or
`PropertySignatureEncoder`); the output layer has one unit per grammar rule,
so a trained model is tied to its grammar -- the trade-off discussed in the
survey.
