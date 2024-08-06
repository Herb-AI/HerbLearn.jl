module HerbLearn

using PyCall
using Random

using Serialization

using JLD
using ProgressBars

const global torch = pyimport("torch")
const global np = pyimport("numpy")
const global nn = pyimport("torch.nn")
const global metrics = pyimport("sklearn.metrics")

function __init__()
  copy!(torch, pyimport("torch"))
  copy!(np, pyimport("numpy"))
  copy!(nn, pyimport("torch.nn"))
  copy!(metrics, pyimport("sklearn.metrics"))
  
  # reload Julia models during run-time
  include(string(@__DIR__, "/models.jl"))
end

using HerbCore
using HerbConstraints
using HerbGrammar
using HerbInterpret
using HerbSearch
using HerbSpecification
using HerbBenchmarks: ProblemGrammarPair

include("data_generation.jl")
include("data_representation.jl")
include("models.jl")
include("program_representation.jl")
include("program_representation_utils.jl")
include("data_loaders.jl")

include("learn.jl")
include("utils.jl")

export 
  torch,

  generate_data,
  pretrain_heuristic,

  GeneratedProblem,
  ProblemGrammarPair,

  AbstractIOEncoder,
  AbstractStarCoderIOEncoder,
  DeepCoderIOEncoder,
  StarEnCoderIOEncoder,
  StarCoderIOEncoder, 
  StarCoder2IOEncoder,
  PropertySignatureIOEncoder,
  encode_IO_examples,

  AbstractProgramEncoder,
  ZeroProgramEncoder,
  GraphProgramEncoder,
  AbstractStarCoderProgramEncoder,
  StarCoderProgramEncoder,
  StarEnCoderProgramEncoder,
  StarCoder2ProgramEncoder,
  encode_programs,
  encode_grammar,
  embed_programs,

  AbstractProgramDecoder,
  decode_programs,

  DerivationPredNet,
  SemanticDerivationPredNet,
  NonNNSemanticDerivationPredNet,
  MLP,

  input_rules
end #module HerbLearn