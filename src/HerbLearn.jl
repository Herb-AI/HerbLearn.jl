module HerbLearn

using PyCall, PyCallUtils
using Random

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
using HerbGrammar
using HerbInterpret
using HerbSearch
using HerbSpecification

include("data_generation.jl")
include("data_representation.jl")
include("program_representation.jl")
include("program_representation_utils.jl")
include("data_loaders.jl")

include("learn.jl")
include("utils.jl")

export 
  torch,

  generate_data,
  pretrain_heuristic,

  AbstractIOEncoder,
  AbstractStarCoderIOEncoder,
  DeepCoderIOEncoder,
  StarEnCoderIOEncoder,
  StarCoderIOEncoder, 
  encode_IO_examples,

  AbstractProgramEncoder,
  ZeroProgramEncoder,
  GraphProgramEncoder,
  AbstractStarCoderProgramEncoder,
  StarCoderProgramEncoder,
  StarEnCoderProgramEncoder,
  encode_programs,
  embed_programs,

  AbstractProgramDecoder,
  decode_programs,

  DerivationPredNet,
  MLP
end #module HerbLearn