module HerbLearn

using PyCall, PyCallUtils
using Random

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
using HerbData
using HerbGrammar
using HerbEvaluation

include("data_generation.jl")
include("data_representation.jl")
include("program_representation.jl")

include("learn.jl")
include("utils.jl")

export 
  torch,

  generate_data,
  pretrain,

  IOEncoder,
  StarspaceIOEncoder, 
  DeepCoderIOEncoder,

  ProgramEncoder,
  ZeroProgramEncoder,
  GraphProgramEncoder,
  TransformerProgramEncoder
end #module HerbLearn