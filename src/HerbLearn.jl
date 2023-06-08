module HerbLearn

using PyCall, PyCallUtils
using Random

const global torch = pyimport("torch")
const global np = pyimport("numpy")
const global nn = pyimport("torch.nn")

const models = PyNULL()

function __init__()
  copy!(torch, pyimport("torch"))
  copy!(np, pyimport("numpy"))
  copy!(nn, pyimport("torch.nn"))
  
  scriptdir = @__DIR__
  include(string(scriptdir, "/models.jl"))
end


# const F = pyimport("torch.nn.functional")

using ..HerbData
using ..HerbGrammar
using ..HerbEvaluation

include("data_generation.jl")
include("data_representation.jl")
include("models.jl")
include("learn.jl")
include("utils.jl")

export 
  generate_data,
  pretrain,
  torch
end #module HerbLearn