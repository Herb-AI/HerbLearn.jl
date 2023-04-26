module HerbLearn

using ..HerbDataGeneration
using PyCall
const torch = pyimport("torch")

include("data_representation.jl")

export

end #module HerbLearn