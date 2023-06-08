# HerbLearn.jl
Machine Learning module of Herb

This module takes care of preprocessing, i.e., representation of data, training and prediction of models.

## Repository structure:
This module is separated in multiple sub-module, that take care of
1. data representation in `data_representation.jl`,
2. defining and building neural network models in `models.jl`, 
3. learning, training, and evaluation in `learn.jl`, and
4. data loading and other useful things in `utils.jl`

## How to define a neural network (NN) in Julia using PyTorch:

```Julia
@pydef mutable struct Model <: nn.Module
    function __init__(self, ...)
        pybuiltin(:super)(Model, self).__init__()
        self.f = ...
    end
    function forward(self, x)
      self.f(x)
    end
end
```

## Comments:
The combination of `PyCall` and configuring the correct conda environment is a bit tricky/non-trivial for now. Julia can handle Conda environments automatically and we will leverage this.
