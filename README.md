# HerbLearn.jl
Machine Learning module of Herb

This module takes care of preprocessing, i.e., representation of data, training and prediction of models.

## Repository structure:
This module is separated in multiple sub-module, that take care of
- generating input/output example + programs in `data_generation.jl`
- input/output representation in `data_representation.jl`,
- program representation in `program_representation.jl`,
- defining and building neural network models in `models.jl`, 
- learning, training, and evaluation in `learn.jl`, and
- data loading and other useful things in `utils.jl`

## Using PyCall
`PyCall` is a package to communicate with Python code from inside Julia that we use to build our PyTorch modules. Hence, make sure to configure the desired Python binary path as follows from the project's Julia REPL:

```
using Pkg

# On Unix:
ENV["PYTHON"]="/home/d-oughnut/anaconda3/envs/prog_synth/bin/python"
# And on Windows:
ENV["PYTHON"] = raw"C:\Users\DOughnut\AppData\Local\Programs\Python\Python37-32\python.exe"

Pkg.build(PyCall)
```

We recommend using Conda environments.

### Required python packages:
In order to use `HerbLearn` please install the following packages:
```
- torch
- numpy
```

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

