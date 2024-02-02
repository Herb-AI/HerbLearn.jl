
"""

"""
@pydef mutable struct SemanticDerivationPredNet <: torch.nn.Module
    function __init__(self, num_IO_features::Int, num_prog_features::Int, grammar_embeddings, hidden_layer_sizes::Vector{Int}=[])
        println("num_IO_features: $num_IO_features\nnum_prog_features: $num_prog_features\ngrammar_embeddings: $(grammar_embeddings.shape)")

        assert(typeof(grammar_embeddings.shape) == Tuple{Int, Int})

        pybuiltin(:super)(SemanticDerivationPredNet, self).__init__()
        self.num_IO_features = num_IO_features
        self.num_prog_features = num_prog_features
        self.grammar_embeddings = grammar_embeddings

        self.sizes = [[num_IO_features + num_prog_features]; hidden_layer_sizes; grammar_embeddings.shape[2]]
        self.MLP = MLP(self.sizes)
    end

    function forward(self, io_x, prog_x)
        batch_size = io_x.shape[1]
        x = torch.cat([io_x.view(batch_size, -1), prog_x])

        x = self.MLP(x)

        # calculate cosine similarity
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        grammar_embeddings = torch.nn.functional.normalize(self.grammar_embeddings, p=2, dim=1)

        x = torch.matmul(x, grammar_embeddings.T)
        
        # return x
        return x.sigmoid()
        # return x.softmax(dim=1)
   end
end

"""

"""
@pydef mutable struct Residual <: torch.nn.Module
    function __init__(self, fn)
        pybuiltin(:super)(Residual, self).__init__()
        self.fn = fn
    end

    function forward(self, x)
        return x + self.fn(x)
    end
end


"""

"""
@pydef mutable struct MLPBlock <: torch.nn.Module
    function __init__(self, in_features, out_features, bias=true, layer_norm=true, dropout=0.3, activation=torch.nn.ReLU)
        pybuiltin(:super)(MLPBlock, self).__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias)
        self.activation = activation()
        self.layer_norm = layer_norm ? torch.nn.LayerNorm(out_features) : nothing
        self.dropout = !isnothing(dropout) ? torch.nn.Dropout(dropout) : nothing
    end

    function forward(self, x)
        x = self.activation(self.linear(x))
        if !isnothing(self.layer_norm)
            x = self.layer_norm(x)
        end
        if !isnothing(self.dropout)
            x = self.dropout(x)
        end
        return x
    end
end

"""

"""
@pydef mutable struct MLP <: torch.nn.Module
    function __init__(self, sizes::Array{Int}, bias=true, layer_norm=true, dropout=0.1, activation=torch.nn.ReLU)
        pybuiltin(:super)(MLP, self).__init__()

        self.layers = torch.nn.ModuleList()  # Use ModuleList to hold submodules
        for i in 1:length(sizes)-1
            in_features = sizes[i]
            out_features = sizes[i+1]
            if i == length(sizes)-1
                block = MLPBlock(in_features, out_features, false, false, 0.0, torch.nn.Identity)
            else 
                block = MLPBlock(in_features, out_features, bias, layer_norm, dropout, activation)
            end
            self.layers.append(block)
        end
    end

    function forward(self, x)
        for layer in self.layers
            x = layer(x)
        end
        return x
    end
end

"""
A simple model with one hidden layer.
"""
@pydef mutable struct SimpleModel <: torch.nn.Module
    function __init__(self, num_features, num_classes)
        pybuiltin(:super)(SimpleModel, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes

        lat_dim = 128
        self.MLP = MLP([self.num_features, lat_dim, self.num_classes])
    end

    function forward(self, x)
        x = self.MLP(x)
        return x.softmax(dim=1)
    end
end

"""
A simple auto-encoder neural network model
"""
@pydef mutable struct AutoEncoderModel <: torch.nn.Module
    function __init__(self, num_features, lat_dim)
        pybuiltin(:super)(AutoEncoderModel, self).__init__()
        self.num_features = num_features
        self.lat_dim = lat_dim

        hid_layer_dim = 128
        # Encoding module:
        self.encoder = MLP([self.num_features, hid_layer_dim, self.lat_dim])

        # Decoding module:
        self.decoder = MLP([self.lat_dim, hid_layer_dim, self.num_features])
    end

    function encode(self, x)
        x = self.encoder(x)
        return x

    end

    function decode(self, x)
        x = self.decoder(x).tanh()
        return x

    end

    function forward(self, x)
        x = self.encode(x)
        x = self.decode(x)
        return x
    end
end

"""

"""
@pydef mutable struct DerivationPredNet <: torch.nn.Module
    function __init__(self, num_IO_features::Int, num_prog_features::Int, num_derivations::Int, hidden_layer_sizes::Vector{Int}=[])
        pybuiltin(:super)(DerivationPredNet, self).__init__()
        println("num_IO_features: $num_IO_features\nnum_prog_features: $num_prog_features\nnum_derivations: $num_derivations")

        self.num_IO_features = num_IO_features
        self.num_prog_features = num_prog_features
        self.num_derivations = num_derivations

        self.sizes = [[num_IO_features + num_prog_features]; hidden_layer_sizes; num_derivations]
        self.MLP = MLP(self.sizes)
    end

    function forward(self, io_x, prog_x)
        batch_size = io_x.shape[1]
        x = torch.cat([io_x.view(batch_size, -1), prog_x], dim=1)

        x = self.MLP(x)

        # return x.sigmoid()
        return x.softmax(dim=1)
   end
end


"""
@TODO add support for graph representation techniques by implementing a naive autoencoder over ASTs
"""

