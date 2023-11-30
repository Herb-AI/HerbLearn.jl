"""
A simple model with one hidden layer.
"""
@pydef mutable struct SimpleModel <: torch.nn.Module
    function __init__(self, num_features, num_classes)
        pybuiltin(:super)(SimpleModel, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes

        lat_dim = 128

        self.linear1 = torch.nn.Linear(self.num_features, lat_dim)
        self.linear2 = torch.nn.Linear(lat_dim, self.num_classes)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
    end

    function forward(self, x)
        x = self.relu(self.linear1(x))                
        x = self.softmax(self.linear2(x))
        return x
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
        self.enc_linear1 = torch.nn.Linear(self.num_features, hid_layer_dim)
        self.enc_linear2 = torch.nn.Linear(hid_layer_dim, self.lat_dim)

        # Decoding module:
        self.dec_linear1 = torch.nn.Linear(self.lat_dim, hid_layer_dim)
        self.dec_linear2 = torch.nn.Linear(hid_layer_dim, self.num_features)

        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
    end

    function encode(self, x)
        x = self.relu(self.enc_linear1(x))                
        x = self.relu(self.enc_linear2(x))
        return x

    end

    function decode(self, x)
        x = self.relu(self.dec_linear1(x))                
        x = self.dec_linear2(x).sigmoid()
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
    function __init__(self, num_IO_features, num_prog_features, num_derivations)
        println("num_IO_features: $num_IO_features\nnum_prog_features: $num_prog_features\nnum_derivations: $num_derivations")
        pybuiltin(:super)(DerivationPredNet, self).__init__()
        self.num_IO_features = num_IO_features
        self.num_prog_features = num_prog_features
        self.num_derivations = num_derivations

        lat_dim = 32

        # Init IO encoder
        self.IO_linear1 = torch.nn.Linear(self.num_IO_features, lat_dim)
        self.IO_linear2 = torch.nn.Linear(lat_dim, lat_dim)

        # Init prog encoder
        self.prog_linear1 = torch.nn.Linear(self.num_prog_features, lat_dim)
        self.prog_linear2 = torch.nn.Linear(lat_dim, lat_dim)

        # Init joint learnin + pred
        self.joint_linear1 = torch.nn.Linear(lat_dim*2, lat_dim)
        self.joint_linear2 = torch.nn.Linear(lat_dim, num_derivations)

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
    end

    function forward(self, IO_X, prog_X)
        batch_size = IO_X.shape[1]
        IO_X = self.relu(self.IO_linear1(IO_X.view(batch_size,-1))) 
        IO_X = self.relu(self.IO_linear2(IO_X))                

        prog_X = self.relu(self.prog_linear1(prog_X))                
        prog_X = self.relu(self.prog_linear2(prog_X))                

        x = torch.cat([IO_X, prog_X], dim=1)
        x = self.relu(self.joint_linear1(x))                
        x = self.joint_linear2(x)
        return x.sigmoid()
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
        self.dropout = dropout ? torch.nn.Dropout(dropout) : nothing
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
@TODO add support for graph representation techniques by implementing a naive autoencoder over ASTs
"""
