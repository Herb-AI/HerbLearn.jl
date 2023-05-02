"""
A simple model for learning grammar derivation heuristics based on I/O examples
"""
@pydef struct SimpleModel <: torch.nn.Module
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
@pydef struct AutoEncoderModel <: torch.nn.Module
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
@pydef struct Residual <: torch.nn.Module
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
@pydef struct MLPBlock <: torch.nn.Module
    function __init__(self, in_features, out_features, bias=True, layer_norm=True, dropout=0.3, activation=torch.nn.ReLU)
        pybuiltin(:super)(MLPBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.activation = activation()
        self.layer_norm = layer_norm ? torch.nn.LayerNorm(out_features) : nothing
        self.dropout = dropout ? torch.nn.Dropout(dropout) : nothing
    end

    function forward(self, x)
        x = self.activation(self.linear(x))
        if self.layer_norm
            x = self.layer_norm(x)
        end
        if self.dropout
            x = self.dropout(x)
        end
        return x
    end
end



"""
@TODO add support for graph representation techniques by implementing a naive autoencoder over ASTs
"""
