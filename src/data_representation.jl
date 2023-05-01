"""
Given a set of I/O examples, this returns an encoding of that. If it is also given a (partial) dictionary, then examples are represented with respect to that dict and the function returns the updated dict.
"""
function encode_IO_examples(io_examples::Vector{HerbData.IOExample}, emb_method::AbstractString="onehot", dict_partial=Dict{Any, Any})

    dict = Dict()
    if dict_partial !== nothing
        dict = deepcopy(dict_partial)
    end

    encoded = []
    if emb_method == "onehot"

    else
        # encode with another embedding method (not yet implemented)
        error("Unknown embedding method: $emb_method")
    end
    return encoded, dict
end

"""

"""
function ()
    
end

"""
DeepCoder-based (Balog et al., 2017) one-hot encoding of input-output examples. 

Description from their appendix:
1. Pad arrays appearing in the inputs and in the output to a maximum length L = 20 with a special NULL value.
2. Represent the type (singleton integer or integer array) of each input and of the output using a one-hot-encoding vector. Embed each integer in the valid integer range (−256 to 255) using a learned embedding into E = 20 dimensional space. Also learn an embedding for the padding NULL value.
3. Concatenate the representations of the input types, the embeddings of integers in the inputs, the representation of the output type, and the embeddings of integers in the output into a single (fixed-length) vector.
"""
function deepcoder_onehot_encoding(io_examples::vector{HerbData.IOExample}, MAX_LEN=20, EMBEDG_DIM=20)
    # Define constants
    EMB_NULL = torch.randn(embed_dim)  # @TODO Randomly pick NULL embedding?

    # Initialize variables
    emb_dict = Dict{Any, torch.Tensor}()
    emb_dict[0] = EMB_NULL
    inputs = torch.Tensor(length(io_examples), MAX_LEN, EMBED_DIM)
    input_types = torch.Tensor(length(io_examples), 2)
    outputs = torch.Tensor(length(io_examples), MAX_LEN, EMBED_DIM)
    output_type = torch.Tensor(length(io_examples), 2) 

    # Pad arrays and encode inputs and outputs
    for (i, ex) in enumerate(io_examples)
        # Pad inputs
        input_vals = ex.in[:x]
        input_type = typeof(input_vals)
        input_types[i, :] = torch.Tensor([input_type == Int, input_type == Array{Int, 1}])  # @TODO generalize this to general symbols
        input_emb = torch.cat([emb_dict[val] for val in input_vals], dim=0)
        pad_input_emb = torch.cat([input_emb, EMB_NULL.repeat(MAX_LEN - length(input_vals), EMBED_DIM)], dim=0)
        inputs[i, :, :] = pad_input_emb
        
        # Pad output
        output_val = ex.out
        output_type[i, :] = torch.Tensor([typeof(output_val) == Int, typeof(output_val) == Array{Int, 1}])  # @TODO see above
        output_emb = torch.cat([emb_dict[val] for val in output_val], dim=0)
        pad_output_emb = torch.cat([output_emb, EMB_NULL.repeat(MAX_LEN - length(output_val), 1)], dim=0) 
        outputs[i, :, :] = pad_output_emb
        
        # Update embeddings dictionary
        for val in input_vals
            if !(val in keys(emb_dict))
                emb_dict[val] = torch.randn(EMBED_DIM)
            end
        end
        for val in output_val
            if !(val in keys(emb_dict))
                emb_dict[val] = torch.randn(EMBED_DIM)
            end
        end
    end

    return inputs, input_types, outputs, output_type, emb_dict
end

"""
Given a list of programs or expressions, this function returns a one-hot encoding of these programs. If given a dict, then the programs will be represented with respect to that dict and the function returns the updated dict while adding padding to the vectors.
"""
function encode_onehot_programs()

end

"""
Given input-program-output triples, this function returns a one-hot encoding of these triples, using functionality from `encode_onehot_IO_examples` and `encode_onehot_programs`.
"""
function encode_onehot_IOP_examples(iop_examples::Vector{Tuple{HerbData.IOExample, Any}}, index_dict=Dict{Any, Int})
    
end
