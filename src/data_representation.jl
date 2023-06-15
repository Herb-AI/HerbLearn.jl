abstract type IOEmbedder end

"""
Given a set of I/O examples, this returns an encoding of that. If it is also given a (partial) dictionary, then examples are represented with respect to that dict and the function returns the updated dict.
"""
function encode_IO_examples(io_examples::Vector{HerbData.IOExample}, embedder::IOEmbedder, dict_partial=Dict{Any, Any}, MAX_LEN=20, EMBED_DIM=20)
    dict = Dict()
    if !isempty(dict_partial)
        dict = deepcopy(dict_partial)
    end

    embeddings, emb_dict = embedder.encode(io_examples, MAX_LEN, EMBED_DIM)
    return embeddings, merge(dict_partial, emb_dict)
end

"""

"""
function encode_IO_examples(io_examples::Vector{HerbData.IOExample}, embedder::StarspaceIOEmbedder, model_path::AbstractString, dict_partial=Dict{Any, Any}, MAX_LEN=20, EMBED_DIM=20)
    dict = Dict()
    if !isempty(dict_partial)
        dict = deepcopy(dict_partial)
    end

    embeddings, emb_dict = embedder.encode(io_examples, model_path, MAX_LEN, EMBED_DIM)
    return embeddings, merge(dict_partial, emb_dict)
end


"""
DeepCoder-based (Balog et al., 2017) one-hot encoding of input-output examples. 

Description from their appendix:
1. Pad arrays appearing in the inputs and in the output to a maximum length L = 20 with a special NULL value.
2. Represent the type (singleton integer or integer array) of each input and of the output using a one-hot-encoding vector. Embed each integer in the valid integer range (−256 to 255) using a learned embedding into E = 20 dimensional space. Also learn an embedding for the padding NULL value.
3. Concatenate the representations of the input types, the embeddings of integers in the inputs, the representation of the output type, and the embeddings of integers in the output into a single (fixed-length) vector.

@TODO this is a bit misleading. The (learned) embedding is not really done here, but the encoding.
@TODO update type checking dynamically
"""
function deepcoder_IO_encoding(io_examples::Vector{HerbData.IOExample}, MAX_LEN=20, EMBED_DIM=20)
    println("length: ", length(io_examples))
    # Define constants
    EMB_NULL = torch.rand(EMBED_DIM)  # @TODO Randomly pick NULL embedding?

    # Initialize variables
    emb_dict = Dict{Any, PyObject}()
    emb_dict[0] = EMB_NULL
    inputs = torch.Tensor(length(io_examples), MAX_LEN, EMBED_DIM)
    input_types = torch.Tensor(length(io_examples), 2)
    outputs = torch.Tensor(length(io_examples), MAX_LEN, EMBED_DIM)
    output_type = torch.Tensor(length(io_examples), 2) 

    # Pad arrays and encode inputs and outputs
    for (i, ex) ∈ enumerate(io_examples)
        # Pad inputs @TODO update inputs may be both vectors and 0-dim arrays
        input_vals = [ex.in[:x]]
        output_vals = [ex.out]

        input_type = typeof(input_vals)
        input_types[i-1, :] = torch.Tensor([input_type == Int, input_type == Array{Int, 1}])  # @TODO generalize this to general symbols

        for val ∈ [input_vals; output_vals]
            if val ∉ keys(emb_dict)
                emb_dict[val] = torch.rand(EMBED_DIM)
            end
        end

        input_emb = torch.cat([emb_dict[val] for val ∈ input_vals], dim=0)
        pad_input_emb = torch.cat([input_emb.view(1,-1),EMB_NULL.view(1,-1).repeat(MAX_LEN - length(input_vals),1)], dim=0)
        inputs[i-1, :, :] = pad_input_emb
        
        # Pad output
        output_type[i-1, :] = torch.Tensor([typeof(output_vals) == Int, typeof(output_vals) == Array{Int, 1}])  # @TODO see above
        output_emb = torch.cat([emb_dict[val] for val ∈ output_vals], dim=0)
        pad_output_emb = torch.cat([output_emb.view(1,-1), EMB_NULL.view(1,-1).repeat(MAX_LEN - length(output_vals),1)], dim=0) 
        outputs[i-1, :, :] = pad_output_emb
    end

    return torch.concat([inputs, outputs], dim=1), emb_dict  # TODO types are not used yet
end

mutable struct DeepCoderIOEmbedder <: IOEmbedder
    encode::Function = deepcoder_IO_encoding
end

"""

"""
function starspace_IO_encoding(io_examples::Vector{HerbData.IOExample}, model_filepath="", MAX_LEN=20, EMBED_DIM=20)
    # Write io_examples to file


    # Run starspace model on embeddings

    # Read predictions from tsv file
    
    # Put predictions into torch tensors

    # Pickle torch tensors to file
    error("Starspace encoding not yet implemented.") 
end

mutable struct StarspaceIOEmbedder <: IOEmbedder
    encode::Function = starspace_IO_encoding
end