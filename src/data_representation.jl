abstract type IOEncoder end

mutable struct StarCoderIOEncoder <: IOEncoder
    EMBED_DIM::Int
    data_dir::AbstractString
    pretrain::Bool
end

mutable struct DeepCoderIOEncoder <: IOEncoder
    MAX_LEN::Int
    EMBED_DIM::Int
end


"""
Given a set of I/O examples, this returns an encoding of that. If it is also given a (partial) dictionary, then examples are represented with respect to that dict and the function returns the updated dict.
"""
function encode_IO_examples(io_examples::Vector{HerbData.IOExample}, embedder::DeepCoderIOEncoder, dict_partial=Dict{Any, Any})
    dict = Dict()
    if !isempty(dict_partial)
        dict = deepcopy(dict_partial)
    end

    embeddings = deepcoder_IO_encoding(io_examples, embedder.MAX_LEN, embedder.EMBED_DIM)
    return embeddings
end


"""

"""
function encode_IO_examples(io_examples::Vector{HerbData.IOExample}, embedder::StarspaceIOEncoder, dict_partial=Dict{Any, Any})
    dict = Dict()
    if !isempty(dict_partial)
        dict = deepcopy(dict_partial)
    end

    embeddings = starspace_IO_encoding(io_examples, embedder.data_dir, embedder.pretrain, 20, 20)
        
    return embeddings
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

    return torch.concat([inputs, outputs], dim=1)
end


"""

"""
function starcoder_IO_encoding(io_examples::Vector{HerbData.IOExample}, checkpoint="bigcode/starencoder")
    device = torch.device(ifelse(nocuda && torch.cuda.is_available(), "cuda", "cpu"))

    # Access the necessary classes from transformers
    transformers = pyimport("transformers")
    AutoTokenizer = transformers.AutoTokenizer

    # Define the checkpoint and device
    checkpoint = "bigcode/starencoder"  # gpt-2 based foundation model
    
    # Load the pretrained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    encoder = transformers.AutoModelForPreTraining.from_pretrained("bigcode/starencoder")

    
    # Check if the pad_token is None, and if so, add a special pad token and resize the token embeddings
    if isnothing(tokenizer.pad_token)
        tokenizer.add_special_tokens(Dict("pad_token" => "[PAD]"))
        encoder.resize_token_embeddings(length(tokenizer))
    end

    # Use string representation of Dict
    input_strings = [string(example.in)[findfirst(isequal('('), string(example.in))+1:end-1] for example in io_examples]
    output_strings = [string(example.out) for example in io_examples]

    # tokenize 
    encoded_inputs = tokenizer(input_strings, padding=True, truncation=True, return_tensors="pt")
    encoded_outputs = tokenizer(output_strings, padding=True, truncation=True, return_tensors="pt")

    encoded_inputs = encoder(input_ids=encoded_inputs.input_ids, attention_mask=encoded_inputs.attention_mask)
    encoded_outputs = encoder(input_ids=encoded_outputs.input_ids, attention_mask=encoded_outputs.attention_mask)

    input_matrix = encoded_inputs.hidden_states[0][:,0,:].view(length(io_examples), -1)
    output_matrix = encoded_outputs.hidden_states[0][:,0,:].view(length(io_examples), -1)

    return torch.concat([input_matrix, output_matrix], dim=1)
end