abstract type IOEncoder end

mutable struct StarCoderIOEncoder <: IOEncoder
    tokenizer
    encoder
    EMBED_DIM::Int

    function StarCoderIOEncoder()
        transformers = pyimport("transformers")
        device = torch.device(ifelse(nocuda && torch.cuda.is_available(), "cuda", "cpu"))

        checkpoint = "bigcode/starencoder"
        tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint).to(device)
        encoder = transformers.AutoModelForPreTraining.from_pretrained(checkpoint).to(device)

        if isnothing(tokenizer.pad_token)
            tokenizer.add_special_tokens(Dict("pad_token" => "[PAD]"))
            encoder.resize_token_embeddings(length(tokenizer))
        end

        embed_dim = encoder.config.hidden_size

        new(tokenizer, encoder, embed_dim)
    end
end

mutable struct DeepCoderIOEncoder <: IOEncoder
    MAX_LEN::Int
    EMBED_DIM::Int
    EMB_NULL::PyObject

    function DeepCoderIOEncoder(MAX_LEN::Int, EMBED_DIM::Int)
        EMB_NULL = torch.rand(EMBED_DIM)
        new(MAX_LEN, EMBED_DIM, EMB_NULL)
    end
end


"""
Given a set of I/O examples, this returns an encoding of that. If it is also given a (partial) dictionary, then examples are represented with respect to that dict and the function returns the updated dict.
"""
function encode_IO_examples(io_examples::Vector{HerbData.IOExample}, encoder::DeepCoderIOEncoder)
    # Initialize variables
    MAX_LEN = encoder.MAX_LEN
    EMBED_DIM = encoder.EMBED_DIM
    EMB_NULL = encoder.EMB_NULL
    emb_dict = encoder.emb_dict

    emb_dict[0] = EMB_NULL

    inputs = torch.Tensor(length(io_examples), MAX_LEN, EMBED_DIM)
    input_types = torch.Tensor(length(io_examples), 2)
    outputs = torch.Tensor(length(io_examples), MAX_LEN, EMBED_DIM)
    output_type = torch.Tensor(length(io_examples), 2) 

    # Pad arrays and encode inputs and outputs
    for (i, ex) ∈ enumerate(io_examples)
        # Pad inputs
        input_vals = [ex.in[:x]]
        output_vals = [ex.out]

        input_type = typeof(input_vals)
        input_types[i, :] = torch.Tensor([input_type == Int, input_type == Array{Int, 1}])

        for val ∈ [input_vals; output_vals]
            if val ∉ keys(emb_dict)
                emb_dict[val] = torch.rand(EMBED_DIM)
            end
        end

        input_emb = torch.cat([emb_dict[val] for val ∈ input_vals], dim=0)
        pad_input_emb = torch.cat([input_emb.view(1,-1), EMB_NULL.view(1,-1).repeat(MAX_LEN - length(input_vals), 1)], dim=0)
        inputs[i-1, :, :] = pad_input_emb

        # Pad output
        output_type[i-1, :] = torch.Tensor([typeof(output_vals) == Int, typeof(output_vals) == Array{Int, 1}])
        output_emb = torch.cat([emb_dict[val] for val ∈ output_vals], dim=0)
        pad_output_emb = torch.cat([output_emb.view(1,-1), EMB_NULL.view(1,-1).repeat(MAX_LEN - length(output_vals), 1)], dim=0) 
        outputs[i-1, :, :] = pad_output_emb
    end

    return torch.cat([inputs, input_types, outputs, output_type], dim=2)
end


"""
    encode_IO_examples(io_examples::Vector{HerbData.IOExample}, encoder::StarCoderIOEncoder)

Encode input/output examples using the provided StarCoderIOEncoder.

# Arguments
- `io_examples::Vector{HerbData.IOExample}`: A vector of input/output examples to be encoded.
- `encoder::StarCoderIOEncoder`: The encoder to be used for encoding the examples.
- `dict_partial::Dict{Any, Any}`: A dictionary containing any additional information needed for encoding. Defaults to an empty dictionary.

# Returns
- `embeddings`: The encoded input/output examples.

# Examples
```julia
io_examples = [HerbData.IOExample(input1, output1), HerbData.IOExample(input2, output2)]
encoder = StarCoderIOEncoder()
encoded_examples = encode_IO_examples(io_examples, encoder)
```

"""
function encode_IO_examples(io_examples::Vector{HerbData.IOExample}, encoder::StarCoderIOEncoder)
    device = torch.device(ifelse(nocuda && torch.cuda.is_available(), "cuda", "cpu"))

    # Use string representation of Dict
    input_strings = [string(example.in)[findfirst(isequal('('), string(example.in))+1:end-1] for example in io_examples]
    output_strings = [string(example.out) for example in io_examples]

    # tokenize string represenations
    encoded_inputs = encoder.tokenizer(input_strings, padding=true, truncation=true, return_tensors="pt").to(device)
    encoded_outputs = encoder.tokenizer(output_strings, padding=true, truncation=true, return_tensors="pt").to(device)

    # encode examples using encoder model
    encoded_inputs = encoder.encoder(input_ids=encoded_inputs["input_ids"], attention_mask=encoded_inputs["attention_mask"])
    encoded_outputs = encoder.encoder(input_ids=encoded_outputs["input_ids"], attention_mask=encoded_outputs["attention_mask"])

    # take first hidden state of the transformer
    input_matrix = encoded_inputs["hidden_states"][0][:, 0, :].view(length(io_examples), encoder.EMBED_DIM)
    output_matrix = encoded_outputs["hidden_states"][0][:, 0, :].view(length(io_examples), encoder.EMBED_DIM)

    # concatenate embeddings to final output matrix
    embeddings = torch.cat([input_matrix, output_matrix], dim=1)

    return embeddings
end
