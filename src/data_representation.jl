abstract type AbstractIOEncoder end

mutable struct DeepCoderIOEncoder <: AbstractIOEncoder
    MAX_LEN::Int
    EMBED_DIM::Int
    EMB_NULL::PyObject
    emb_dict::Dict

    function DeepCoderIOEncoder(MAX_LEN::Int, EMBED_DIM::Int)
        EMB_NULL = torch.rand(EMBED_DIM)
        new(MAX_LEN, EMBED_DIM, EMB_NULL, Dict{Any, PyObject}())
    end
end

abstract type AbstractStarCoderIOEncoder <: AbstractIOEncoder end

mutable struct StarEnCoderIOEncoder <: AbstractStarCoderIOEncoder
    tokenizer
    encoder
    EMBED_DIM::Int
    batch_size::Int
    max_sequence_length::Int

    function StarEnCoderIOEncoder(batch_size::Int=128, max_sequence_length::Int=256, hidden_state::Int=1)
        transformers = pyimport("transformers")
        device = torch.device(ifelse(torch.cuda.is_available(), "cuda", "cpu"))

        checkpoint = "bigcode/starencoder"
        tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint)
        encoder = transformers.AutoModelForPreTraining.from_pretrained(checkpoint)

        # Prune the model
        if hasproperty(encoder, :bert) && hasproperty(encoder.bert, :encoder) && hasproperty(encoder.bert.encoder, :layer)
            encoder.bert.encoder.layer = torch.nn.ModuleList([layer for (i, layer) in enumerate(encoder.bert.encoder.layer) if i<hidden_state]) # in range 1 to 13
        else
            throw(ArgumentError("Non-BERT model chosen for BERT-only embedding method."))
        end

        for param in encoder.parameters() 
            param.requires_grad = false
        end

        if isnothing(tokenizer.pad_token)
            tokenizer.add_special_tokens(Dict("pad_token" => "[PAD]"))
            encoder.resize_token_embeddings(length(tokenizer))
        end

        embed_dim = encoder.config.hidden_size * max_sequence_length

        encoder = torch.nn.DataParallel(encoder).to(device)

        new(tokenizer, encoder, embed_dim, batch_size, max_sequence_length)
    end
end


"""

Make sure that `model_path` ends with a '/'. 
"""
mutable struct StarCoderIOEncoder <: AbstractStarCoderIOEncoder
    tokenizer
    encoder
    EMBED_DIM::Int
    batch_size::Int
    max_sequence_length::Int

    function StarCoderIOEncoder(batch_size::Int=128, max_sequence_length::Int=256, hidden_state::Int=1, model_path::AbstractString=nothing)
        transformers = pyimport("transformers")

        checkpoint = "bigcode/starcoder"
        tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint)

        if isnothing(model_path)
            encoder = transformers.AutoModel.from_pretrained(checkpoint, device_map = "auto")  # load to GPUs
        elseif endswith(model_path, '/') && !isdir(model_path)
            # Load model if not existent
            encoder = transformers.AutoModel.from_pretrained(checkpoint, device_map = "auto") 
            # Prune model to hidden state
            encoder.h = torch.nn.ModuleList([layer for (i, layer) in enumerate(encoder.h) if i<hidden_state]) # in range 1 to 40
            encoder.config.n_layer = hidden_state

        elseif isdir(model_path)
            # Load encoder from disk
            encoder = transformers.AutoModel.from_pretrained(model_path, device_map = "auto")  # load to gpu from disk
        else
            throw(ArgumentError("Invalid model path selected. Make sure to select a directory."))
        end

        for param in encoder.parameters() 
            param.requires_grad = false
        end

        if !isnothing(model_path) && !isdir(model_path) 
            # Save model to disk
            encoder.save_pretrained(model_path)
        end

        # >>> output_2["last_hidden_state"].size()
        # torch.Size([1, 15, 6144])

        if isnothing(tokenizer.pad_token)
            tokenizer.add_special_tokens(Dict("pad_token" => "[PAD]"))
            encoder.resize_token_embeddings(length(tokenizer))
        end

        embed_dim = encoder.config.hidden_size * max_sequence_length
        new(tokenizer, encoder, embed_dim, batch_size, max_sequence_length)
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

    types = [Int, Array{Int, 1}, String]

    encoder.emb_dict[0] = EMB_NULL

    inputs = torch.Tensor(length(io_examples), MAX_LEN, EMBED_DIM)
    input_types = torch.Tensor(length(io_examples), length(types))
    outputs = torch.Tensor(length(io_examples), MAX_LEN, EMBED_DIM)
    output_type = torch.Tensor(length(io_examples), length(types)) 

    # Pad arrays and encode inputs and outputs
    for (i, ex) ∈ enumerate(io_examples)
        # Pad inputs
        input_vals = [val for (key, val) in ex.in]
        output_vals = [ex.out]

        input_types[i-1, :] = torch.Tensor([typeof(input_vals)==type for type in types])

        for val ∈ [input_vals; output_vals]
            if val ∉ keys(encoder.emb_dict)
                encoder.emb_dict[val] = torch.rand(EMBED_DIM)
            end
        end

        input_emb = torch.cat([encoder.emb_dict[val] for val ∈ input_vals], dim=0)
        pad_input_emb = torch.cat([input_emb.view(1,-1), EMB_NULL.view(1,-1).repeat(MAX_LEN - length(input_vals), 1)], dim=0)
        inputs[i-1, :, :] = pad_input_emb

        # Pad output
        output_type[i-1, :] = torch.Tensor([typeof(output_vals)==type for type in types])
        output_emb = torch.cat([encoder.emb_dict[val] for val ∈ output_vals], dim=0)
        pad_output_emb = torch.cat([output_emb.view(1,-1), EMB_NULL.view(1,-1).repeat(MAX_LEN - length(output_vals), 1)], dim=0) 
        outputs[i-1, :, :] = pad_output_emb
    end

    println(inputs.shape, input_types.shape, outputs.shape, output_type.shape)
    return torch.cat([vec.view(length(io_examples), -1) for vec in [inputs, input_types, outputs, output_type]], dim=1)
end


"""
    encode_IO_examples(io_examples::Vector{HerbData.IOExample}, encoder::AbstractStarCoderIOEncoder)

Encode input/output examples using the provided AbstractStarCoderIOEncoder.

# Arguments
- `io_examples::Vector{HerbData.IOExample}`: A vector of input/output examples to be encoded.
- `encoder::AbstractStarCoderIOEncoder`: The encoder to be used for encoding the examples.

# Returns
- `embeddings`: The encoded input/output examples.

# Examples
```julia
io_examples = [HerbData.IOExample(input1, output1), HerbData.IOExample(input2, output2)]
encoder = StarEnCoderIOEncoder()
encoded_examples = encode_IO_examples(io_examples, encoder)
```

The resulting embeddings will be stored and returned as `torch.cpu()`. 
"""
function encode_IO_examples(io_examples::Vector{HerbData.IOExample}, encoder::AbstractStarCoderIOEncoder)
    device = torch.device(ifelse(torch.cuda.is_available(), "cuda", "cpu"))

    num_batches = ceil(Int, length(io_examples) / encoder.batch_size)
    data_batches = [io_examples[(i-1)*encoder.batch_size+1:min(i*encoder.batch_size, end)] for i in 1:num_batches]

    return_matrix = []

    @pywith torch.no_grad() begin
        for batch in ProgressBar(data_batches)
            # Use string representation of Dict
            input_strings = [string(example.in)[findfirst(isequal('('), string(example.in))+1:end-1] for example in batch]
            output_strings = [string(example.out) for example in batch]

            # tokenize string represenations
            encoded_inputs = encoder.tokenizer(input_strings, padding="max_length", return_tensors="pt", max_length=encoder.max_sequence_length).to(device)
            encoded_outputs = encoder.tokenizer(output_strings, padding="max_length", return_tensors="pt", max_length=encoder.max_sequence_length).to(device)

            # encode examples using encoder model and take first hidden state of the transformer
            encoded_inputs = encoder.encoder(input_ids=encoded_inputs["input_ids"], attention_mask=encoded_inputs["attention_mask"])["hidden_states"][end].detach()
            encoded_outputs = encoder.encoder(input_ids=encoded_outputs["input_ids"], attention_mask=encoded_outputs["attention_mask"])["hidden_states"][end].detach()

            # concatenate embeddings to final output matrix
            embeddings = torch.cat([encoded_inputs, encoded_outputs], dim=1)

            push!(return_matrix, embeddings.cpu())
            GC.gc()
        end
    end
    max_length = maximum([tensor.size(1) for tensor in return_matrix])

    padded_tensors = []
    for tensor in return_matrix
        padding = max_length - tensor.size(1)
        # Pad the tensor. The padding configuration (0, 0, 0, padding) is for the last two dimensions.
        padded_tensor = torch.nn.functional.pad(tensor, (0, 0, 0, padding))
        push!(padded_tensors, padded_tensor.view(tensor.size(0), -1))
    end

    # Concatenate all padded tensors along the 0th dimension
    return torch.cat(padded_tensors, dim=0) 
end
