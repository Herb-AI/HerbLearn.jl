abstract type AbstractIOEncoder end

mutable struct DeepCoderIOEncoder <: AbstractIOEncoder
    MAX_LEN::Int
    VECTOR_DIM::Int
    EMBED_DIM::Int
    EMB_NULL::PyObject
    emb_dict::Dict

    types::Array{Any}

    function DeepCoderIOEncoder(MAX_LEN::Int, VECTOR_DIM::Int)
        EMB_NULL = torch.zeros(VECTOR_DIM)
        types = [Int, Array{Int, 1}, String]
        EMBED_DIM = 2 * (VECTOR_DIM * MAX_LEN + length(types))
        new(MAX_LEN, VECTOR_DIM, EMBED_DIM, EMB_NULL, Dict{Any, PyObject}(), types)
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

        embed_dim = encoder.config.hidden_size * 2

        # encoder = torch.nn.DataParallel(encoder).to(device)

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
            encoder = transformers.AutoModel.from_pretrained(checkpoint, device_map = "auto", load_in_8bit=true)  # load to GPUs
        elseif endswith(model_path, '/') && !isdir(model_path)
            # Load model if not existent
            encoder = transformers.AutoModel.from_pretrained(checkpoint, device_map = "auto", load_in_8bit=true) 
            # Prune model to hidden state
            encoder.h = torch.nn.ModuleList([layer for (i, layer) in enumerate(encoder.h) if i<hidden_state]) # in range 1 to 40
            encoder.config.n_layer = hidden_state

        elseif isdir(model_path)
            # Load encoder from disk
            encoder = transformers.AutoModel.from_pretrained(model_path, device_map = "auto", load_in_8bit=true)  # load to gpu from disk
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

        embed_dim = encoder.config.hidden_size * 2
        new(tokenizer, encoder, embed_dim, batch_size, max_sequence_length)
    end
end


function encode_IO_examples(gen_problems::Vector{GeneratedProblem}, encoder::AbstractIOEncoder)
    return_features = []
    for gen_problem in ProgressBar(gen_problems)
        push!(return_features, encode_IO_examples(gen_problem.io_examples, encoder))
    end
    
    return torch.cat(return_features, dim=0).view(length(gen_problems), -1)
end


"""
Given a set of I/O examples, this returns an encoding of that. If it is also given a (partial) dictionary, then examples are represented with respect to that dict and the function returns the updated dict.
"""
function encode_IO_examples(io_examples::Vector{HerbSpecification.IOExample}, encoder::DeepCoderIOEncoder)
    # Initialize variables
    VECTOR_DIM = encoder.VECTOR_DIM
    EMB_NULL = encoder.EMB_NULL

    types = encoder.types

    encoder.emb_dict[0] = EMB_NULL

    inputs = torch.zeros((length(io_examples), encoder.MAX_LEN, VECTOR_DIM), dtype=torch.bool)
    input_types = torch.zeros((length(io_examples), length(types)), dtype=torch.bool)
    outputs = torch.zeros((length(io_examples), encoder.MAX_LEN, VECTOR_DIM), dtype=torch.bool)
    output_type = torch.zeros((length(io_examples), length(types)), dtype=torch.bool)

    # Pad arrays and encode inputs and outputs
    for (i, ex) ∈ enumerate(io_examples)
        # Pad inputs
        input_vals = [val for (arg, arg_val) in ex.in for val in arg_val]
        output_vals = [val for val in ex.out]

        input_types[i-1, :] = torch.Tensor([typeof(input_vals)==type for type in types])

        for val ∈ [input_vals; output_vals]
            if val ∉ keys(encoder.emb_dict)
                encoder.emb_dict[val] = torch.rand(VECTOR_DIM)
            end
        end

        input_emb = torch.cat([encoder.emb_dict[val].view(1,-1) for val ∈ input_vals], dim=0)
        pad_input_emb = nothing
        if max(length(input_vals), length(output_vals)) > encoder.MAX_LEN
            new_MAX_LEN = max(length(input_vals), length(output_vals))
            pad_size = new_MAX_LEN - encoder.MAX_LEN
            padding = (0, 0, 0, pad_size) 
            encoder.MAX_LEN = new_MAX_LEN
            
            inputs = torch.nn.functional.pad(inputs, padding, "constant", 0)  # Add 0-padding
            outputs = torch.nn.functional.pad(outputs, padding, "constant", 0)  # Add 0-padding
        end
        pad_input_emb = torch.cat([input_emb, EMB_NULL.view(1,-1).repeat(encoder.MAX_LEN - length(input_vals), 1)], dim=0)
        inputs[i-1, :, :] = pad_input_emb

        # Pad output
        output_type[i-1, :] = torch.Tensor([typeof(output_vals)==type for type in types])
        # try 
        output_emb = torch.cat([encoder.emb_dict[val].view(1,-1) for val ∈ output_vals], dim=0)
        pad_output_emb = torch.cat([output_emb, EMB_NULL.view(1,-1).repeat(encoder.MAX_LEN - length(output_vals), 1)], dim=0) 
        outputs[i-1, :, :] = pad_output_emb
        """ 
        catch ex
            if ex isa PyCall.PyError
                continue
            else
                println("Unexpected error in encode_IO_examples")
                throw(error)
            end
        end
        """ 
    end

    result_mat = torch.cat([vec.view(length(io_examples), -1) for vec in [inputs, input_types, outputs, output_type]], dim=1)
    return result_mat.max(dim=0)
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
function encode_IO_examples(io_examples::Vector{HerbSpecification.IOExample}, encoder::AbstractStarCoderIOEncoder)
    device = torch.device(ifelse(torch.cuda.is_available(), "cuda", "cpu"))

    @pywith torch.no_grad() begin
        num_batches = ceil(Int, length(io_examples) / encoder.batch_size)
        data_batches = [io_examples[(i-1)*encoder.batch_size+1:min(i*encoder.batch_size, end)] for i in 1:num_batches]

        problem_io_encodings = []

        for batch in data_batches
            # Use string representation of Dict
            input_strings = [string(example.in)[findfirst(isequal('('), string(example.in))+1:end-1] for example in batch]
            output_strings = [string(example.out) for example in batch]

            # tokenize string represenations
            encoded_inputs = encoder.tokenizer(input_strings, padding=true, truncation=true, return_tensors="pt", max_length=encoder.max_sequence_length).to(device)
            encoded_outputs = encoder.tokenizer(output_strings,  padding=true, truncation=true,return_tensors="pt", max_length=encoder.max_sequence_length).to(device)

            # encode examples using encoder model and take first hidden state of the transformer
            encoded_inputs = encoder.encoder(input_ids=encoded_inputs["input_ids"], attention_mask=encoded_inputs["attention_mask"])["hidden_states"][end].detach()
            encoded_outputs = encoder.encoder(input_ids=encoded_outputs["input_ids"], attention_mask=encoded_outputs["attention_mask"])["hidden_states"][end].detach()

            encoded_inputs = encoded_inputs.mean(dim=1)
            encoded_outputs = encoded_outputs.mean(dim=1)

            # concatenate embeddings to final output matrix
            embeddings = torch.cat([encoded_inputs, encoded_outputs], dim=1)

            push!(problem_io_encodings, embeddings.cpu())
            GC.gc()
        end

        problem_io_encodings = torch.cat(problem_io_encodings, dim=0)

        problem_io_encodings = problem_io_encodings.mean(dim=0)

        return problem_io_encodings
    end
end
