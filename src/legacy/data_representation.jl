abstract type AbstractIOEncoder end

function encode_IO_examples(gen_problems::Vector{GeneratedProblem}, encoder::AbstractIOEncoder)
    return_features = []
    for gen_problem in ProgressBar(gen_problems)
        push!(return_features, encode_IO_examples(gen_problem.io_examples, encoder))
    end
    
    return torch.cat(return_features, dim=0).view(length(gen_problems), -1)
end

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


"""
Given a set of I/O examples, this returns an encoding of that. If it is also given a (partial) dictionary, then examples are represented with respect to that dict and the function returns the updated dict.
"""
function encode_IO_examples(io_examples::Vector{<:HerbSpecification.IOExample}, encoder::DeepCoderIOEncoder)
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
    end

    result_mat = torch.cat([vec.view(length(io_examples), -1) for vec in [inputs, input_types, outputs, output_type]], dim=1)
    return result_mat.max(dim=0)
end



abstract type AbstractStarCoderIOEncoder <: AbstractIOEncoder end

mutable struct StarEnCoderIOEncoder <: AbstractStarCoderIOEncoder
    tokenizer
    encoder
    EMBED_DIM::Int
    batch_size::Int

    function StarEnCoderIOEncoder(batch_size::Int=128, hidden_state::Int=13)
        transformers = pyimport("transformers")
        device = torch.device(ifelse(torch.cuda.is_available(), "cuda", "cpu"))

        checkpoint = "bigcode/starencoder"
        tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint)
        encoder = transformers.AutoModelForPreTraining.from_pretrained(checkpoint)
        encoder.eval()

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
            tokenizer.add_tokens(["[PAD]"])
            tokenizer.pad_token = "[PAD]"
            encoder.resize_token_embeddings(length(tokenizer), 8)
        end

        embed_dim = encoder.config.hidden_size# * 2

        encoder = torch.nn.DataParallel(encoder).to(device)

        new(tokenizer, encoder, embed_dim, batch_size)
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

    function StarCoderIOEncoder(batch_size::Int=128, max_sequence_length::Int=256, hidden_state::Int=1, model_path::AbstractString="")
        transformers = pyimport("transformers")

        checkpoint = "bigcode/starcoder"
        tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint)

        if isempty(model_path)
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

        if !isempty(model_path) && !isdir(model_path) 
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

"""

Make sure that `model_path` ends with a '/'. 
"""
mutable struct StarCoder2IOEncoder <: AbstractStarCoderIOEncoder
    tokenizer
    encoder
    EMBED_DIM::Int
    batch_size::Int
    max_sequence_length::Int

    function StarCoder2IOEncoder(batch_size::Int=128, max_sequence_length::Int=256, hidden_state::Int=1, model_path::AbstractString="")
        transformers = pyimport("transformers")

        checkpoint = "bigcode/starcoder2-15b"
        quantization_config = transformers.BitsAndBytesConfig(load_in_8bit=True)
        tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint)

        if isempty(model_path)
            encoder = transformers.AutoModel.from_pretrained(checkpoint, device_map = "auto", quantization_config=quantization_config)  # load to GPUs
        elseif endswith(model_path, '/') && !isdir(model_path)
            # Load model if not existent
            encoder = transformers.AutoModel.from_pretrained(checkpoint, device_map = "auto", quantization_config=quantization_config) 
            # Prune model to hidden state
            encoder.h = torch.nn.ModuleList([layer for (i, layer) in enumerate(encoder.h) if i<hidden_state]) # in range 1 to 40
            encoder.config.n_layer = hidden_state

        elseif isdir(model_path)
            # Load encoder from disk
            encoder = transformers.AutoModel.from_pretrained(model_path, device_map = "auto", quantization_config=quantization_config)  # load to gpu from disk
        else
            throw(ArgumentError("Invalid model path selected. Make sure to select a directory."))
        end

        for param in encoder.parameters() 
            param.requires_grad = false
        end

        if !isempty(model_path) && !isdir(model_path) 
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


function encode_IO_examples(gen_problems::Vector{GeneratedProblem}, encoder::AbstractStarCoderIOEncoder)
    num_examples = []
    all_examples = Vector{HerbSpecification.IOExample}()
    for gen_problem in gen_problems 
        append!(all_examples, gen_problem.io_examples)
        push!(num_examples, length(gen_problem.io_examples))
    end

    embed_matrix = encode_IO_examples(all_examples, encoder; summarize=false, verbose=true)

    return_features = []
    start_index = 1
    for spec_length in num_examples
        end_index = start_index + spec_length - 1
        
        # Extract a batch using PyTorch tensor slicing
        batch = embed_matrix.narrow(0, start_index - 1, spec_length)  # PyTorch uses zero-based indexing
        push!(return_features, batch.mean(dim=0))
        
        start_index = end_index + 1
    end
    
    return torch.cat(return_features, dim=0).view(length(gen_problems), -1)
end


"""
    encode_IO_examples(io_examples::Vector{<:HerbSpecification.IOExample}, encoder::AbstractStarCoderIOEncoder)

Encode input/output examples using the provided AbstractStarCoderIOEncoder.

# Arguments
- `io_examples::Vector{<:HerbSpecification.IOExample}`: A vector of input/output examples to be encoded.
- `encoder::AbstractStarCoderIOEncoder`: The encoder to be used for encoding the examples.

# Returns
- `embeddings`: The encoded input/output examples.

# Examples
```julia
io_examples = [HerbSpecification.IOExample(input1, output1), HerbSpecification.IOExample(input2, output2)]
encoder = StarEnCoderIOEncoder()
encoded_examples = encode_IO_examples(io_examples, encoder)
```

The resulting embeddings will be stored and returned as `torch.cpu()`. 
"""
function encode_IO_examples(io_examples::Vector{<:HerbSpecification.IOExample}, encoder::AbstractStarCoderIOEncoder; summarize::Bool=true, verbose=true, separate_io=true)
    device = torch.device(ifelse(torch.cuda.is_available(), "cuda", "cpu"))
    @pywith torch.no_grad() begin
        num_batches = ceil(Int, length(io_examples) / encoder.batch_size)
        data_batches = [io_examples[(i-1)*encoder.batch_size+1:min(i*encoder.batch_size, end)] for i in 1:num_batches]

        problem_io_encodings = []

        BOS_TOKEN = encoder.tokenizer.special_tokens_map["bos_token"]
        EOS_TOKEN = encoder.tokenizer.special_tokens_map["eos_token"]

        batch_iter = data_batches
        if verbose 
            batch_iter = ProgressBar(data_batches)
        end

        for batch in batch_iter
            # Use string representation of Dict
            input_strings = [string(example.in)[findfirst(isequal('('), string(example.in))+1:end-1] for example in batch]
            output_strings = [string(example.out) for example in batch]

            # Add special tokens
            if separate_io
                # Add special tokens to sentences; input and output are separated
                io_strings = ["$BOS_TOKEN$input$EOS_TOKEN$output$EOS_TOKEN" for (input, output) in zip(input_strings, output_strings)]
                
                # println("encode_IO_examples.io_strings: $io_strings")

                # generate embedding
                embeddings = _encode(io_strings,encoder)
                push!(problem_io_encodings, embeddings.cpu())
            else 
                # Add special tokens to sentences
                input_strings = ["$BOS_TOKEN$sentence$EOS_TOKEN" for sentence in input_strings]
                output_strings = ["$BOS_TOKEN$sentence$EOS_TOKEN" for sentence in output_strings]

                # embed string representations
                encoded_inputs = _encode(input_strings, encoder)
                encoded_outputs = _encode(output_strings, encoder)

                # concatenate embeddings to final output matrix
                embeddings = torch.cat([encoded_inputs, encoded_outputs], dim=1)
                push!(problem_io_encodings, embeddings.cpu())
            end
        end

        problem_io_encodings = torch.cat(problem_io_encodings, dim=0)

        problem_io_encodings = summarize ? problem_io_encodings.mean(dim=0) : problem_io_encodings

        return problem_io_encodings
    end
end

function encode_IO_examples2(gen_problems::Vector{GeneratedProblem}, encoder::AbstractStarCoderIOEncoder)
    error()
end

function encode_IO_examples2(io_examples::Vector{<:HerbSpecification.IOExample}, encoder::AbstractStarCoderIOEncoder; verbose=true)
    device = torch.device(ifelse(torch.cuda.is_available(), "cuda", "cpu"))
    @pywith torch.no_grad() begin



        # Use string representation of Dict
        input_strings = [string(example.in)[findfirst(isequal('('), string(example.in))+1:end-1] for example in batch]
        output_strings = [string(example.out) for example in batch]

        formatted_examples = CLS_TOKEN*join([input * " returns " * output], SEPARATOR_TOKEN) * SEPARATOR_TOKEN

        println("input_strings: ", input_strings)
        println("output_strings: ", output_strings)
        println("formatted_examples: ", formatted_examples)

        error()

        # Add special tokens
        if separate_io
            # Add special tokens to sentences; input and output are separated
            io_strings = ["$CLS_TOKEN$input$SEPARATOR_TOKEN$output$SEPARATOR_TOKEN" for (input, output) in zip(input_strings, output_strings)]

            # generate embedding
            embeddings = _encode(io_strings,encoder)
            push!(problem_io_encodings, embeddings.cpu())
        else 
            # Add special tokens to sentences
            input_strings = ["$CLS_TOKEN$sentence$SEPARATOR_TOKEN" for sentence in input_strings]
            output_strings = ["$CLS_TOKEN$sentence$SEPARATOR_TOKEN" for sentence in output_strings]

            # embed string representations
            encoded_inputs = _encode(input_strings, encoder)
            encoded_outputs = _encode(output_strings, encoder)

            # concatenate embeddings to final output matrix
            embeddings = torch.cat([encoded_inputs, encoded_outputs], dim=1)
            push!(problem_io_encodings, embeddings.cpu())
        end

        return problem_io_encodings
    end

end

function _encode(input_strings::Vector{<:AbstractString}, encoder::StarEnCoderIOEncoder) 
    device = torch.device(ifelse(torch.cuda.is_available(), "cuda", "cpu"))

    encoded_inputs = encoder.tokenizer(input_strings, padding="longest", return_tensors="pt", pad_to_multiple_of=8).to(device) 

    # println("_encode.encoded_inputs[input_ids]: $(encoded_inputs["input_ids"])")
    # println("_encode.encoded_inputs[attention_mask]: $(encoded_inputs["attention_mask"])")

    outputs = encoder.encoder(input_ids=encoded_inputs["input_ids"], attention_mask=encoded_inputs["attention_mask"])
    embeddings = pool_and_normalize(outputs["hidden_states"][end], encoded_inputs["attention_mask"]).detach()

    GC.gc(false)

    return embeddings
end

function _encode(input_strings::Vector{<:AbstractString}, encoder::StarCoderIOEncoder) 
    error("Rewrite this to use a proper embedding function.")
    device = torch.device(ifelse(torch.cuda.is_available(), "cuda", "cpu"))

    encoded_inputs = encoder.tokenizer(input_strings, padding="longest", return_tensors="pt", pad_to_multiple_of=8).to(device) 

    encoded_inputs = encoder.encoder(input_ids=encoded_inputs["input_ids"], attention_mask=encoded_inputs["attention_mask"])["last_hidden_state"].detach()
    GC.gc(false)

    return encoded_inputs
end

mutable struct PropertySignatureIOEncoder <: AbstractIOEncoder
    EMBED_DIM::Int  # default for String typed and related inputs/outputs

    function PropertySignatureIOEncoder()
        new(17)
    end
end

function encode_IO_examples(io_examples::Vector{<:HerbSpecification.IOExample}, encoder::PropertySignatureIOEncoder)
    # Generate a matrix where each row corresponds to properties from one pair
    properties_matrix = torch.vstack([encode_string_relationships(ex) for ex in io_examples])

    @assert properties_matrix.size(1) == encoder.EMBED_DIM

    # Check if all are true (1), all are false (0), or mixed (-1, 0, 1)
    all_true = torch.all(properties_matrix == 1, dim=0)
    all_false = torch.all(properties_matrix == 0, dim=0)

    # Calculate the signature
    signature = torch.zeros(properties_matrix.size(1), dtype=torch.int32)
    signature = torch.where(all_true, torch.tensor(1, dtype=torch.int32), signature)
    signature = torch.where(all_false, torch.tensor(-1, dtype=torch.int32), signature)

    return signature.float()
end

function encode_string_relationships(example::T) where T<:HerbSpecification.IOExample
    return_vec = torch.vstack([encode_string_relationships(string(val), string(example.out)) for val in values(example.in)])
    return return_vec.max(dim=0)[1]
end

function encode_string_relationships(str::String, outputStr::String)
    lower = lowercase(str)
    outputStrLower = lowercase(outputStr)

    properties = [
        occursin(str, outputStr),                            # output contains input?
        startswith(outputStr, str),                          # output starts with input?
        endswith(outputStr, str),                            # output ends with input?
        occursin(outputStr, str),                            # input contains output?
        startswith(str, outputStr),                          # input starts with output?
        endswith(str, outputStr),                            # input ends with output?
        occursin(lower, outputStrLower),                     # output contains input ignoring case?
        startswith(outputStrLower, lower),                   # output starts with input ignoring case?
        endswith(outputStrLower, lower),                     # output ends with input ignoring case?
        occursin(outputStrLower, lower),                     # input contains output ignoring case?
        startswith(lower, outputStrLower),                   # input starts with output ignoring case?
        endswith(lower, outputStrLower),                     # input ends with output ignoring case?
        str == outputStr,                                    # input equals output?
        lower == outputStrLower,                             # input equals output ignoring case?
        length(str) == length(outputStr),                    # input same length as output?
        length(str) < length(outputStr),                     # input shorter than output?
        length(str) > length(outputStr)                      # input longer than output?
    ]

    # Convert boolean array to a tensor of type Float
    return torch.tensor(Float32.(properties))
end

function encode_single_string_signature(str::String)
    lower = lowercase(str)
    upper = uppercase(str)

    properties = [
        isempty(str),                          # is empty?
        length(str) == 1,                      # is single char?
        length(str) <= 5,                      # is short string?
        str == lower,                          # is lowercase?
        str == upper,                          # is uppercase?
        occursin(" ", str),                    # contains space?
        occursin(",", str),                    # contains comma?
        occursin(".", str),                    # contains period?
        occursin("-", str),                    # contains dash?
        occursin("/", str),                    # contains slash?
        occursin(r"\d", str),                  # contains digits?
        occursin(r"^\d+$", str),               # only digits?
        occursin(r"[a-zA-Z]", str),            # contains letters?
        occursin(r"^[a-zA-Z]+$", str)          # only letters?
    ]

    # Convert boolean array to a tensor of type Float
    return torch.tensor(Float32.(properties))
end


