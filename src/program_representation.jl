abstract type AbstractProgramEncoder end

mutable struct ZeroProgramEncoder <: AbstractProgramEncoder 
    EMBED_DIM::Int

    function ZeroProgramEncoder(embed_dim::Int=2)
        new(embed_dim)
    end
end


mutable struct GraphProgramEncoder <: AbstractProgramEncoder
    EMBED_DIM::Int

    function GraphProgramEncoder()
        error("Not implemented yet: GraphProgramEncoder")
    end
end

abstract type AbstractStarCoderProgramEncoder <: AbstractProgramEncoder end

mutable struct StarEnCoderProgramEncoder <: AbstractStarCoderProgramEncoder
    tokenizer
    encoder
    EMBED_DIM::Int
    batch_size::Int
    max_sequence_length::Int

    embedder

    function StarEnCoderProgramEncoder(latent_dim::Union{Int, Nothing}=nothing, batch_size::Int=128, max_sequence_length::Int=256, hidden_state::Int=1)
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

        embed_dim = encoder.config.hidden_size

        encoder = torch.nn.DataParallel(encoder).to(device)

        # if latent_dim==nothing, don't init and apply dimensionality reduction/embedding function
        if isnothing(latent_dim)
            embedder = (x->x)
        else
            embedder = MLP([embed_dim, latent_dim])
            embed_dim = latent_dim
        end

        new(tokenizer, encoder, embed_dim, batch_size, max_sequence_length, embedder)

    end
end


"""

Make sure that `model_path` ends with a '/'. 
"""
mutable struct StarCoderProgramEncoder <: AbstractStarCoderProgramEncoder
    tokenizer
    encoder
    EMBED_DIM::Int
    batch_size::Int
    max_sequence_length::Int

    embedder

    function StarCoderProgramEncoder(latent_dim::Union{Int, Nothing}=nothing, batch_size::Int=128, max_sequence_length::Int=256, hidden_state::Int=1, model_path::AbstractString="")
        transformers = pyimport("transformers")

        checkpoint = "bigcode/starcoder"
        tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint)

        if isemtpy(model_path)
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

        embed_dim = encoder.config.hidden_size * max_sequence_length
        # if latent_dim==nothing, don't init and apply dimensionality reduction/embedding function
        if isnothing(latent_dim)
            embedder = (x->x)
        else
            embedder = MLP([embed_dim, latent_dim])
            embed_dim = latent_dim
        end

        new(tokenizer, encoder, embed_dim, batch_size, max_sequence_length, embedder)
    end
end


"""

Make sure that `model_path` ends with a '/'. 
"""
mutable struct StarCoder2ProgramEncoder <: AbstractStarCoderProgramEncoder
    tokenizer
    encoder
    EMBED_DIM::Int
    batch_size::Int
    max_sequence_length::Int

    embedder

    function StarCoder2ProgramEncoder(latent_dim::Union{Int, Nothing}=nothing, batch_size::Int=128, max_sequence_length::Int=256, hidden_state::Int=1, model_path::AbstractString="")
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

        if isnothing(tokenizer.pad_token)
            tokenizer.add_special_tokens(Dict("pad_token" => "[PAD]"))
            encoder.resize_token_embeddings(length(tokenizer))
        end

        embed_dim = encoder.config.hidden_size * max_sequence_length
        # if latent_dim==nothing, don't init and apply dimensionality reduction/embedding function
        if isnothing(latent_dim)
            embedder = (x->x)
        else
            embedder = MLP([embed_dim, latent_dim])
            embed_dim = latent_dim
        end

        new(tokenizer, encoder, embed_dim, batch_size, max_sequence_length, embedder)
    end
end



"""

"""
function encode_programs(generated_problems::Vector{GeneratedProblem}, encoder::AbstractProgramEncoder)
    return_vec = []
    for gen_problem in ProgressBar(generated_problems)
        partial_programs = [tup[1] for tup in gen_problem.partial_programs]
        input_expr = [string(rulenode2expr(rulenode, gen_problem.grammar)) for rulenode ∈ partial_programs]
        push!(return_vec, encode_programs(input_expr, encoder))
    end
    return return_vec
end

"""

"""
function encode_grammar(generated_problems::Vector{GeneratedProblem}, encoder::AbstractProgramEncoder)
    return_vec = []
    if all(p.grammar == generated_problems[1].grammar for p in generated_problems) 
        println("Bumm")
        vec = encode_grammar(generated_problems[1].grammar, encoder)
        return [vec for _ in 1:length(generated_problems)]
    end

    for gen_problem in ProgressBar(generated_problems)
        push!(return_vec, encode_grammar(gen_problem.grammar, encoder))
    end
    return return_vec
end

function encode_grammar(grammar::AbstractGrammar, encoder::AbstractProgramEncoder)
    rules = [string(rule) for rule in grammar.rules]
    return encode_programs(rules, encoder)
end

"""

"""
function encode_programs(programs::Vector{<:AbstractRuleNode}, encoder::ZeroProgramEncoder)
    return torch.zeros(length(programs), encoder.EMBED_DIM)
end

function encode_programs(programs::Vector{<:AbstractString}, encoder::ZeroProgramEncoder)
    return torch.zeros(length(programs), encoder.EMBED_DIM)
end

"""

"""
function encode_programs(programs::Vector{<:AbstractString}, encoder::AbstractStarCoderProgramEncoder)
    device = torch.device(ifelse(torch.cuda.is_available(), "cuda", "cpu"))

    num_batches = ceil(Int, length(programs) / encoder.batch_size)
    data_batches = [programs[(i-1)*encoder.batch_size+1:min(i*encoder.batch_size, end)] for i in 1:num_batches]

    return_matrix = []

    @pywith torch.no_grad() begin
        for batch in data_batches
            # tokenize string represenations
            encoded_programs = encoder.tokenizer(batch, padding=true, truncation=true, return_tensors="pt").to(device)
            encoded_programs = encoder.encoder(input_ids=encoded_programs["input_ids"], attention_mask=encoded_programs["attention_mask"])["hidden_states"][end].detach()

            encoded_programs = encoded_programs.mean(dim=1)

            # encoded_programs = encoded_programs.view(encoded_programs.size(0), -1)

            push!(return_matrix, encoded_programs.cpu())
            GC.gc()
        end
    end
    return torch.cat(return_matrix, dim=0) 
end

"""

"""
function encode_programs(programs::Vector{<:AbstractRuleNode}, encoder::GraphProgramEncoder)
    error("Not implemented yet: $encoder")
end


embed_programs(programs::Vector{<:AbstractRuleNode}, encoder::ZeroProgramEncoder) = encode_programs(programs, encoder)

"""

"""
function embed_programs(programs::Vector{AbstractRuleNode}, encoder::AbstractStarCoderProgramEncoder)
    encoded_programs = encode_programs(programs, encoder)
    encoded_programs = encoded_programs.view(encoded_programs.size(0), -1)

    # apply embedding function
    embedded_programs = encoder.embedder(encoded_programs)

    return embedded_programs
end

"""

"""
function embed_programs(programs::Vector{<:AbstractRuleNode}, encoder::GraphProgramEncoder)
    error("Not implemented yet: $encoder")
end


abstract type AbstractProgramDecoder end

mutable struct ProgramDecoder <: AbstractProgramDecoder
    LATENT_DIM::Int
    decoder_policy

    function ProgramDecoder(latent_dim::Int, num_derivations::Int, decoder_policy::PyObject=nothing)
        if isnothing(decoder_policy)
            decoder_policy = MLP([latent_dim, 128, num_derivations])
        end

        new(latent_dim, decoder_policy)
    end

end

"""

"""
function decode_programs(programs_so_far::Vector{<:AbstractRuleNode}, program_emb::Vector, grammar::AbstractGrammar)
    error("Not implemented yet: decode_programs")
end

"""
Function to generate labels in a DeepCoder fashion, i.e. mapping each program to a binary encoding whether a derivation was used in that program or not. Programs need to be given as a vector of HerbGrammar.AbstractRuleNode.
"""
function deepcoder_labels(gen_problems::Vector{GeneratedProblem})
    return_vec = []
    for gen_problem in gen_problems
        @assert length(gen_problem.partial_programs) == 1
        push!(return_vec, torch.unsqueeze(deepcoder_labels_for_node(gen_problem.program, gen_problem.grammar), dim=0))
    end
    return return_vec
end

"""
Recursively computes DeepCoder encoding for a program given the root (Abstract)RuleNode.
"""
function deepcoder_labels_for_node(node::AbstractRuleNode, grammar::AbstractGrammar)
    retval = torch.zeros(length(grammar.rules))

    retval[node.ind-1] = 1

    if isempty(node.children)
        return retval
    else
        for child ∈ node.children
            retval = torch.logical_or(deepcoder_labels_for_node(child, grammar), retval).int()  # logical OR over Boolean vector
        end
        return retval
    end
end


function label_encoding(problems::Vector{GeneratedProblem})
    return_vec = []
    for gen_problem in problems
        correct_derivations = [tup[2] for tup in gen_problem.partial_programs]
        push!(return_vec, label_encoding(correct_derivations, gen_problem.grammar))
    end
    return return_vec
end


function label_encoding(correct_derivations, grammar)
    # Size: #num_sampled_programs x #rules
    label_vec = torch.zeros(length(correct_derivations), length(grammar.rules))
    for (i, correct_derivation) in enumerate(correct_derivations)
        label_vec[i-1][correct_derivation-1] = 1
    end
    return label_vec
end