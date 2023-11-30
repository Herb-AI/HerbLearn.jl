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

mutable struct StarCoderProgramEncoder <: AbstractProgramEncoder
    tokenizer
    embedder
    EMBED_DIM::Int

    encoder

    function StarCoderProgramEncoder(latent_dim::Int, encoder::PyObject=nothing)
        transformers = pyimport("transformers")
        device = torch.device(ifelse(torch.cuda.is_available(), "cuda", "cpu"))

        checkpoint = "bigcode/starencoder"
        tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint).to(device)
        embedder = transformers.AutoModelForPreTraining.from_pretrained(checkpoint).to(device)

        if isnothing(tokenizer.pad_token)
            tokenizer.add_special_tokens(Dict("pad_token" => "[PAD]"))
            embedder.resize_token_embeddings(length(tokenizer))
        end

        embed_dim = embedder.config.hidden_size

        if isnothing(encoder)
            encoder = MLP([EMBED_DIM, 128])
        end

        new(tokenizer, embedder, embed_dim, encoder)
    end

end


"""

"""
function represent_programs(programs::Vector{RuleNode}, encoder::ZeroProgramEncoder)
    return torch.zeros(length(programs), encoder.EMBED_DIM)
end

"""

"""
function represent_programs(programs::Vector{RuleNode}, grammar::Grammar, encoder::StarCoderProgramEncoder)
    device = torch.device(ifelse(torch.cuda.is_available(), "cuda", "cpu"))

    input_expr = [string(rulenode2expr(rulenode, grammar)) for rulenode ∈ programs]

    # tokenize string represenations
    encoded_programs = encoder.tokenizer(input_expr, padding=true, truncation=true, return_tensors="pt").to(device)

    encoded_programs = encoder.embedder(input_ids=encoded_programs["input_ids"], attention_mask=encoded_programs["attention_mask"])

    # take first hidden state of the transformer
    embeddings = encoded_programs["hidden_states"][0][:, 0, :].view(length(programs), encoder.EMBED_DIM)

    return embeddings
end

"""

"""
function represent_programs(programs::Vector{RuleNode}, encoder::GraphProgramEncoder)
    error("Not implemented yet: $encoder")
end


encode_programs(programs::Vector{RuleNode}, encoder::ZeroProgramEncoder) = represent_programs(programs, encoder)

"""

"""
function encode_programs(programs::Vector{RuleNode}, encoder::StarCoderProgramEncoder)
    #@TODO remove represent programs from here?

    embedded_programs = represent_programs(programs, encoder)
    return encoder.encoder(embedded_programs)
end


"""

"""
function encode_programs(programs::Vector{RuleNode}, encoder::GraphProgramEncoder)
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
function decode_programs(programs_so_far::Vector{RuleNode}, program_emb::Vector, grammar::Grammar)
    
end

"""
Function to generate labels in a DeepCoder fashion, i.e. mapping each program to a binary encoding whether a derivation was used in that program or not. Programs need to be given as a vector of HerbGrammar.RuleNode.
"""
function deepcoder_labels(programs::Vector{RuleNode}, grammar::Grammar)
    labels = torch.zeros(length(programs), length(grammar.rules))
    for (i, program) ∈ enumerate(programs)
        labels[i-1, :] = deepcoder_labels_for_node(program, grammar)
    end
    return labels
end


"""
Recursively computes DeepCoder encoding for a program given the root RuleNode.
"""
function deepcoder_labels_for_node(node::RuleNode, grammar::Grammar)
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

