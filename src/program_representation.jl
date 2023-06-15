abstract type ProgramEmbedder end

mutable struct ZeroProgramEmbedder <: ProgramEmbedder end

mutable struct GraphProgramEmbedder <: ProgramEmbedder
    embedding_function::Function
end

mutable struct TransformerProgramEmbedder <: ProgramEmbedder
    embedding_function::Function
end

"""

"""
function encode_programs(programs::Vector{RuleNode}, embedder::ProgramEmbedder, EMBED_DIM::Int=20)
    return torch.zeros(length(programs), EMBED_DIM)
end

"""

"""
function encode_programs(programs::Vector{RuleNode}, embedder::GraphProgramEmbedder, EMBED_DIM::Int=20)
    error("Not implemented yet: $embedder")
end

"""

"""  
function encode_programs(programs::Vector{RuleNode}, embedder::TransformerProgramEmbedder, EMBED_DIM::Int=20)
    error("Not implemented yet: $embedder")
end


mutable struct ProgramDecoder end

"""

"""
function decode_programs(program_emb::Vector)
    error("Decoding programs is not yet supported.")
end

"""
Function to generate labels in a DeepCoder fashion, i.e. mapping each program to a binary encoding whether a derivation was used in that program or not. Programs need to be given as HerbGrammar.RuleNode.
"""
function deepcoder_labels(programs::Vector{RuleNode}, grammar::Grammar)
    labels = torch.zeros(length(programs), length(grammar.rules))
    # @TODO get labels by traversing each AST
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
            retval *= deepcoder_labels_for_node(child, grammar)  # logical AND over Boolean vector
        end
        return retval
    end

end