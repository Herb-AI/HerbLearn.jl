abstract type ProgramEncoder end

mutable struct ZeroProgramEncoder <: ProgramEncoder end

mutable struct GraphProgramEncoder <: ProgramEncoder
    EMBED_DIM::Int
end

mutable struct TransformerProgramEncoder <: ProgramEncoder
    EMBED_DIM::Int
end

"""

"""
function encode_programs(programs::Vector{RuleNode}, encoder::ProgramEncoder, EMBED_DIM::Int=20)
    return torch.zeros(length(programs), EMBED_DIM)
end

"""

"""
function encode_programs(programs::Vector{RuleNode}, encoder::GraphProgramEncoder)
    error("Not implemented yet: $encoder")
end

"""

"""  
function encode_programs(programs::Vector{RuleNode}, encoder::TransformerProgramEncoder)
    error("Not implemented yet: $encoder")
end


mutable struct ProgramDecoder end

"""

"""
function decode_programs(program_emb::Vector)
    error("Decoding programs is not yet supported.")
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