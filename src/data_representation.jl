"""
Given a set of I/O examples, this returns an encoding of that. If it is also given a (partial) dictionary, then examples are represented with respect to that dict and the function returns the updated dict.
"""
function encode_IO_examples(io_examples::Vector{HerbData.IOExample}, emb_method::AbstractString="deepcoder", dict_partial=Dict{Any, Any}, MAX_LEN=20, EMBED_DIM=20)
    dict = Dict()
    if !isempty(dict_partial)
        dict = deepcopy(dict_partial)
    end

    if emb_method == "deepcoder"
        embeddings, emb_dict = deepcoder_IO_encoding(io_examples, MAX_LEN, EMBED_DIM)
        return embeddings, merge(dict_partial, emb_dict)
    else
        # encode with another embedding method (not yet implemented)
        error("Unknown embedding method: $emb_method")
    end
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

"""

"""
function encode_programs(programs::Vector{RuleNode}, emb_method::AbstractString="none", EMBED_DIM::Int=20)
    if emb_method == "none"
        return torch.zeros(length(programs), EMBED_DIM)
    elseif emb_method == "graph"
        error("Not implemented yet: $emb_method")
    elseif emb_method == "transformer"
        error("Not implemented yet: $emb_method")
    else
        error("Unknown embedding method: $emb_method")
    end
end

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