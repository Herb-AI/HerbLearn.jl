"""

"""
function iterate_partial_programs(programs::Vector{RuleNode}, grammar::Grammar)
    # Iterate over programs and for each derivation step,
    # add the program-so-far, the domain and the actual derivation to the data
    training_pairs = []
    for program in programs
        append(training_pairs, get_partial_programs(program, grammar))
    end

    return training_pairs
end


"""
    get_partial_programs(node::RuleNode, grammar::Grammar)

Extract all sub-programs from a given program, substituting each sub-tree in turn with a Hole.
"""
function get_partial_programs(node::RuleNode, grammar::Grammar)
    data = []

    # Recursive function to walk through the program tree and replace subtrees with Holes
    function walk_and_replace_with_holes(localnode::AbstractRuleNode, path::Vector{Int})
        if isa(localnode, RuleNode)
            # Create a Hole that could be replaced by a node with the same index
            hole = Hole(get_domain(grammar, grammar.childtypes[localnode.ind]))
            # Create a copy of the program with this node replaced by the Hole
            subprogram = deepcopy(node)
            swap_node(subprogram, hole, path)
            push!(data, (subprogram, hole, localnode.ind))

            # Recurse on children
            for (i, child) in enumerate(localnode.children)
                walk_and_replace_with_holes(child, [path; i])
            end
        end
    end

    # Initialize the recursion on the root node
    walk_and_replace_with_holes(node, [])

    return data
end
