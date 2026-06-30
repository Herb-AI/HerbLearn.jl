"""
    get_partial_programs(programs::Vector{AbstractRuleNode}, grammar::AbstractGrammar)

Iterates over all given programs and samples `num_samples` partial programs by calling [`sample_partial_program`](@ref). Returns a vector of partial programs.
"""
function get_partial_programs(programs::Vector{RuleNode}, grammar::AbstractGrammar, num_samples::Int=5)
    data = []
    for program in programs
        for _ in 1:num_samples
            sampled_program = sample_partial_program(program, grammar)
            push!(data, sampled_program)
        end
    end
    return data
end

"""
    sample_partial_program(program::RuleNode, grammar::AbstractGrammar)

Given a program, samples a random node as a [`HerbGrammar.NodeLoc`](@ref) and substitutes it with a [`HerbCore.Hole`](@ref) of the same type. Return a tuple of the resulting partial program and the substituted correct rule derivation index.
"""
function sample_partial_program(program::RuleNode, grammar::AbstractGrammar)
    return_program = deepcopy(program)
    node_location::NodeLoc = sample(NodeLoc, return_program)
    parent, i = node_location.parent, node_location.i
    if i > 0
        orig_rule = parent.children[i].ind
        typ = grammar.types[orig_rule]
        parent.children[i] = Hole(get_domain(grammar, typ))
        return return_program, orig_rule
    else
        typ = grammar.types[return_program.ind]
        return_program = Hole(get_domain(grammar, typ))
        return return_program, program.ind
    end
end