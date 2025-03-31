mutable struct GeneratedProblem
    io_examples::Vector{IOExample}
    program::AbstractRuleNode
    # partial_programs::Vector{Tuple{AbstractRuleNode, Int}}
    grammar::AbstractGrammar
end

contains_node(rn::RuleNode, ind::Int) = rn.ind == ind || any(contains_node(c, ind) for c ∈ rn.children)
contains_node(hole::Hole, ind::Int) = false

input_rules(grammar::AbstractGrammar) = findall(rule -> occursin("_arg_", string(rule)), grammar.rules)

@programiterator RandomSampler(min_depth::Union{Nothing, Int}=nothing) 

function Base.iterate(iter::RandomSampler, state=nothing)
    grammar = HerbConstraints.get_grammar(iter.solver)
    start = get_starting_symbol(iter.solver)
    max_depth = get_max_depth(iter.solver)
    for i in 1:100000
        h = rand(RuleNode, grammar, start, max_depth)
        # check whether hypothesis rulenode has minimum depth
        if !isnothing(iter.min_depth) && depth(h) < iter.min_depth
            continue
        end
        # check whether `_arg_X` are present at least once

        if length(grammar.rules) > 50
            if !any(contains_node(h, ind) for ind in input_rules(grammar))
                continue
            end
        else
            if !all(contains_node(h, ind) for ind in input_rules(grammar))
                continue
            end
        end

        return h, nothing
    end
    iter.max_depth = max_depth + 1

    error("Too many samples had to be drawn. Maybe try another iterator or higher depth instead.")
end


"""
This function 1. takes a vector of problems from the Benchmark directory, 2. samples programs from the grammar, 3. applies the generated programs on the inputs and 4. returns I/O samples + the program leading from input to output. This will generate `length(problem.spec) × num_samples` data points minus the number of samples+programs that failed on execution.+
"""
function generate_data(
        problem_grammar_pairs::Vector{ProblemGrammarPair},
        start::Symbol,
        num_samples::Int;
        ben_module::Module=Main,
        min_depth::Union{Int, Nothing}=2,
        max_depth::Int=20, 
        max_time::Union{Int, Nothing}=nothing,
        max_enumerations::Union{Int, Nothing}=nothing,
    )::Vector{GeneratedProblem}
    
    # Init checks
    start_time = time()
    check_time = max_time !== nothing
    check_enumerations = max_enumerations !== nothing

    # Sample programs searching program space
    gen_IO_data = []
    for pair in ProgressBar(problem_grammar_pairs)
        problem, grammar = pair.problem, deepcopy(pair.grammar)
        gen_programs = Set{RuleNode}()

        # discard longer examples
        if length(problem.spec) > 50
            continue
        end

        iter = nothing
        if length(grammar.rules) < 100
            iter = RandomSampler(grammar, start, min_depth=min_depth, max_depth=max_depth)
        else 
            iter = HerbSearch.RandomSearchIterator(grammar, start, max_depth=max_depth)
            # check whether `_arg_X` are present at least once
            for i in 1:min(length(problem.spec[1].in),2)
                addconstraint!(grammar, Contains(findfirst(rule -> occursin("_arg_$i", string(rule)), grammar.rules)))
            end
        end

        st = nothing
        if string(ben_module) != "HerbBenchmarks.String_transformations_2020"
            st = grammar2symboltable(grammar)
        end

        i = 0
        for h in iter
            if !isnothing(min_depth) && depth(h) < min_depth
                continue
            end
            if h in gen_programs
                continue
            end

            expr = rulenode2expr(h, grammar)

            try
                # Check that every generated program is executable on every single input
                output = nothing
                if string(ben_module) == "HerbBenchmarks.String_transformations_2020"
                    outputs = [ben_module.interpret(h, grammar, ex) for ex in problem.spec]
                else
                    outputs = [execute_on_input(st, expr, ex.in) for ex in problem.spec]
                end

                io_examples = []

                # Guard for BV manipulations discarding 0x000 and equivalents
                if all(output == 0 || length(output)==0 for output in outputs)
                    throw(BoundsError)
                end

                # Align inputs and generated outputs
                for (example, output) in zip(problem.spec, outputs)
                    if length(string(output)) == 0 && typeof(output) == String
                        #@TODO allow empty strings to be generated?
                        throw(BoundsError)
                    end

                    # Guard for too long string transformations
                    if all(length(string(output)) > 2*length(string(value)) for value in values(example.in))
                        throw(BoundsError)
                    end

                    io_example = IOExample(example.in, output)
                    push!(gen_programs, h)
                    push!(io_examples, io_example)
                end
                # gen_problem = GeneratedProblem(io_examples, h, get_partial_programs([h], grammar, parts_per_program), grammar)
                gen_problem = GeneratedProblem(io_examples, h, grammar)
                
                push!(gen_IO_data, gen_problem)
            catch ex
                continue
            end

            i += 1
            if i == num_samples
                break
            end
        end
    end

    return gen_IO_data
end