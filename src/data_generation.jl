mutable struct GeneratedProblem
    io_examples::Vector{IOExample}
    program::AbstractRuleNode
    # partial_programs::Vector{Tuple{AbstractRuleNode, Int}}
    grammar::AbstractGrammar
end

contains_node(rn::RuleNode, ind::Int) = rn.ind == ind || any(contains_node(c, ind) for c ∈ rn.children)
contains_node(hole::Hole, ind::Int) = false

input_rules(grammar::AbstractGrammar) = findall(rule -> occursin("_arg_", string(rule)), grammar.rules)


"""
This function 1. takes a vector of problems from the Benchmark directory, 2. samples programs from the grammar, 3. applies the generated programs on the inputs and 4. returns I/O samples + the program leading from input to output. This will generate `length(problem.spec) × num_samples` data points minus the number of samples+programs that failed on execution.+
"""
function generate_data(
        problem_grammar_pairs::Vector{ProblemGrammarPair},
        start::Symbol,
        num_samples::Int,
        evaluator::Function=HerbInterpret.execute_on_input;
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

        if length(problem.spec) > 50
            continue
        end

        symboltable::SymbolTable = SymbolTable(grammar)
        i = 1
        while i <= num_samples
            h = rand(RuleNode, grammar, start, max_depth)
            # check whether hypothesis rulenode has minimum depth
            if !isnothing(min_depth) && depth(h) < min_depth
                continue
            end
            # check whether `_arg_X` are present at least once
            if !all(contains_node(h, ind) for ind in input_rules(grammar))
                continue
            end

            expr = rulenode2expr(h, grammar)

            try
                # This assumes that every generated program is executable on every single input
                outputs = [output for output in [evaluator(
                    symboltable, expr, example.in) for example in problem.spec]]
                
                io_examples = []

                # Guard for BV manipulations discarding 0x000 and equivalents
                if all(output == 0 || length(output)==0 for output in outputs)
                    throw(BoundsError)
                end

                # Align inputs and generated outputs
                for (example, output) in zip(problem.spec, outputs)
                    if length(string(output)) == 0 && typeof(output) == String
                        output = "()"
                    end

                    # Guard for too long string transformations
                    if all(length(string(output)) > 2*length(string(value)) for value in values(example.in))
                        throw(BoundsError)
                    end

                    io_example = IOExample(example.in, output)
                    push!(io_examples, io_example)
                end
                # gen_problem = GeneratedProblem(io_examples, h, get_partial_programs([h], grammar, parts_per_program), grammar)
                gen_problem = GeneratedProblem(io_examples, h, grammar)
                
                push!(gen_IO_data, gen_problem)
            catch ex
                continue
            end

            i += 1

            # Check stopping conditions
            if check_enumerations && i > max_enumerations || check_time && time() - start_time > max_time
                return nothing
            end
        end
    end

    return gen_IO_data
end