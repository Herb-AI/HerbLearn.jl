"""
This function 1. takes a vector of problems from the Benchmark directory, 2. samples programs from the grammar, 3. applies the generated programs on the inputs and 4. returns I/O samples + the program leading from input to output. This will generate `length(problem.examples) × num_samples` data points minus the number of samples+programs that failed on execution.+
"""
function generate_data(
        grammar::Grammar, 
        problems::Vector{Problem}, 
        start::Symbol,
        num_samples::Int,
        part_per_program::Int,
        evaluator::Function=execute_on_examples;
        min_depth::Union{Int, Nothing}=3,
        max_depth::Union{Int, Nothing}=20, 
        max_time::Union{Int, Nothing}=nothing,
        max_enumerations::Union{Int, Nothing}=nothing,
    )::Vector{Tuple{IOExample, RuleNode, AbstractRuleNode, Int}}
    
    # Init checks
    start_time = time()
    check_time = max_time !== nothing
    check_enumerations = max_enumerations !== nothing

    # generate symbol table from given grammar
    symboltable::SymbolTable = SymbolTable(grammar)

    # Sample programs searching program space
    gen_IO_data = []
    for problem in problems
        i = 1
        while i <= num_samples
            h = rand(RuleNode, grammar, start, max_depth)
            if !isnothing(min_depth) && depth(h) < min_depth
                continue
            end
            expr = rulenode2expr(h, grammar)

            try
                outputs = [output for output in evaluator(
                    symboltable, expr, [example.in for example in problem.examples])]

                # Sample partial programs and extend return list
                for (partial_program, correct_derivation) in get_partial_programs([h], grammar, part_per_program)

                    # Align inputs and generated outputs
                    for (example, output) in zip(problem.examples, outputs)
                        if length(string(output)) == 0 && typeof(output) == String
                            output = "()"
                        end
                        io_example = IOExample(example.in, output)
                        push!(gen_IO_data, (io_example, h, partial_program, correct_derivation))
                    end
                end
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