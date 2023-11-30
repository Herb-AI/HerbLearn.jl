"""
This function 1. takes a vector of problems from the Benchmark directory, 2. samples programs from the grammar, 3. applies the generated programs on the inputs and 4. returns I/O samples + the program leading from input to output. This will generate `length(problem.examples) × num_samples` data points minus the number of samples+programs that failed on execution.+
"""
function generate_data(
        grammar::Grammar, 
        problem::Problem, 
        start::Symbol,
        num_samples::Int,
        evaluator::Function=execute_on_examples,
        max_depth::Union{Int, Nothing}=20, 
        max_time::Union{Int, Nothing}=nothing,
        max_enumerations::Union{Int, Nothing}=nothing,
        allow_evaluation_errors::Bool=false
    )::Vector{Tuple{IOExample, Any}}
    
    # Init checks
    start_time = time()
    check_time = max_time !== nothing
    check_enumerations = max_enumerations !== nothing

    # generate symbol table from given grammar
    symboltable :: SymbolTable = SymbolTable(grammar)


    # Sample programs searching program space
    gen_IO_data::Vector{Tuple{IOExample, Any}} = Vector{Tuple{IOExample, Expr}}()

    for i ∈ 1:num_samples
        h = rand(RuleNode, grammar, start, max_depth)
        expr = rulenode2expr(h, grammar)

        try
            outputs = evaluator(symboltable, expr, [example.in for example in problem.examples])

            # Align inputs and generated outputs
            IO_batch = []
            for (example, output) in zip(problem.examples, outputs)
                if length(string(output)) == 0
                    output = "()"
                end
                push!(IO_batch, IOExample(example.in, output))
            end
            
            # Repeat expr and align expr with generated IO examples
            IO_prog_batch = zip(IO_batch, repeat([h], length(problem.examples)))
            append!(gen_IO_data, collect(IO_prog_batch))
        catch ex
            if allow_evaluation_errors
                continue
            else
                throw(ex)
            end
        end

        # Check stopping conditions
        if check_enumerations && i > max_enumerations || check_time && time() - start_time > max_time
            return nothing
        end
    end

    return gen_IO_data
end