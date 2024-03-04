mutable struct GeneratedProblem
    io_examples::Vector{IOExample}
    program::AbstractRuleNode
    partial_programs::Vector{Tuple{AbstractRuleNode, Int}}
    grammar::AbstractGrammar
end

"""
This function 1. takes a vector of problems from the Benchmark directory, 2. samples programs from the grammar, 3. applies the generated programs on the inputs and 4. returns I/O samples + the program leading from input to output. This will generate `length(problem.spec) × num_samples` data points minus the number of samples+programs that failed on execution.+
"""
function generate_data(
        grammar::AbstractGrammar, 
        problems::Vector{Problem{Vector{IOExample}}}, 
        start::Symbol,
        num_samples::Int,
        parts_per_program::Int,
        evaluator::Function=HerbInterpret.execute_on_input;
        min_depth::Union{Int, Nothing}=2,
        max_depth::Union{Int, Nothing}=20, 
        max_time::Union{Int, Nothing}=nothing,
        max_enumerations::Union{Int, Nothing}=nothing,
    )::Vector{GeneratedProblem}
    
    # Init checks
    start_time = time()
    check_time = max_time !== nothing
    check_enumerations = max_enumerations !== nothing

    # generate symbol table from given grammar
    symboltable::SymbolTable = SymbolTable(grammar)

    # Sample programs searching program space
    gen_IO_data = []
    for problem in ProgressBar(problems)
        i = 1
        while i <= num_samples
            h = rand(RuleNode, grammar, start, max_depth)
            if !isnothing(min_depth) && depth(h) < min_depth
                continue
            end
            expr = rulenode2expr(h, grammar)

            try
                # This assumes that every generated program is executable on every single input
                outputs = [output for output in [evaluator(
                    symboltable, expr, example.in) for example in problem.spec]]

                io_examples = []
                # Align inputs and generated outputs
                for (example, output) in zip(problem.spec, outputs)
                    if length(string(output)) == 0 && typeof(output) == String
                        output = "()"
                    end
                    if typeof(output) == String && typeof(example.in) == String && length(output) > 2*length(example.in)
                        throw(BoundsError)
                    end

                    io_example = IOExample(example.in, output)
                    push!(io_examples, io_example)
                    # push!(gen_IO_data, (io_example, h, partial_program, correct_derivation))
                end

                gen_problem = GeneratedProblem(io_examples, h, get_partial_programs([h], grammar, parts_per_program), grammar)
                
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