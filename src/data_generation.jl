"""
This function 1. takes a vector of problems from the Benchmark directory, 2. samples programs from the grammar, 3. applies the generated programs on the inputs and 4. returns I/O samples + the program leading from input to output. This will generate `length(problem.examples) × num_samples` data points.
@TODO introduce minimum depth for examples?
@TODO sample random programs instead of considering consecutively explored ones.
"""
function generate_data(
        g::Grammar, 
        problem::Problem, 
        start::Symbol,
        num_samples::Int,
        enumerator::Function; 
        evaluator::Function=execute_on_examples,
        max_depth::Union{Int, Nothing}=nothing, 
        max_size::Union{Int, Nothing}=nothing,
        max_time::Union{Int, Nothing}=nothing,
        max_enumerations::Union{Int, Nothing}=nothing,
        allow_evaluation_errors::Bool=false
    )::Vector{Tuple{IOExample, Any}}
    
    # Init checks
    start_time = time()
    check_time = max_time !== nothing
    check_enumerations = max_enumerations !== nothing

    # generate symbol table from given grammar
    symboltable :: SymbolTable = SymbolTable(g)


    # Sample programs searching program space
    # sampled_programs::Vector{HerbGrammar.Expr} = Vector{HerbGrammar.Expr}()

    hypotheses = enumerator(
        g, 
        max_depth ≡ nothing ? typemax(Int) : max_depth, 
        max_size ≡ nothing ? typemax(Int) : max_size,
        start
    )

    gen_IO_data::Vector{Tuple{IOExample, Any}} = Vector{Tuple{IOExample, Expr}}()

    for (i, h) ∈ enumerate(hypotheses)
        expr = rulenode2expr(h, g)

        outputs = evaluator(symboltable, expr, [example.in for example in problem.examples])
        
        # Align inputs and generated outputs
        IO_batch = [IOExample(example.in, output) for (example, output) ∈ zip(problem.examples, outputs)]
        
        # Repeat expr and align iHerb.HerbGrammar.t with generated IO examples
        IO_prog_batch = zip(IO_batch, repeat([h], length(problem.examples)))
        append!(gen_IO_data, collect(IO_prog_batch))
        if length(gen_IO_data) / length(problem.examples) >= num_samples
            break
        end

        # Check stopping conditions
        if check_enumerations && i > max_enumerations || check_time && time() - start_time > max_time
            return nothing
        end
    end

    return gen_IO_data
end