
"""
This function 1. takes a vector of problems from the Benchmark directory, 2. samples programs from the grammar, 3. applies the generated programs on the inputs and 4. returns I/O samples + the program leading from input to output. This will generate `length(problem.examples) × num_samples` data points.
@TODO introduce minimum depth for examples?
"""
function generate_data(g::HerbGrammar.Grammar, problem::HerbData.Problem, max_depth::Int, num_samples::Int, enumerator, start::Symbol)::Vector{Tuple{HerbData.IOExample, Any}}
    # generate symbol table from given grammar
    symboltable :: SymbolTable = HerbGrammar.SymbolTable(g)

    # Sample programs searching program space
    # sampled_programs::Vector{HerbGrammar.Expr} = Vector{HerbGrammar.Expr}()

    hypotheses = enumerator(g, max_depth, start)

    gen_IO_data::Vector{Tuple{HerbData.IOExample, Any}} = Vector{Tuple{HerbData.IOExample, Expr}}()

    for h :: HerbGrammar.RuleNode ∈ hypotheses
        expr = HerbGrammar.rulenode2expr(h, g)

        # Run generated program on input examples
        outputs = HerbEvaluation.execute_on_examples(symboltable, expr, [example.in for example in problem.examples])
        
        # Align inputs and generated outputs
        IO_batch = [HerbData.IOExample(example.in, output) for (example, output) in zip(problem.examples, outputs)]
        
        # Repeat expr and align iHerb.HerbGrammar.t with generated IO examples
        IO_prog_batch = zip(IO_batch, repeat([h], length(problem.examples)))
        append!(gen_IO_data, collect(IO_prog_batch))
        if length(gen_IO_data) / length(problem.examples) >= num_samples
            break
        end
    end

    return gen_IO_data
end

