"""
Pre-training procedure taking care of 
1. data-loading and preparation
2. training 
3. evaluation
of a NN model
"""
function pretrain_heuristic(
    problem_grammar_pairs::Union{Vector{ProblemGrammarPair}, Tuple{Vector{Problem{Vector{IOExample}}}, <:AbstractGrammar}};
    data_encoder::AbstractIOEncoder,
    grammar_encoder::AbstractProgramEncoder,
    start::Symbol=:Start,
    num_programs::Int=100,
    num_partial_programs::Int=5,
    n_epochs::Int=100,
    batch_size::Int=20,
    data_dir::AbstractString="",
    semantic=false)

    program_encoder = ZeroProgramEncoder(2)

    # init computing device
    device = torch.device(ifelse(torch.cuda.is_available(), "cuda", "cpu"))
    println("Computing device: ", device)
    flush(stdout)

    num_problems = typeof(problem_grammar_pairs) <: Vector{ProblemGrammarPair} ? length(problem_grammar_pairs) : length(problem_grammar_pairs[1])

    data_ids = "$(semantic ? "sem" : "nonsem")_$(string(typeof(data_encoder)))_$(string(typeof(grammar_encoder)))"
    encoded_data_path = joinpath(data_dir, "encoded_$(data_ids)_data.pth")
    generated_problems_path = joinpath(data_dir, "generated_$(data_ids)_problems.jls")
    if isempty(data_dir) || !isfile(encoded_data_path) || !isfile(generated_problems_path)
        println("Generating data...")
        gen_problems = generate_data(problem_grammar_pairs, start, num_programs, num_partial_programs; min_depth=3, max_depth=6)

        println("Encoding data...")
        io_encodings = encode_IO_examples(gen_problems, data_encoder)
        println("io_encodings", io_encodings.size())
        println("Encoding programs... ")
        program_encodings = encode_programs(gen_problems, program_encoder)
        println("Encoding labels...")
        # label_data = label_encoding(gen_problems)
        label_data = deepcoder_labels(gen_problems)

        println("Encoding grammars...")
        grammar_encodings = semantic ? encode_grammar(gen_problems, grammar_encoder) : nothing
        println("Done.")
        ""
        data_to_save = Dict(
            "io_encodings" => io_encodings,
            "program_encodings" => program_encodings,
            "label_data" => label_data,
            "grammar_encodings" => grammar_encodings
        )
        torch.save(data_to_save, encoded_data_path)
        
        open(generated_problems_path, "w") do file
            serialize(file, gen_problems)
        end
    else
        loaded_data = torch.load(encoded_data_path)
        io_encodings = loaded_data["io_encodings"]
        program_encodings = loaded_data["program_encodings"]
        label_data = loaded_data["label_data"]

        grammar_encodings = semantic ? loaded_data["grammar_encodings"] : nothing

        gen_problems = open(deserialize, generated_problems_path)

        #assert to check whether parts per program matches with loaded data size
        @assert length(gen_problems) / num_problems == num_partial_programs
    end
    println("Number of generated IOP problems: ", length(gen_problems))

    label_size = sum([reduce(*, vec.size()) for vec in label_data])
    label_sum = sum([vec.sum() for vec in label_data])
    println("label_data.size(): ", label_size)
    println("label_data.sum(): ", label_sum)
    println("label_data.ratio: ", label_sum/label_size)

    # init train-validation split
    train_size = floor(Int, 0.8 * num_problems)
    test_size = num_problems - train_size
    train_indices, test_indices = torch.utils.data.random_split(torch.arange(num_problems), [train_size, test_size])

    # init dataloaders
    if !semantic
        train_dataloader = DataLoader(train_indices, io_encodings, program_encodings, label_data, batch_size, true, num_programs)
        test_dataloader = DataLoader(test_indices, io_encodings, program_encodings, label_data, batch_size, false, num_programs)

        grammar_length(g::AbstractGrammar) = length(g.rules)
        reference_length = grammar_length(gen_problems[1].grammar)
        @assert all(gp -> grammar_length(gp.grammar) == reference_length, gen_problems)

        model = DerivationPredNet(data_encoder.EMBED_DIM, program_encoder.EMBED_DIM, reference_length, [64, 64])
    else 
        train_dataloader = nothing
        test_dataloader = nothing
        if typeof(problem_grammar_pairs) <: Tuple{Vector{Problem{Vector{IOExample}}}, <:AbstractGrammar}
            train_dataloader = MonoGrammarPrototypeDataLoader(train_indices, io_encodings, program_encodings, grammar_encodings, label_data, batch_size, true, num_programs)
            test_dataloader = MonoGrammarPrototypeDataLoader(test_indices, io_encodings, program_encodings, grammar_encodings, label_data, batch_size, false, num_programs)

            println("length of dataloader: $(length(train_dataloader)) $(0.8*length(gen_problems))")
        elseif typeof(problem_grammar_pairs) <: Vector{ProblemGrammarPair}
            train_dataloader = PrototypeDataLoader(train_indices, io_encodings, program_encodings, grammar_encodings, label_data, true, num_programs)
            test_dataloader = PrototypeDataLoader(test_indices, io_encodings, program_encodings, grammar_encodings, label_data, false, num_programs)
        else
            error("Invalid type for input data")
        end

        model = SemanticDerivationPredNet(data_encoder.EMBED_DIM, program_encoder.EMBED_DIM, grammar_encoder.EMBED_DIM, [64, 64])
    end
   
    model = torch.nn.DataParallel(model).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    weights = torch.tensor([label_sum/label_size, 1-label_sum/label_size]) 
    println("weights:", weights)
    loss_func = BCELoss_class_weighted(weights/weights.sum())

    # do training within splits
    println("Starting training...")
    @time train!(model, train_dataloader, test_dataloader, n_epochs, optimizer, device, loss_func)

    return model
end


function train!(model, train_dataloader::DataLoader, valid_dataloader::DataLoader, n_epochs, optimizer, device, loss_func)
    for epoch in 1:n_epochs
        accu_loss = 0
        for (i, (io_emb, prog_emb, y)) in enumerate(train_dataloader)
            batch_size = io_emb.size(0)
            (io_emb, prog_emb, y) = io_emb.to(device), prog_emb.to(device), y.to(device)
            output = model(io_emb, prog_emb)
            loss = loss_func(output, y)/length(train_dataloader)
            accu_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        end
        epoch % 100 == 0 && println("Epoch: $epoch\tLoss: $(accu_loss)")  # Probe performance

        if epoch % 500 == 0 
            train_preds, train_labels = eval(model, train_dataloader, device)
            valid_preds, valid_labels = eval(model, valid_dataloader, device)

            train_score = metrics.accuracy_score(train_preds.flatten()>0.5, train_labels.flatten())
            valid_score = metrics.accuracy_score(valid_preds.flatten()>0.5, valid_labels.flatten())
            println("Accuracy train/test:\t", train_score, "\t", valid_score)
        end

        flush(stdout)
        GC.gc()
    end
end

function train!(model, train_dataloader::MonoGrammarPrototypeDataLoader, valid_dataloader::MonoGrammarPrototypeDataLoader, n_epochs, optimizer, device, loss_func)
    for epoch in 1:n_epochs
        accu_loss = 0
        for (i, (io_emb, prog_emb, grammar_emb, y)) in enumerate(train_dataloader)
            batch_size = io_emb.size(0)
            (io_emb, prog_emb, grammar_emb, y) = io_emb.to(device), prog_emb.to(device), grammar_emb.to(device), y.to(device)

            output = model(io_emb, prog_emb, grammar_emb)
            loss = loss_func(output, y)/length(train_dataloader)
            accu_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        end
        epoch % 100 == 0 && println("Epoch: $epoch\tLoss: $(accu_loss)")  # Probe performance

        if epoch % 500 == 0 
            train_preds, train_labels = eval(model, train_dataloader, device)
            valid_preds, valid_labels = eval(model, valid_dataloader, device)

            train_score = metrics.accuracy_score(train_labels.flatten(), train_preds.flatten()>0.5)
            valid_score = metrics.accuracy_score(valid_labels.flatten(),valid_preds.flatten()>0.5)
            println("Accuracy train/test:\t", train_score, "\t", valid_score)
            println("Train sanity check: ", train_preds.max(), train_preds.mean())
            println("Validation sanity check: ", valid_preds.max(), valid_preds.mean())
        end

        flush(stdout)
        GC.gc()
    end
end


function eval(model, testLoader::DataLoader, device)
    model.eval()
    preds, labels = torch.Tensor(), torch.Tensor()
    @pywith torch.no_grad() begin
        for (i, (io_emb, prog_emb, y)) in enumerate(testLoader)
            (io_emb, prog_emb, y) = io_emb.to(device), prog_emb.to(device), y.to(device)
            output = model(io_emb, prog_emb)

            preds = torch.cat([preds, output.to("cpu")], dim=0)
            labels = torch.cat([labels, y.to("cpu")], dim=0)

            GC.gc(false)
        end
        GC.gc()
    end
    return preds, labels
end

function eval(model, testLoader::MonoGrammarPrototypeDataLoader, device)
    model.eval()
    preds, labels = torch.Tensor(), torch.Tensor()
    @pywith torch.no_grad() begin
        for (i, (io_emb, prog_emb, grammar_emb, y)) in enumerate(testLoader)
            (io_emb, prog_emb, grammar_emb, y) = io_emb.to(device), prog_emb.to(device), grammar_emb.to(device), y.to(device)
            output = model(io_emb, prog_emb, grammar_emb)

            preds = torch.cat([preds, output.to("cpu")], dim=0)
            labels = torch.cat([labels, y.to("cpu")], dim=0)

            GC.gc(false)
        end
        GC.gc()
    end
    return preds, labels
end