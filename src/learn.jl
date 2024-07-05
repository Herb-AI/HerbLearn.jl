"""
Pre-training procedure taking care of 
1. data-loading and preparation
2. training 
3. evaluation
of a NN model
"""
function pretrain_heuristic(
    problem_grammar_pairs::Vector{ProblemGrammarPair};
    data_encoder::AbstractIOEncoder,
    grammar_encoder::AbstractProgramEncoder,
    start::Symbol=:Start,
    num_programs::Int=100,
    # num_partial_programs::Int=5,
    n_epochs::Int=100,
    batch_size::Int=20,
    data_dir::AbstractString="",
    semantic=false)

    # init computing device
    device = torch.device(ifelse(torch.cuda.is_available(), "cuda", "cpu"))
    println("Computing device: ", device)
    flush(stdout)

    all_rule_grammar = deepcopy(@cfgrammar begin end)
    for pair in problem_grammar_pairs    
        merge_grammars_function_save!(all_rule_grammar, pair.grammar)
    end

    data_ids = "$(semantic ? "sem" : "nonsem")_$(string(typeof(data_encoder)))_$(string(typeof(grammar_encoder)))"
    encoded_data_path = joinpath(data_dir, "encoded_$(data_ids)_data.pth")
    generated_problems_path = joinpath(data_dir, "generated_$(semantic ? "sem" : "nonsem")_problems.jls")
    if !isfile(generated_problems_path)
        println("Generating data...")
        gen_problems = generate_data(problem_grammar_pairs, start, num_programs; min_depth=2, max_depth=6)

        open(generated_problems_path, "w") do file
            serialize(file, gen_problems)
        end
    else
        println("Loading data...")
        gen_problems = open(deserialize, generated_problems_path)
        #assert to check whether parts per program matches with loaded data size
    end

    println("Number of generated IOP problems: ", length(gen_problems))

    if isempty(data_dir) || !isfile(encoded_data_path)
        println("Encoding data...")
        io_encodings = encode_IO_examples(gen_problems, data_encoder)
        println("Encoding labels...")
        # label_data = label_encoding(gen_problems)
        label_data = deepcoder_labels(gen_problems)

        println("Encoding grammars...")
        grammar_indices = get_rule_indices(gen_problems, all_rule_grammar)
        all_rule_grammar_encoding = semantic ? encode_grammar(all_rule_grammar, grammar_encoder) : nothing
        println("Done.")
        ""
        data_to_save = Dict(
            "io_encodings" => io_encodings,
            "label_data" => label_data,
            "grammar_indices" => grammar_indices, 
            "all_rule_grammar_encoding" => all_rule_grammar_encoding
        )
        torch.save(data_to_save, encoded_data_path)
        
    else
        loaded_data = torch.load(encoded_data_path)
        io_encodings = loaded_data["io_encodings"]
        label_data = loaded_data["label_data"]
        grammar_indices = loaded_data["grammar_indices"]
        all_rule_grammar_encoding = semantic ? loaded_data["all_rule_grammar_encoding"] : nothing

    end

    label_size = sum([reduce(*, vec.size()) for vec in label_data])
    label_sum = sum([vec.sum() for vec in label_data])
    println("label_data.size(): ", label_size)
    println("label_data.sum(): ", label_sum)
    println("label_data.ratio: ", label_sum/label_size)

    # init train-validation split
    num_problems = Int(length(gen_problems) / num_programs)
    train_size = floor(Int, 0.8 * num_problems)
    test_size = num_problems - train_size
    train_indices, test_indices = torch.utils.data.random_split(torch.arange(num_problems), [train_size, test_size])

    # init dataloaders
    if !semantic
        train_dataloader = DataLoader(train_indices, io_encodings, label_data, batch_size, true, num_programs)
        test_dataloader = DataLoader(test_indices, io_encodings, label_data, batch_size, false, num_programs)

        grammar_length(g::AbstractGrammar) = length(g.rules)
        reference_length = grammar_length(gen_problems[1].grammar)
        @assert all(gp -> grammar_length(gp.grammar) == reference_length, gen_problems)

        model = DerivationPredNet(data_encoder.EMBED_DIM, reference_length, [64, 64])
    else 
        train_dataloader = nothing
        test_dataloader = nothing
        
        train_dataloader = PrototypeDataLoader(train_indices, io_encodings, grammar_indices, all_rule_grammar_encoding, label_data, batch_size, true, num_programs)
        test_dataloader = PrototypeDataLoader(test_indices, io_encodings, grammar_indices, all_rule_grammar_encoding, label_data, batch_size, false, num_programs)

        model = SemanticDerivationPredNet(data_encoder.EMBED_DIM, grammar_encoder.EMBED_DIM, [64, 64])
    end
   
    model = torch.nn.DataParallel(model).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    weights = torch.tensor([label_sum/label_size, 1-label_sum/label_size]) 
    println("weights:", weights)
    loss_func = BCELoss_class_weighted(weights/weights.sum())

    # do training within splits
    println("Starting training...")
    train!(model, train_dataloader, test_dataloader, n_epochs, optimizer, device, loss_func)

    return model
end


function train!(model, train_dataloader::DataLoader, valid_dataloader::DataLoader, n_epochs, optimizer, device, loss_func)
    for epoch in 1:n_epochs
        accu_loss = 0
        for (i, (io_emb, y)) in enumerate(train_dataloader)
            batch_size = io_emb.size(0)
            (io_emb, y) = io_emb.to(device), y.to(device)
            output = model(io_emb)
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
            valid_score = metrics.accuracy_score(valid_labels.flatten(), valid_preds.flatten()>0.5)
            println("Accuracy train/test:\t", train_score, "\t", valid_score)
        end

        flush(stdout)
        GC.gc()
    end
end

function train!(model, train_dataloader::PrototypeDataLoader, valid_dataloader::PrototypeDataLoader, n_epochs, optimizer, device, loss_func)
    all_rules_grammar_emb = train_dataloader.all_rules_grammar_emb.to(device)
    for epoch in ProgressBar(1:n_epochs)
        accu_loss = 0
        for (i, (io_emb, grammar_indices, y)) in enumerate(train_dataloader)
            batch_size = io_emb.size(0)
            (io_emb, grammar_indices, y) = io_emb.to(device), grammar_indices.to(device)[1], y.to(device)
            grammar_emb = all_rules_grammar_emb.index_select(0, grammar_indices)

            output = model(io_emb, grammar_emb)
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


function eval(model, test_dataloader::DataLoader, device)
    model.eval()
    preds, labels = torch.Tensor(), torch.Tensor()
    @pywith torch.no_grad() begin
        for (i, (io_emb, y)) in enumerate(test_dataloader)
            (io_emb, y) = io_emb.to(device), y.to(device)
            output = model(io_emb)

            preds = torch.cat([preds, output.to("cpu")], dim=0)
            labels = torch.cat([labels, y.to("cpu")], dim=0)

            GC.gc(false)
        end
        GC.gc()
    end
    return preds, labels
end

function eval(model, test_dataloader::PrototypeDataLoader, device)
    model.eval()
    all_rules_grammar_emb = test_dataloader.all_rules_grammar_emb.to(device)
    preds, labels = torch.Tensor(), torch.Tensor()
    @pywith torch.no_grad() begin
        for (i, (io_emb, grammar_indices, y)) in enumerate(test_dataloader)
            (io_emb, grammar_indices, y) = io_emb.to(device), grammar_indices.to(device)[1], y.to(device)
            grammar_emb = all_rules_grammar_emb.index_select(0, grammar_indices)

            output = model(io_emb, grammar_emb)

            preds = torch.cat([preds, output.to("cpu")], dim=0)
            labels = torch.cat([labels, y.to("cpu")], dim=0)

            GC.gc(false)
        end
        GC.gc()
    end
    return preds, labels
end