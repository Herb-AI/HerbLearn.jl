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
    ben_module::Module=Main,
    # num_partial_programs::Int=5,
    n_epochs::Int=100,
    batch_size::Int=20,
    model_size=nothing,
    data_dir::AbstractString="",
    semantic=false)

    # init computing device
    device = torch.device(ifelse(torch.cuda.is_available(), "cuda", "cpu"))
    println("Computing device: ", device)
    flush(stdout)

    # Generate programs
    data_ids = "$(semantic ? "sem" : "nonsem")_$(string(typeof(data_encoder)))_$(string(typeof(grammar_encoder)))"
    encoded_data_path = joinpath(data_dir, "encoded_$(data_ids)_data.pth")
    generated_problems_path = joinpath(data_dir, "generated_$(semantic ? "sem" : "nonsem")_problems.jls")
    if !isfile(generated_problems_path)
        println("Generating data...")
        gen_problems = generate_data(problem_grammar_pairs, start, num_programs; ben_module=ben_module, min_depth=2, max_depth=4)

        open(generated_problems_path, "w") do file
            serialize(file, gen_problems)
        end
    else
        println("Loading data...")
        gen_problems = open(deserialize, generated_problems_path)
    end

    println("Number of generated IOP problems: ", length(gen_problems))

    # println("generated_problems: ", gen_problems)

    # Merge grammars into all_rule_grammar; allows to mask rules present instead
    all_rule_grammar = deepcopy(@cfgrammar begin end)
    for pair in problem_grammar_pairs    
        merge_grammars_function_save!(all_rule_grammar, pair.grammar)
    end

    println("all_rule_grammar: $all_rule_grammar")

    for gen_problem in gen_problems
        println("gen_problems:")
        println("\t$(length(gen_problem.grammar.rules))")
        println("\t$(gen_problem.program)")
    end

    # Encode generated problems
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

    show_tensor(all_rule_grammar_encoding; prefix="Before")
    #l = torch.arange(length(all_rule_grammar.rules))
    #all_rule_grammar_encoding = torch.nn.functional.one_hot(l).float()
    #println("length(all_rule_grammar):\t$(length(all_rule_grammar.rules))\t$(all_rule_grammar_encoding.shape)")

    all_rule_grammar_encoding = all_but_the_top(all_rule_grammar_encoding; k_dim=7)
    show_tensor(all_rule_grammar_encoding; prefix="After")

    show_tensor(io_encodings; prefix="IO Before")
    # io_encodings = z_score_norm(io_encodings)
    io_encodings = all_but_the_top(io_encodings; k_dim=2)
    io_encodings = z_score_norm(io_encodings)
    show_tensor(io_encodings; prefix="IO After")

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

    # init dataloaders and models
    model_size = isnothing(model_size) ? [64,64] : model_size
    if !semantic
        train_dataloader = DataLoader(train_indices, io_encodings, label_data, batch_size, true, num_programs)
        test_dataloader = DataLoader(test_indices, io_encodings, label_data, batch_size, false, num_programs)

        grammar_length(g::AbstractGrammar) = length(g.rules)
        reference_length = grammar_length(gen_problems[1].grammar)
        @assert all(gp -> grammar_length(gp.grammar) == reference_length, gen_problems)

        model = DerivationPredNet(data_encoder.EMBED_DIM, reference_length, model_size)
    else 
        train_dataloader = PrototypeDataLoader(train_indices, io_encodings, grammar_indices, all_rule_grammar_encoding, label_data, batch_size, true, num_programs)
        test_dataloader = PrototypeDataLoader(test_indices, io_encodings, grammar_indices, all_rule_grammar_encoding, label_data, batch_size, false, num_programs)

        model = SemanticDerivationPredNet(data_encoder.EMBED_DIM, grammar_encoder.EMBED_DIM, model_size)

    end
   
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    weights = torch.tensor([1-label_sum/label_size, label_sum/label_size]) 
    println("weights:", weights)
    # loss_func = BCELoss_class_weighted(weights/weights.sum())
    loss_func = PairwiseRankingLoss()

    # do training within splits
    println("Starting training...")
    train!(model, train_dataloader, test_dataloader, n_epochs, optimizer, device, loss_func)

    return model
end


function train!(model, train_dataloader::DataLoader, valid_dataloader::DataLoader, n_epochs, optimizer, device, loss_func)
    for epoch in ProgressBar(1:n_epochs)
        accu_loss = 0
        num_samples = 0
        for (i, (io_emb, y)) in enumerate(train_dataloader)
            batch_size = io_emb.size(0)
            (io_emb, y) = io_emb.to(device), y.to(device)
            output = model(io_emb)

            loss = loss_func(output, y).sum()
            accu_loss += loss.item()
            num_samples += batch_size

            optimizer.zero_grad()
            loss.backward()
            optimizer.step(), 
            GC.gc(false)
        end
        accu_loss /= num_samples
        epoch % 20 == 0 && println("Epoch: $epoch\tLoss: $(accu_loss)\n")  # Probe performance

        if epoch % 20 == 0 
            train_preds, train_labels = eval(model, train_dataloader, device)
            valid_preds, valid_labels = eval(model, valid_dataloader, device)

            train_score = metrics.accuracy_score(train_labels.flatten(), train_preds.flatten()>0.5)
            valid_score = metrics.accuracy_score(valid_labels.flatten(), valid_preds.flatten()>0.5)
            println("\nAccuracy train/test:\t", train_score, "\t", valid_score)
            println("Train sanity check: ", train_preds.max(), train_preds.mean())
            println("Validation sanity check: ", valid_preds.max(), valid_preds.mean())

            #@TODO Add misranked_pairs after testing
        end

        flush(stdout)
        GC.gc()
    end
end

function train!(model, train_dataloader::PrototypeDataLoader, valid_dataloader::PrototypeDataLoader, n_epochs, optimizer, device, loss_func)
    all_rules_grammar_emb = train_dataloader.all_rules_grammar_emb.to(device)

    log_results(model, train_dataloader, device)
    log_results(model, valid_dataloader, device)

    for epoch in ProgressBar(1:n_epochs)
        accu_loss = 0
        num_samples = 0

        for (i, (io_emb, grammar_indices, y)) in enumerate(train_dataloader)
            batch_size = io_emb.size(0)
            (io_emb, grammar_indices, y) = io_emb.to(device), grammar_indices.to(device), y.to(device)

            grammar_indices = grammar_indices[1]
            @show grammar_indices

            grammar_emb = all_rules_grammar_emb.index_select(0, grammar_indices)

            output = model(io_emb, grammar_emb)
            # println("\toutput.shape: $(output.shape)")
            show_tensor(output; prefix="train!.output")
            y = y.squeeze()

            # println("output[1]:\t", output[1])

            # println("y:\t", y.float())

            loss = loss_func(output, y.float()).sum()
            accu_loss += loss.item()
            loss /= batch_size

            if epoch % 50 == 0
                println("pairs: $(y.sum(dim=1)),  $((1-y).sum(dim=1))")
                println("data_length(train_dataloader): $(data_length(train_dataloader))")

                println("grammar_indices: $grammar_indices")
                println("y: $y")

                masked_output = output.masked_fill(y == 0, float(10))
                min_val, min_pos = masked_output.min(dim=1)
                min_pos = grammar_indices.index_select(0, min_pos)
                masked_output = output.masked_fill(y == 1, float(-1))
                max_val, max_neg = masked_output.max(dim=1)
                max_neg = grammar_indices.index_select(0, max_neg)
                println("min_pos: $min_pos\tmin_val: $min_val")
                println("max_neg: $max_neg\tmax_val: $max_val")

                if false && sum(min_val < max_val).item() > 0
                    println("-------------------")
                    ind = (min_val < max_val).float().argmax().item() + 1 

                    println("ind: ", ind)

                    println("min_pos:", min_pos)
                    println("max_neg:", max_neg)
                    println("output[ind]:", output[ind])
                    println("y[ind]:", y[ind])
                    println("grammar_emb: ", grammar_emb)

                    println("ind: ", ind)
                    io_emb = io_emb[ind].view(1,-1)

                    _ = model(io_emb, grammar_emb)

                    error()
                end

            end
            
            num_samples += batch_size

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            GC.gc(false)
        end

        # if epoch == 2000 error() end

        @assert num_samples == data_length(train_dataloader)
        accu_loss /= num_samples
        epoch % 50 == 0 && println("Epoch: $epoch\tLoss: $(accu_loss)\n")

        if epoch % 50 == 0 
            log_results(model, train_dataloader, device)
            log_results(model, valid_dataloader, device)
        end

        flush(stdout)
        GC.gc()
    end

    log_results(model, train_dataloader, device)
    log_results(model, valid_dataloader, device)
end

function eval(model, test_dataloader::DataLoader, device)
    model.eval()
    preds, labels = torch.Tensor(), torch.Tensor()
    missed_pairs = 0
    total_pairs = 0
    @pywith torch.no_grad() begin
        for (i, (io_emb, y)) in enumerate(test_dataloader)
            (io_emb, y) = io_emb.to(device), y.to(device)
            output = model(io_emb)

            pairs = misranked_pairs(output, y)
            missed_pairs += pairs.sum()
            y = y.squeeze()
            total_pairs += (y.sum(dim=1) * (1-y).sum(dim=1)).sum()

            preds = torch.cat([preds, output.to("cpu")], dim=0)
            labels = torch.cat([labels, y.to("cpu")], dim=0)

            GC.gc(false)
        end
        GC.gc()
    end

    missed_pairs /= data_length(test_dataloader)
    total_pairs = total_pairs.item() / data_length(test_dataloader)
    path = "logging/$(data_length(test_dataloader))_output.csv"
    open(path, "a") do file
        # println(file, "$(missed_pairs.item()),$(preds.max().item()),$(preds.min().item())")
        println("$(data_length(test_dataloader)): Missed $(missed_pairs.item())/$(total_pairs) average pairs|\t$(preds.max().item()),$(preds.min().item()),$(preds.mean().item())")
    end

    return preds, labels
end

function eval(model, test_dataloader::PrototypeDataLoader, device)
    model.eval()
    all_rules_grammar_emb = test_dataloader.all_rules_grammar_emb.to(device)
    preds, labels = torch.Tensor(), torch.Tensor()
    missed_pairs = 0
    total_pairs = 0

    @pywith torch.no_grad() begin
        for (i, (io_emb, grammar_indices, y)) in enumerate(test_dataloader)
            (io_emb, grammar_indices, y) = io_emb.to(device), grammar_indices.to(device)[1], y.to(device)
            grammar_emb = all_rules_grammar_emb.index_select(0, grammar_indices)

            output = model(io_emb, grammar_emb)

            pairs = misranked_pairs(output, y)
            missed_pairs += pairs.sum()
            y = y.squeeze()
            total_pairs += (y.sum(dim=1) * (1-y).sum(dim=1)).sum()

            preds = torch.cat([preds, output.view(-1).to("cpu")], dim=0)
            labels = torch.cat([labels, y.view(-1).to("cpu")], dim=0)

            GC.gc(false)
        end
        GC.gc()
    end
    missed_pairs /= data_length(test_dataloader)
    total_pairs = total_pairs.item() / data_length(test_dataloader)
    path = "logging/$(data_length(test_dataloader))_output.csv"
    # open(path, "a") do file
        # println(file, "$(missed_pairs.item()),$(preds.max().item()),$(preds.min().item())")
    # end
    
    println("$(data_length(test_dataloader)): Missed $(missed_pairs.item())/$(total_pairs) average pairs|\t$(preds.max().item()),$(preds.min().item())")
    pos_mean = (preds * labels).sum()/labels.sum()
    neg_mean = (preds * (1-labels)).sum()/(1-labels).sum()
    println("\t pos/neg_preds.mean: $(pos_mean.item())/$(neg_mean.item())")
    return preds, labels 
end

function log_results(model, dataloader::AbstractDataLoader, device)
    preds, labels = eval(model, dataloader, device)
    println()

    # println("train_preds.shape: $(train_preds.shape), $(train_labels.shape)")
    # println("valid_preds.shape: $(valid_preds.shape), $(valid_labels.shape)")

    # top_k = 10

    # train_score = metrics.accuracy_score(train_labels.flatten(), mask_top_k(train_preds; top_k=top_k).flatten())
    # valid_score = metrics.accuracy_score(valid_labels.flatten(), mask_top_k(valid_preds; top_k=top_k).flatten())
    # println("\nAccuracy train/test:\t", train_score, "\t", valid_score)
    # println("Train sanity check: ", train_preds.max(), train_preds.mean())
    # println("Validation sanity check: ", valid_preds.max(), valid_preds.mean())

    # println("Misranked pairs: ")
    # println("\tTrain: $(misranked_pairs(train_preds, train_labels))\tValid: $(misranked_pairs(valid_preds, valid_labels))")
end