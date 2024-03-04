"""
Pre-training procedure taking care of 
1. data-loading and preparation
2. training 
3. evaluation
of a NN model
"""
function pretrain_heuristic(
    grammar::AbstractGrammar,
    problems::Vector{Problem{Vector{IOExample}}};
    data_encoder::AbstractIOEncoder,
    program_encoder::AbstractProgramEncoder,
    start::Symbol=:Start,
    num_programs::Int=100,
    num_partial_programs::Int=5,
    n_epochs::Int=100,
    batch_size::Int=20,
    model_dir::AbstractString="")

    # init computing device
    device = torch.device(ifelse(torch.cuda.is_available(), "cuda", "cpu"))
    println("Computing device: ", device)
    flush(stdout)
    
    println("Generating data...")
    gen_problems = generate_data(grammar, problems, start, num_programs, num_partial_programs; max_depth=4)

    println("Number of generated IOP problems: ", length(gen_problems))
    flush(stdout)

    println("Encoding data...")
    io_encodings = encode_IO_examples(gen_problems, data_encoder)
    println("io_encodings", io_encodings.size())
    program_encodings = encode_programs(gen_problems, program_encoder)
    label_data = label_encoding(gen_problems)

    label_size = sum([reduce(*, vec.size()) for vec in label_data])
    label_sum = sum([vec.sum() for vec in label_data])
    println("label_data.size(): ", label_size)
    println("label_data.sum(): ", label_sum)
    println("label_data.ratio: ", label_sum/label_size)

    # init train-validation split
    train_size = floor(Int, 0.8 * length(problems))
    test_size = length(problems) - train_size
    train_indices, test_indices = torch.utils.data.random_split(torch.arange(length(problems)), [train_size, test_size])

    # init dataloaders
    train_dataloader = DataLoader(train_indices, io_encodings, program_encodings, label_data, batch_size, true)
   
    test_dataloader = DataLoader(test_indices, io_encodings, program_encodings, label_data, batch_size, false)
   
    # init model
    println("Initializing prediction model...")
    model = DerivationPredNet(data_encoder.EMBED_DIM, program_encoder.EMBED_DIM, length(grammar.rules), [32])

    # grammar_embeddings = program_encoder([RuleNode(i) for i in 1:length(grammar.rules)], program_encoder).view(length(grammar.rules), -1)
    # model = SemanticDerivationPredNet(data_encoder.EMBED_DIM, program_encoder.embed_dim, grammar_embeddings, [64, 64])

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


function train!(model, train_dataloader, valid_dataloader, n_epochs, optimizer, device, loss_func)
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

        if epoch % 100 == 0 
            train_preds, train_labels = eval(model, train_dataloader, device)
            valid_preds, valid_labels = eval(model, valid_dataloader, device)
            println("torch.mean(labels):", torch.mean(valid_labels), torch.mean(valid_preds))

            train_score = metrics.accuracy_score(train_preds.flatten()>0.5, train_labels.flatten())
            valid_score = metrics.accuracy_score(valid_preds.flatten()>0.5, valid_labels.flatten())
            println("Accuracy train/test:\t", train_score, "\t", valid_score)
        end

        GC.gc()
    end
end

function eval(model, testLoader, device)
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


"""

"""
function train_program_autoencoder(grammar::AbstractGrammar, programs::Vector{RuleNode}, encoder::AbstractProgramEncoder, decoder::AbstractProgramDecoder)
    # Step 1: Prepare the program decoding training data
    training_data = prepare_decoding_training_data(grammar, programs)
    
    # Step 2: Encode the programs
    encoded_programs = encode_programs(training_data[:programs], encoder)
    
    # Step 3: Train the decoder policy
    train_decoder_policy(encoded_programs, training_data[:possible_derivations], decoder)
end