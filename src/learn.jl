"""
Pre-training procedure taking care of 
1. data-loading and preparation
2. training 
3. evaluation
of a NN model
"""
function pretrain(grammar::Grammar,
    problem::Problem,
    enumerator::Function,
    start::Symbol,
    data_encoder::IOEncoder,
    program_encoder::ProgramEncoder,
    n_epochs::Int=100,
    batch_size::Int=20,
    model_dir::AbstractString="")

    # init computing device
    nocuda = true
    device = torch.device(ifelse(nocuda && torch.cuda.is_available(), "cuda", "cpu"))
    println("Computing device: ", device)

    # generate I/O + program examples
    max_depth::Int = 7
    num_programs::Int = 1000
    
    iop_data = generate_data(grammar, problem, start, num_programs, enumerator, max_depth=max_depth)

    println("Number of generated IOP samples: ", length(iop_data))

    # prune trivial examples
    iop_data = iop_data[10:length(iop_data)]

    # encode examples and programs
    io_data = [tup[1] for tup in iop_data]
    program_data = [tup[2] for tup in iop_data]

    io_emb_data = encode_IO_examples(io_data, data_encoder, Dict())
    program_emb_data = encode_programs(program_data, program_encoder, 1)

    label_data = deepcoder_labels(program_data, grammar)

    println("label_data.size(): ", label_data.size())
    println("label_data.sum(): ", label_data.sum())
    println("label_data.ratio: ", label_data.sum()/torch.prod(torch.tensor(label_data.size())))

    # init train-validation split
    train_size = floor(Int, 0.8 * length(iop_data))
    test_size = length(iop_data) - train_size
    train_indices, test_indices = torch.utils.data.random_split(torch.arange(length(iop_data)), [train_size, test_size])

    # init dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        dataset=[(io_emb_data[i], program_emb_data[i], label_data[i]) for i ∈ train_indices],
        batch_size=batch_size,
        shuffle=true)

    test_dataloader = torch.utils.data.DataLoader(
        dataset=[(io_emb_data[i], program_emb_data[i], label_data[i]) for i ∈ test_indices],
        batch_size=batch_size,
        shuffle=false)

    # init model
    println("Initializing prediction model...")
    model = DerivationPredNet(np.prod(io_emb_data.shape[2:length(io_emb_data.shape)]), program_emb_data.shape[2], length(grammar.rules)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    loss_func = BCELoss_class_weighted(torch.ones((2,))/2)

    # do training within splits
    println("Starting training...")
    @time train!(model, train_dataloader, test_dataloader,n_epochs, optimizer, device, loss_func)

    # evaluate performance
    @time preds, labels = eval(model, test_dataloader, device)
    println("torch.mean(labels):", torch.mean(labels))
    println("torch.mean(preds):", torch.mean(preds))
    score = metrics.accuracy_score(preds, labels)
    println("Accuracy score:\t", score)

    # Save model to disk
    torch.save(model.state_dict(), string(model_dir, "DeepCoder_model.th"))

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

        if epoch % 500 == 0 
            train_preds, train_labels = eval(model, train_dataloader, device)
            valid_preds, valid_labels = eval(model, valid_dataloader, device)
            println("torch.mean(labels):", torch.mean(valid_labels), torch.mean(valid_preds))
            
            train_score = metrics.accuracy_score(train_preds, train_labels)
            valid_score = metrics.accuracy_score(valid_preds, valid_labels)
            println("Accuracy train/test:\t", train_score, "\t", valid_score)
        end

        GC.gc(false)
    end
end

function eval(model, testLoader, device)
    model.eval()
    preds, labels = torch.Tensor(), torch.Tensor()
    @pywith torch.no_grad() begin
        for (i, (io_emb, prog_emb, y)) in enumerate(testLoader)
            (io_emb, prog_emb, y) = io_emb.to(device), prog_emb.to(device), y.to(device)
            output = torch.round(model(io_emb, prog_emb))

            preds = torch.cat([preds, output.to("cpu")], dim=0)
            labels = torch.cat([labels, y.to("cpu")], dim=0)

            i % 10 == 0 && GC.gc(false)
        end
        GC.gc(false)
    end
    return preds, labels
end