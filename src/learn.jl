"""
Pre-training procedure taking care of 
1. data-loading and preparation
2. training 
3. evaluation
of a NN model
"""
function pretrain(grammar::Grammar, problem::Problem, enumerator, start::Symbol, n_epochs::Int=100, batch_size::Int=20)
    # init computing device
    nocuda = true
    device = torch.device(ifelse(nocuda && torch.cuda.is_available(), "cuda", "cpu"))
    println("Computing device: ", device)

    # generate I/O + program examples
    max_depth::Int = 4
    num_programs::Int = 100
    
    iop_data = generate_data(grammar, problem, max_depth, num_programs, enumerator, start)
    println("Number of generated IOP samples: ", length(iop_data))

    # prune trivial examples
    iop_data = iop_data[10:length(iop_data)]

    # encode examples and programs
    io_data = [tup[1] for tup in iop_data]
    program_data = [tup[2] for tup in iop_data]

    io_emb_data, io_emb_dict = encode_IO_examples(io_data, "deepcoder", Dict(), 20, 10)
    program_emb_data = encode_programs(program_data, "none", 20)

    label_data = deepcoder_labels(program_data, grammar)

    # @TODO use one-hot encoding to describe whether derivation is in program or not

    # init train-validation split
    train_size = floor(Int, 0.8 * length(iop_data))
    test_size = length(iop_data) - train_size
    train_indices, test_indices = torch.utils.data.random_split(torch.arange(length(iop_data)), [train_size, test_size])

    # init dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        dataset=[(io_emb_data[i], program_emb_data[i], label_data[i]) for i ∈ train_indices],
        batch_size=batch_size,
        shuffle=true)
    # train_prog_dataloader = torch.utils.data.DataLoader(program_emb_data[train_indices])

    valid_io_dataloader = torch.utils.data.DataLoader(io_emb_data[test_indices])
    # valid_prog_dataloader = torch.utils.data.DataLoader(program_emb_data[test_indices])


    # init model
    println("Initializing prediction model...")
    model = DerivationPredNet(np.prod(io_emb_data.shape[2:length(io_emb_data.shape)]), program_emb_data.shape[2], length(grammar.rules)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    loss_func = torch.nn.BCELoss()

    # do training within splits
    println("Starting training...")
    @time train!(model, train_dataloader, n_epochs, optimizer, device, loss_func)

    # evaluate performance
end


function train!(model, trainLoader, n_epochs, optimizer, device, loss_func)
    for epoch in 1:n_epochs
        acc_loss = 0
        for (i, (io_emb, prog_emb, y)) in enumerate(trainLoader)
            (io_emb, prog_emb, y) = io_emb.to(device), prog_emb.to(device), y.to(device)
            output = model(io_emb, prog_emb)
            loss = loss_func(output, y)
            acc_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        end
        epoch % 10 == 0 && println("Epoch: $epoch\tLoss: $(acc_loss)")  # Probe performance once in a while
        GC.gc(false)
    end
end
