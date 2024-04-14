abstract type AbstractDataLoader end

struct DataLoader <: AbstractDataLoader
    loader::PyObject 

    function DataLoader(indices, io_encodings, program_encodings, label_data, batch_size::Int, shuffle::Bool, num_programs)
        dataset = []
        for i in indices
            i = i.item() + 1 # PyTorch tensor to Julia index
            range = collect((i-1) * num_programs+1:(i) * num_programs)
            for k in range
                for j in 0:length(program_encodings[i])-1 # PyTorch indices
                    push!(dataset, (io_encodings[k], program_encodings[k][j], torch.unsqueeze(label_data[k][j],0)))
                end
            end
        end
        
        # Create DataLoader
        new(torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle
        ))
    end
end 

struct PrototypeDataLoader <: AbstractDataLoader 
    indices::Vector{Int}  # Store the current order of indices to access the dataset

    io_data # num_problems * num_programs x lat_dim * 2
    program_data  # num_programs x num_sampled_programs x lat_dim
    grammar_data  # num_derivations x lat_dim
    label_data  # num_programs x num_sampled_programs x num_derivations

    shuffle::Bool
    num_programs::Int

    function PrototypeDataLoader(indices, args...)
        new([i.item() for i in indices], args...)
    end
end

struct MonoGrammarPrototypeDataLoader <: AbstractDataLoader
    loader::PyObject

    function MonoGrammarPrototypeDataLoader(indices, io_encodings, program_encodings, grammar_encodings, label_data, batch_size::Int, shuffle::Bool, num_programs)
        @assert all([enc.shape == grammar_encodings[1].shape for enc in grammar_encodings])
        dataset = []
        for i in indices
            i = i.item() + 1 # PyTorch tensor to Julia index
            range = collect((i-1) * num_programs+1:(i) * num_programs)
            for k in range
                for j in 0:length(program_encodings[i])-1 # PyTorch indices
                    push!(dataset, (io_encodings[k], program_encodings[k][j], grammar_encodings[k], label_data[k][j]))
                end
            end
        end

        println("length dataset: $(length(dataset))")
        
        # Create DataLoader
        new(torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle
        ))
    end
end


function Base.iterate(dataloader::PrototypeDataLoader, state=1)
    if state > length(dataloader.indices)
        return nothing
    end
    i =  dataloader.indices[state] + 1  # Convert PyTorch tensor to Julia index
    range = collect((i-1) * dataloader.num_programs+1:(i) * dataloader.num_programs)
    println("Range: $i $range")

    num_sampled_programs = length(dataloader.program_data[range[1]-1])
    println("num_sampled_programs: $num_sampled_programs")
    io_batch = dataloader.io_data[torch.tensor(range)-1].repeat_interleave(num_sampled_programs)
    println("io_batch $(io_batch.shape)")

    program_batch = dataloader.program_data[range]
    program_batch = torch.cat(program_batch, dim=0)
    println("program_batch $(program_batch.shape)")

    grammar_data = dataloader.grammar_data[i]
    println("grammar_data: $(grammar_data.shape)")

    label_data = dataloader.label_data[range]
    label_data = torch.cat(label_data, dim=0)
    println("label_data: $(label_data.shape)")

    error()
    next_item = "bumm"
    return next_item, state + 1
end

# Implement the length function for PrototypeDataLoader
Base.length(dataloader::PrototypeDataLoader) = length(dataloader.indices)

Base.length(iter::AbstractDataLoader) = length(iter.loader)

function Base.iterate(dataloader::AbstractDataLoader, state=nothing)
    py_iter = pyimport("builtins").iter  # Import the Python iter function

    if state === nothing
        iter_obj = py_iter(dataloader.loader)  # Create a Python iterator for the py_loader
        next_item = pybuiltin("next")(iter_obj, nothing)
        if next_item === nothing
            return nothing
        else
            return (next_item, iter_obj)
        end
    else
        next_item = pybuiltin("next")(state, nothing)
        if next_item === nothing
            return nothing
        else
            return (next_item, state)
        end
    end
end