using Random 

abstract type AbstractDataLoader end

struct DataLoader <: AbstractDataLoader
    loader::PyObject 

    function DataLoader(indices, io_encodings, label_data, batch_size::Int, shuffle::Bool, num_programs)
        dataset = []
        for i in indices
            i = i.item() + 1 # PyTorch tensor to Julia index
            range = collect((i-1) * num_programs+1:(i) * num_programs)
            for k in range
                push!(dataset, (io_encodings[k], torch.unsqueeze(label_data[k],0)))
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

Base.length(iter::DataLoader) = length(iter.loader)
data_length(iter::DataLoader) = length(iter)

function Base.iterate(dataloader::DataLoader, state=nothing)
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

struct PrototypeDataLoader <: AbstractDataLoader
    problem_dataloaders::Vector{PyObject}
    all_rules_grammar_emb::PyObject
    shuffle::Bool

    function PrototypeDataLoader(indices, io_encodings, grammar_indices, all_rules_grammar_encoding, label_data, batch_size::Int, shuffle::Bool, num_programs)
        problem_dataloaders = [] 
        for i in indices
            i = i.item() + 1 # PyTorch tensor to Julia index
            range = collect((i-1) * num_programs+1:(i) * num_programs)
            dataset = []
            for k in range
                push!(dataset, (io_encodings[k], grammar_indices[k], label_data[k]))
            end
            push!(problem_dataloaders, torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle))
        end

        # Create DataLoader
        new(problem_dataloaders, all_rules_grammar_encoding, shuffle)
    end
end

Base.length(iter::PrototypeDataLoader) = sum([length(loader) for loader in iter.problem_dataloaders])
data_length(iter::PrototypeDataLoader) = sum([length(loader.dataset) for loader in iter.problem_dataloaders])


struct DataLoaderIteratorState
    current_indices::Vector{Integer}
    current_states::Vector{Any}
end

function DataLoaderIteratorState(iter::PrototypeDataLoader)
    indices = []

    # Creates list of loader indices in which they should be enumerated
    for (i, loader) in enumerate(iter.problem_dataloaders)
        append!(indices, [i for _ in 1:length(loader)])
    end

    indices = iter.shuffle ? Random.shuffle(indices) : indices
    states = [nothing for _ in 1:length(iter.problem_dataloaders)]
    return DataLoaderIteratorState(indices, states)
end

function Base.iterate(dataloader::PrototypeDataLoader, state::Union{DataLoaderIteratorState, Nothing}=nothing)

    if state === nothing
        # Start from the first dataloader
        state = DataLoaderIteratorState(dataloader)
    end

    # Stop if indices-to-be-enumerated is empty
    if isempty(state.current_indices)
        return nothing
    end

    loader_index = popfirst!(state.current_indices)

    # was sub-iterator iterated already?
    if isnothing(state.current_states[loader_index])
        # Init iteration, and save state to state list
        first_value, first_state = iterate(dataloader.problem_dataloaders[loader_index])

        state.current_states[loader_index] = first_state

        return first_value, state
    else 
        # Loader previous loader state, iterate, and update state
        previous_state = state.current_states[loader_index]
        loader_value, loader_state = iterate(dataloader.problem_dataloaders[loader_index], previous_state)

        state.current_states[loader_index] = loader_state

        return loader_value, state
    end 
end
