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
        new(problem_dataloaders, all_rules_grammar_encoding)
    end
end

Base.length(iter::PrototypeDataLoader) = sum([length(loader) for loader in iter.problem_dataloaders])


struct DataLoaderIteratorState
    current_loader_index::Int
    current_loader_state::Any
end

function Base.iterate(dataloader::PrototypeDataLoader, state::Union{DataLoaderIteratorState, Nothing}=nothing)

    if state === nothing
        # Start from the first dataloader
        first_loader = dataloader.problem_dataloaders[1]
        first_value, first_state = iterate(first_loader)
        return first_value, DataLoaderIteratorState(1, first_state)
    else
        if !ispynull(state.current_loader_state[1])
            # Continue from the current dataloader
            next_value, next_loader_state = iterate(dataloader.problem_dataloaders[state.current_loader_index], state.current_loader_state)
            # More data available in the current loader

            return next_value, DataLoaderIteratorState(state.current_loader_index, next_loader_state)
        elseif state.current_loader_index < length(dataloader.problem_dataloaders)
            # Move to the next loader
            next_loader_index = state.current_loader_index + 1
            next_loader = dataloader.problem_dataloaders[next_loader_index]
            first_value, first_state = iterate(next_loader)
            return first_value, DataLoaderIteratorState(next_loader_index, first_state)
        else
            # No more dataloaders left
            return nothing
        end
    end 
end