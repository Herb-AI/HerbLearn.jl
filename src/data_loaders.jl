"""
DataLoader(indices, io_encodings, program_encodings, label_data, batch_size::Int, shuffle::Bool) = torch.utils.data.DataLoader(
    dataset=[(io_encodings[i], program_encodings[i][j], label_data[i][j]) 
    for i ∈ indices 
        for j in 1:length(program_encodings[i])],
    batch_size=batch_size,
    shuffle=shuffle)
"""

function DataLoader(indices, io_encodings, program_encodings, label_data, batch_size::Int, shuffle::Bool)
    dataset = []
    for i in indices
        i = i.item() + 1 # PyTorch tensor to Julia index
        for j in 0:length(program_encodings[i])-1 # PyTorch indices
            push!(dataset, (io_encodings[i], program_encodings[i][j], label_data[i][j]))
        end
    end
    
    # Create DataLoader
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
end

PrototypeDataLoader(indices, io_encodings, program_encodings, grammar_encodings, label_data, batch_size::Int, shuffle::Bool) = torch.utils.data.DataLoader(
    dataset=[(io_encodings[i], program_encodings[i][j], grammar_encodings[i], label_data[i][j]) 
    for i ∈ indices 
        for j in 1:length(program_encodings[i])],
    batch_size=batch_size,
    shuffle=shuffle)


