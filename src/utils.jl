using Random:randperm!

"""
@TODO write description
Adapted from the JuliaTorch implementation.
"""
struct DataLoader
    dataset::Tuple
    batchsize::Int
    shuffle::Bool
    indices::Vector{Int}
    n::Int
end

function DataLoader(dataset::NTuple{N,AbstractArray}; batchsize::Int=100, shuffle=false) where N
    println("Init DataLoader")
    l = last.(size.(dataset))
    n = first(l)
    all(n .== l) || throw(DimensionMismatch("All data should have the same length."))
    # n = last(size(first(dataset)))
    indices = collect(1:n)
    shuffle && randperm!(indices)
    DataLoader(dataset, batchsize, shuffle, indices, n)
end

DataLoader(dataset...; batchsize::Int=100, shuffle=false) =
    DataLoader(dataset, batchsize=batchsize, shuffle=shuffle)

function Base.iterate(it::DataLoader, start=1)
    if start > it.n
        it.shuffle && randperm!(it.indices)
        return nothing
    end
    nextstart = min(start + it.batchsize, it.n + 1)
    i = it.indices[start:nextstart-1]
    element = Tuple(copy(selectdim(x, ndims(x), i)) for x in it.dataset)
    return element, nextstart
end

Base.length(it::DataLoader) = it.n
Base.eltype(it::DataLoader) = typeof(it.dataset)

# function batchselect(x::AbstractArray, i)
#     inds = CartesianIndices(size(x)[1:end-1])
#     x[inds, i]
# end

function Base.show(io::IO, it::DataLoader)
    print(io, "DataLoader(dataset size = $(it.n)")
    print(io, ", batchsize = $(it.batchsize), shuffle = $(it.shuffle)")
    print(io, ")")
end


"""

"""
const PYLOCK = Ref{ReentrantLock}()
PYLOCK[] = ReentrantLock()

pylock(f::Function) = Base.lock(PYLOCK[]) do
    prev_gc = GC.enable(false)
    try 
        return f()
    finally
        GC.enable(prev_gc) # recover previous state
    end
end