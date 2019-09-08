function gather(input::Array{T,N}, index::Array{<:Integer,N}, dims::Integer;
                out::Array{T,N}=similar(index, T)) where {T,N}
    @assert dims <= N "Specified dimensions must lower or equal to the rank of input matrix."

    @inbounds for x = CartesianIndices(out)
        ind = Tuple(x)
        tup = collect(ind)
        tup[dims] = index[ind...]
        out[ind...] = input[tup...]
    end
    return out
end
