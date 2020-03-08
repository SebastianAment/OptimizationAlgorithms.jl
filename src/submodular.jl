########################### Submodular Minimization ############################
struct Submodular{T, F, X, D} <: Update{T}
    f::F # set-valued input function
    x::X # set to choose from
    δ::D # bound on improvement of adding a certain element
    n::Int # maximum set cardinality
    function Submodular(f, x::AbstractVector, n::Int)
        δ = fill(-Inf, length(x))
        new{eltype(x), typeof(f), typeof(x), typeof(δ)}(f, x, δ, n)
    end
end
# TODO: parallelization, especially for first pass
function update!(S::Submodular, x::Union{AbstractVector, AbstractSet})
    length(x) ≥ S.n ? return x : nothing # only add if we haven't hit maximum cardinality
    f, δ, grid = S.f, S.δ, S.x
    ind = sortperm(δ) # sort grid according to increasing δ (decreasing in magnitude)
    permute!(δ, ind)
    permute!(grid, ind)
    fx = f(x)
    min_i = 0. # keeps track of smallest (i.e. greatest magnitude of) δ
    for (i, xs) in enumerate(grid) # parallelizing here does not seem to benefit much
        if min_i > δ[i] # if previously recorded δ is smaller (larger in magnitude) than the current minimum
            push!(x, xs) # already minimized allocation for (x, xs)?
            δ[i] = f(x) - fx
            min_i = min(min_i, δ[i])
            pop!(x)
        end
    end
    push!(x, grid[argmin(δ)]) # choose next point greedily
end

# Old code:
# function submodular_minimization(x::AbstractArray{T}, grid::AbstractArray{T},
#                             objective::Function,
#                             num_next::Int = 1) where {T}
#     num_next < 1 && error("num_next < 1")
#
#     δ = fill(-Inf, length(grid))
#     base_objective = objective(x)
#     #
#     Threads.@threads for i in 1:length(grid)
#         δ[i] = objective(vcat(x, grid[i])) - base_objective
#     end
#     next = grid[argmin(δ)] # choose point which reduces the objective most
#
#     if num_next == 1
#         return [next], δ
#     else
#         recursive_next, δ = submodular_minimization(vcat(x, next), grid,
#                                                 objective, num_next-1, δ)
#         return vcat(next, recursive_next), δ # could instead return, and spawn task to complete the recursive step
#     end
# end
#
# # posterior uncertainty sampling for batch learning of optical data
# function submodular_minimization(x::AbstractArray{T}, grid::AbstractArray{T},
#                                 objective::Function, num_next::Int,
#                                 δ::Array{<:Real}) where {T}
#     length(δ) ≠ length(grid) && error("length(δ) ≠ length(grid)")
#     num_next < 1 && error("num_next < 1")
#     # sort grid according to increasing δ (decreasing in magnitude)
#     ind = sortperm(δ)
#     δ = δ[ind]
#     grid = grid[ind]
#
#     base_objective = objective(x)
#     min_i = 0 # keeps track of smallest (i.e. greatest magnitude of) δ
#     for (i, xs) in enumerate(grid) # parallelizing here does not seem to benefit much
#         if min_i > δ[i] # if previously recorded δ is smaller (larger in magnitude) than the current minimum
#             δ[i] = objective(vcat(x, xs)) - base_objective
#             min_i = min(min_i, δ[i])
#         end
#     end
#     next = grid[argmin(δ)] # choose the point which reduces posterior uncertainty most
#
#     if num_next == 1
#         return [next], δ
#     else
#         recursive_next, δ = submodular_minimization(vcat(x, next), grid,
#                                                     objective, num_next-1, δ)
#         return vcat(next, recursive_next), δ
#     end
# end
