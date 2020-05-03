abstract type AbstractPursuit{T} <: Update{T} end
# AbstractPursuit
@inline function _residual!(P, x::AbstractVector) # 0 alloc
    copyto!(P.r, P.b)
    mul!(P.r, P.A, x, -1, 1)
end
@inline function _factorize!(P, x::AbstractVector)
    n = size(P.A, 1)
    k = nnz(x)
    Ai = reshape(@view(P.Ai[1:n*k]), n, k)
    @. Ai = @view(P.A[:, x.nzind])
    return qr!(Ai) # this still allocates a little memory
end
@inline function _solve!(P, x::AbstractVector)
    F = _factorize!(P, x)
    ldiv!(x.nzval, F, P.b)     # optimize all active atoms
    dropzeros!(x)
end

###################### Orthogonal Matching Pursuit #############################
# TODO: preconditioning, non-negativity constraint -> NNLS.jl
using SparseArrays
struct OrthogonalMatchingPursuit{T, AT<:AbstractMatrix{T},
                                    B<:AbstractVector{T}} <: Update{T}
    A::AT
    b::B
    k::Int # maximum number of non-zeros

    # temporary storage
    r::B # residual
    Ar::B # inner products between measurement matrix and residual
    Ai::AT # space for A[:, x.nzind] and its qr factorization
end
const OMP = OrthogonalMatchingPursuit
function OMP(A::AbstractMatrix, b::AbstractVector, k::Integer)
    n, m = size(A)
    T = eltype(A)
    r, Ar = zeros(T, n), zeros(T, m)
    Ai = zeros(T, (n, k))
    OMP(A, b, k, r, Ar, Ai)
end

function update!(P::OMP, x::AbstractVector = spzeros(size(P.A, 2)))
    nnz(x) ≥ P.k && return x # return if the maximum number of non-zeros was reached
    _residual!(P, x)
    i = _ompindex!(P)
    x[i] = NaN # add non-zero index to x
    _solve!(P, x) # optimize all active atoms
    return x
end

@inline function _ompindex!(P::OMP) # 0 alloc
    mul!(P.Ar, P.A', P.r)
    @. P.Ar = abs(P.Ar)
    argmax(P.Ar)
end

# calculates k-sparse approximation to Ax = b via orthogonal matching pursuit
function omp(A::AbstractMatrix, b::AbstractVector, k::Int)
    P! = OMP(A, b, k)
    x = spzeros(size(A, 2))
    for i in 1:k
        P!(x)
    end
    return x
end

################################## Subspace Pursuit ############################
struct SubspacePursuit{T, AT<:AbstractMatrix{T}, B<:AbstractVector{T}} <: Update{T}
    A::AT
    b::B
    k::Int # maximum number of non-zeros

    # temporary storage
    r::B # residual
    Ar::B # inner products between measurement matrix and residual
    Ai::AT # space for A[:, x.nzind] and its qr factorization
end
const SSP = SubspacePursuit

function SSP(A::AbstractMatrix, b::AbstractVector, k::Integer)
    n, m = size(A)
    T = eltype(A)
    r, Ar = zeros(T, n), zeros(T, m)
    Ai = zeros(T, (n, 2k))
    SSP(A, b, k, r, Ar, Ai)
end

# returns indices of k atoms with largest inner products with residual
# TODO: need threshold for P.Ar
@inline function _sspindex!(P::SSP)
    mul!(P.Ar, P.A', P.r)
    @. P.Ar = abs(P.Ar)
    partialsortperm(P.Ar, 1:P.k, rev = true)
end

# TODO: could pre-allocate nz arrays to be of length 2K
function update!(P::SSP, x::AbstractVector = spzeros(size(P.A, 2)))
    if nnz(x) == 0
        return x .= omp(P.A, P.b, P.k) # first iteration is via omp
    end
    nnz(x) == P.k || throw("nnz(x) = $(nnz(x)) ≠ $(P.k) = k")

    _residual!(P, x)
    i = _sspindex!(P)
    @. x[i] = NaN # add non-zero index to x

    _solve!(P, x) # optimize all active atoms

    i = partialsortperm(abs.(x.nzval), 1:P.k) # find k smallest atoms
    @. x.nzval[i] = 0
    dropzeros!(x)

    _solve!(P, x) # optimize all active atoms
    return x
end

# calculates k-sparse approximation to Ax = b via orthogonal matching pursuit
function ssp(A::AbstractMatrix, b::AbstractVector, k::Int; iter = 6)
    P! = SSP(A, b, k)
    x = spzeros(size(A, 2))
    for i in 1:iter
        P!(x)
    end
    return x
end


# function precondition_omp()
#     A, μ, σ = center!(A)
#     b, μ, σ = center!(b)
#     A .*= normal
#     x ./= normal
# end
#
# # TODO: this should be in util somewhere
# using StatsBase: mean, std
# # makes x mean zero along dimensions dims
# function center!(x::AbstractVecOrMat, ε = 1e-6; dims = :)
#     μ, σ = mean(x, dims = dims), std(x, dims = dims)
#     @. x = (x - μ) / σ
#     # @. x = (x - (1-ε)*μ) / σ
#     return x, μ, σ
# end
# Base.inv(::typeof(center!)) = uncenter
# function uncenter(x::AbstractVecOrMat, μ, σ)
#     @. x = x * σ + μ
#     x, y
# end
