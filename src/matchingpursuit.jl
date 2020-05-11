abstract type AbstractPursuit{T} <: Update{T} end

############################# Matching Pursuit #################################
# should MP, OMP, have a k parameter?
# SP needs one, but for MP could be handled in outer loop
# WARNING: requires A to have unit norm columns
struct MatchingPursuit{T, AT<:AbstractMatrix{T}, B<:AbstractVector{T}} <: AbstractPursuit{T}
    A::AT
    b::B
    k::Int # maximum number of non-zeros

    # temporary storage
    r::B # residual
    Ar::B # inner products between measurement matrix and residual
    # a approximation # can be updated efficiently
end
const MP = MatchingPursuit
function MP(A::AbstractMatrix, b::AbstractVector, k::Integer)
    n, m = size(A)
    T = eltype(A)
    r, Ar = zeros(T, n), zeros(T, m)
    MP(A, b, k, r, Ar)
end

function update!(P::MP, x::AbstractVector = spzeros(size(P.A, 2)))
    nnz(x) ≥ P.k && return x # return if the maximum number of non-zeros was reached
    residual!(P, x) # TODO: could just update approximation directly for mp
    i = maxabsdot!(P)
    x[i] += dot(@view(P.A[:,i]), P.r) # add non-zero index to x
    return x
end

# calculates k-sparse approximation to Ax = b via matching pursuit
function mp(A::AbstractMatrix, b::AbstractVector, k::Int, x = spzeros(size(A, 2)))
    P! = MP(A, b, k)
    fixedpoint!(P!, x, StoppingCriterion(x, maxiter = k, dx = eps(eltype(A))))
    x
end

###################### Orthogonal Matching Pursuit #############################
# TODO: preconditioning, non-negativity constraint -> NNLS.jl
using SparseArrays
struct OrthogonalMatchingPursuit{T, AT<:AbstractMatrix{T},
                                    B<:AbstractVector{T}} <: AbstractPursuit{T}
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
    residual!(P, x)
    i = maxabsdot!(P)
    x[i] = NaN # add non-zero index to x
    solve!(P, x) # optimize all active atoms
    return x
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
struct SubspacePursuit{T, AT<:AbstractMatrix{T}, B<:AbstractVector{T}} <: AbstractPursuit{T}
    A::AT
    b::B
    k::Int # maximum number of non-zeros

    # temporary storage
    r::B # residual
    Ar::B # inner products between measurement matrix and residual
    Ai::AT # space for A[:, x.nzind] and its qr factorization
end
const SP = SubspacePursuit

function SP(A::AbstractMatrix, b::AbstractVector, k::Integer)
    2k > length(b) && error("2k = $(2k) > $(length(b)) = length(b) is invalid for Subspace Pursuit")
    n, m = size(A)
    T = eltype(A)
    r, Ar = zeros(T, n), zeros(T, m)
    Ai = zeros(T, (n, 2k))
    SP(A, b, k, r, Ar, Ai)
end

# returns indices of k atoms with largest inner products with residual
# TODO: need threshold for P.Ar
@inline function _sspindex!(P::SP)
    mul!(P.Ar, P.A', P.r)
    @. P.Ar = abs(P.Ar)
    partialsortperm(P.Ar, 1:P.k, rev = true)
end

# TODO: could pre-allocate nz arrays to be of length 2K
function update!(P::SP, x::AbstractVector = spzeros(size(P.A, 2)))
    if nnz(x) == 0
        return x .= omp(P.A, P.b, P.k) # first iteration via omp
    end
    nnz(x) == P.k || throw("nnz(x) = $(nnz(x)) ≠ $(P.k) = k")
    residual!(P, x)
    i = _sspindex!(P)
    @. x[i] = NaN # add non-zero index to x
    solve!(P, x) # optimize all active atoms
    i = partialsortperm(abs.(x.nzval), 1:P.k) # find k smallest atoms
    @. x.nzval[i] = 0
    dropzeros!(x)
    solve!(P, x) # optimize all active atoms
    return x
end

# calculates k-sparse approximation to Ax = b via subspace pursuit
function sp(A::AbstractMatrix, b::AbstractVector, k::Int; iter = 8)
    P! = SP(A, b, k)
    x = spzeros(size(A, 2))
    for i in 1:iter
        P!(x)
    end
    return x
end

######################### Noiseless Relevance Pursuit ##########################
using LinearAlgebraExtensions: Projection, AbstractMatOrFac
# noiseless limit of greedy sbl
struct NoiselessRelevancePursuit{T, AT<:AbstractMatrix{T}, B<:AbstractVector{T}} <: AbstractPursuit{T}
    A::AT
    b::B
    # σ::T # noise standard deviation

    k::Int # maximum number of non-zeros

    # temporary storage
    r::B # residual
    Ar::B # inner products between measurement matrix and residual
    Ai::AT # space for A[:, x.nzind] and its qr factorization
end
const NRP = NoiselessRelevancePursuit

function NRP(A::AbstractMatrix, b::AbstractVector, k::Int)
    n, m = size(A)
    T = eltype(A)
    r, Ar = zeros(T, n), zeros(T, m)
    Ai = zeros(T, (n, k))
    NRP(A, b, k, r, Ar, Ai)
end

function residual_operator(A::AbstractMatOrFac)
    I - Matrix(Projection(A))
end

function update!(P::NRP, x::AbstractVector = spzeros(size(P.A, 2)))
    Ai = P.A[:, x.nzind]
    D = residual_operator(Ai)
    AD = P.A' * D
    q = AD * P.b
    s = @views [sqrt(max(0, dot(AD[i,:], P.A[:,i]))) for i in 1:size(P.A, 2)]

    inner = @. abs(q / s)
    @. inner[x.nzind] = 0

    i = argmax(inner)
    x[i] = NaN # add non-zero index to x
    solve!(P, x) # optimize all active atoms
    return x
end

# calculates k-sparse approximation to Ax = b via relevance pursuit
function nrp(A::AbstractMatrix, b::AbstractVector, k::Int)
    P! = NRP(A, b, k)
    x = spzeros(size(A, 2))
    for i in 1:k
        P!(x)
    end
    return x
end

####################### Pursuit Helpers ########################################
# calculates residual of
@inline function residual!(P::AbstractPursuit, x::AbstractVector)
    residual!(P.r, P.A, x, P.b)
end
@inline function residual!(r::AbstractVector, A::AbstractMatrix, x::AbstractVector, b::AbstractVector)
    copyto!(r, b)
    mul!(r, A, x, -1, 1)
end
# returns index of atom with largest dot product with residual
@inline function maxabsdot!(P::AbstractPursuit) # 0 alloc
    mul!(P.Ar, P.A', P.r)
    @. P.Ar = abs(P.Ar)
    argmax(P.Ar)
end
@inline function _factorize!(P::AbstractPursuit, x::AbstractVector)
    n = size(P.A, 1)
    k = nnz(x)
    Ai = reshape(@view(P.Ai[1:n*k]), n, k)
    @. Ai = @view(P.A[:, x.nzind])
    return qr!(Ai) # this still allocates a little memory
end
# ordinary least squares solve
@inline function solve!(P::AbstractPursuit, x::AbstractVector)
    F = _factorize!(P, x)
    ldiv!(x.nzval, F, P.b)     # optimize all active atoms
    dropzeros!(x)
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


############################## Relevance Pursuit ###############################
# eh
# using LinearAlgebraExtensions: Projection, AbstractMatOrFac
# # noiseless limit of greedy sbl
# struct RelevancePursuit{T, AT<:AbstractMatrix{T}, B<:AbstractVector{T},
#         G<:AbstractVector{T}} <: AbstractPursuit{T}
#     A::AT
#     b::B
#     σ::T # noise standard deviation
#     γ::G # prior variance of weights
#     # k::Int # maximum number of non-zeros
#     # temporary storage
#     # r::B # residual
#     # Ar::B # inner products between measurement matrix and residual
#     # Ai::AT # space for A[:, x.nzind] and its qr factorization
# end
# const RP = RelevancePursuit
#
# function RP(A::AbstractMatrix, b::AbstractVector, σ::Real)
#     n, m = size(A)
#     T = eltype(A)
#     # r, Ar = zeros(T, n), zeros(T, m)
#     # Ai = zeros(T, (n, k))
#     γ = spzeros(T, m)
#     RP(A, b, σ, γ)
# end
#
# function update!(P::RP, x::AbstractVector = spzeros(size(P.A, 2)))
#     isin = P.γ.nzind
#     if length(isin) > 0
#         γi = P.γ.nzval
#         Ai = P.A[:, isin]
#         println(size(γi))
#         println(size(Ai))
#         # println(inverse(factorize(P.σ^2 * Diagonal(inv.(γi)) + Ai'Ai)))
#         # D = Woodbury(I(length(isin)), Ai,
#         #             inverse(factorize(P.σ^2 * Diagonal(inv.(γi)) + Ai'Ai)),
#         #             Ai', -)
#         D = factorize(
#                 Woodbury(I(length(P.b)), Ai, Diagonal(inv.(γi/P.σ^2)), Ai')
#             )
#     else
#         D = I
#     end
#
#     AD = P.A' * D
#     q = AD * P.b
#     s = @views [sqrt(max(0, dot(AD[i,:], P.A[:,i]))) for i in 1:size(P.A, 2)]
#
#     inner = @. abs(q / s)
#     # if length(isin) > 0
#     #     minvalue, minindex = findmin(inner[isin])
#     #     if minvalue ≤ P.σ
#     #         P.γ[minindex] = 0
#     #         x[minindex] = 0
#     #         dropzeros!(P.γ)
#     #         dropzeros!(x)
#     #         return x
#     #     end
#     # end
#     inner[isin] .= 0
#     println(inner)
#     maxvalue, maxindex = findmax(inner)
#     println("maxvalue = $maxvalue")
#     println("maxindex = $maxindex")
#     println(P.γ)
#     if maxvalue > P.σ
#         P.γ[maxindex] = ((q[maxindex]/s[maxindex])^2 - 1) / s[maxindex]^2 # (q^2 - s^2) / s^4
#         x[maxindex] = NaN # add non-zero index to x
#         isin = P.γ.nzind
#         γi = P.γ.nzval
#         Ai = P.A[:, isin]
#         C = cholesky!(P.σ^2 * Diagonal(inv.(γi)) + Ai'Ai)
#         ldiv!(x.nzval, C, Ai' * P.b)
#     end
#     return x
# end
#
# # calculates k-sparse approximation to Ax = b via relevance pursuit
# function rp(A::AbstractMatrix, b::AbstractVector, σ::Real; maxiter = size(A, 1))
#     P! = RelevancePursuit(A, b, σ)
#     x = spzeros(size(A, 2))
#     for i in 1:maxiter
#         P!(x)
#     end
#     return x
# end
