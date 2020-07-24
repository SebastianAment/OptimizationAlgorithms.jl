# TODO: more iterative methods: GMRES, Arnoldi, Lanczos
using LinearAlgebraExtensions: AbstractMatOrFac
################################################################################
# TODO: preconditioning, could be external by running cg on
# EAE'x = Eb where inv(E)inv(E)' = M where M is preconditioner
# TODO: nonlinear
# β = polyakribiere(dx₁, dx₂) = dx₁'(dx₁-dx₂) / (dx₂'dx₂)
# Careful with ill-conditioned systems
struct ConjugateGradient{T, M, V<:AbstractVector, R<:CircularBuffer} <: Update{T}
    mul_A!::M # application of (non-)linear operator
    b::V # target
    d::V # direction
    Ad::V # product of A and direction
    r::R # cirular buffer, containing last two residuals
    function ConjugateGradient(mul_A!, b::AbstractVector, x::AbstractVector)
        r₁ = similar(b)
        mul_A!(r₁, x)
        @. r₁ = b - r₁ # mul!(r₁, A, x, -1., 1.) # r = b - Ax
        d = copy(r₁)
        V = typeof(b)
        r = CircularBuffer{V}(2)
        push!(r, r₁); push!(r, similar(b));
        Ad = similar(b)
        new{eltype(x), typeof(mul_A!), V, typeof(r)}(mul_A!, b, d, Ad, r)
    end
end
const CG = ConjugateGradient
# A has to support mul!(b, A, x)
function CG(A::AbstractMatOrFac, b::AbstractVector, x::AbstractVector)
    mul_A!(Ad, d) = mul!(Ad, A, d)
    ConjugateGradient(mul_A!, b, x)
end
CG(A::AbstractMatOrFac, b::AbstractVector) = CG(A, b, zeros(eltype(A), size(A, 2)))

# no allocations :)
# TODO: matrix-solve
function update!(C::ConjugateGradient, x::AbstractVector, t::Int)
    if t > length(x) # algorithm does not have guarantees after n iterations
        return x # restart?
    elseif t > 1
        # TODO: change for non-linear problems
        β = sum(abs2, C.r[1]) / sum(abs2, C.r[2]) # new over old residual norm
        @. C.d = β*C.d + C.r[1] # axpy!
    end
    C.mul_A!(C.Ad, C.d) # C.Ad = C.A * C.d or could be nonlinear
    α = sum(abs2, C.r[1]) / dot(C.d, C.Ad)
    @. x += α * C.d # this directly updates x -> <:Update
    circshift!(C.r)
    @. C.r[1] = C.r[2] - α * C.Ad
    return x
end

function cg(A::AbstractMatOrFac, b::AbstractVector; max_iter::Int = size(A, 2), min_res::Real = 0)
    cg!(A, b, zeros(eltype(A), size(A, 2)), max_iter = max_iter, min_res = min_res)
end
function cg!(A::AbstractMatOrFac, b::AbstractVector, x::AbstractVector;
                                    max_iter::Int = size(A, 2), min_res::Real = 0)
    cg!(CG(A, b, x), x, max_iter = max_iter, min_res = min_res)
end

function cg!(U::CG, x::AbstractVector; max_iter::Int = size(A, 2), min_res::Real = 0)
    for i in 1:max_iter
        update!(U, x, i)
        norm(U.r[1]) > min_res || break
    end
    return x
end

################################################################################
# TODO: could extend for general iterative solvers
# struct IterativeMatrix end
# wrapper type which converts all solves into conjugate gradient solves,
# with minimum residual tolerance min_res
struct ConjugateGradientMatrix{T, M<:AbstractMatOrFac{T}} <: AbstractMatrix{T}
    parent::M
    min_res::T
    function ConjugateGradientMatrix(A; min_res = 0, check::Bool = true)
        ishermitian(A) || throw("input matrix not hermitian")
        T = eltype(A)
        new{T, typeof(A)}(A, convert(T, min_res))
    end
end
const CGMatrix = ConjugateGradientMatrix
Base.getindex(A::CGMatrix, i...) = getindex(A.parent, i...)
Base.setindex!(A::CGMatrix, i...) = setindex!(A.parent, i...)

Base.size(A::CGMatrix) = size(A.parent)

Base.:*(A::CGMatrix, b::AbstractVector) = A.parent * b
Base.:\(A::CGMatrix, b::AbstractVector) = cg(A.parent, b)
import LinearAlgebra: mul!, ldiv!
function mul!(y::AbstractVector, A::CGMatrix, x::AbstractVector, α::Real = 1, β::Real = 0)
    mul!(y, A.parent, x, α, β)
end
function ldiv!(y::AbstractVector, A::CGMatrix, x::AbstractVector)
    cg!(A.parent, x, y) # TODO: pre-allocate
end

# U = A.U
# copy!(U.b, x)
# U.mul_A!(U.r[1], y)
# @. U.r[1] = U.b - U.r[1]
# cg!(U, y)
