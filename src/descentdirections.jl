########################## Auto-Differentiation Setup ##########################
# TODO: generalize from ForwardDiff
# using ForwardDiff for most algorithms if no explicit descent direction is given
import ForwardDiff: derivative, derivative!, gradient, gradient!,
    jacobian, jacobian!, hessian, hessian!
using ForwardDiff
const FD = ForwardDiff

using DiffResults
using DiffResults: DiffResult, GradientResult, HessianResult

gradient(f) = x -> gradient(f, x)
jacobian(f) = x -> jacobian(f, x)
hessian(f) = x -> hessian(f, x)

################################################################################
struct Gradient{T, F, V, R} <: Direction{T}
    f::F # function to be optimized
    d::V # direction
    r::R # differentiation result
    function Gradient(f::F, x::V) where {T, V<:AbstractArray{T}, F}
        r = DiffResults.GradientResult(x)
        new{T, F, V, typeof(r)}(f, similar(x), r)
    end
end
function valdir(G::Gradient, x::AbstractArray)
    gradient!(G.r, G.f, x) # strangely, this is still allocating
    G.d .= -1 .* DiffResults.gradient(G.r)
    return DiffResults.value(G.r), G.d
end
direction(G::Gradient, x::AbstractArray) = valdir(G, x)[2]

################################################################################
# newton's method for scalar-valued, vector-input functions
struct Newton{T, F, V, R} <: Direction{T}
    f::F
    d::V # direction
    r::R # differentiation result
    # λ::T # LevenbergMarquart parameter
    function Newton(f::F, x::V) where {T<:Number, V<:Union{T, AbstractVector{<:T}}, F}
        r = V <: Number ? DiffResult(x, (one(x),)) : HessianResult(x)
        new{T, F, V, typeof(r)}(f, zero(x), r)
    end
end
# TODO: specialize factorization of hessian
# cholesky only if problem is p.d. (constraint optimization with Lagrange multipliers isn't)
function valdir(N::Newton, x)
    hessian!(N.r, N.f, x)
    ∇ = DiffResults.gradient(N.r)
    H = DiffResults.hessian(N.r)
    H = cholesky(H)
    ldiv!(N.d, H, ∇)
    N.d .*= -1
    return DiffResults.value(N.r), N.d
end

# 1D Newton's method for optimization (zero finding of derivative)
function valdir(N::Newton, x::Real)
    g(x) = derivative(N.f, x)
    r = derivative!(N.r, g, x)
    step = - DiffResults.value(r) / DiffResults.derivative(r)
    return N.f(x), step
end

################################################################################
# Diagonal approximation to Hessian
# could also approximate 2nd derivatives as we go along
struct ScaledGradient{T, F, V, R} <: Direction{T}
    f::F
    d::V # direction
    r::R # differentiation result
    function ScaledGradient(f::F, x::V) where {T<:Number, V<:Union{T, AbstractVector{<:T}}, F}
        r = V <: Number ? DiffResult(x, (one(x),)) : HessianResult(x)
        new{T, F, V, typeof(r)}(f, zero(x), r)
    end
end
# ε makes sure we don't divide by zero
function valdir(N::ScaledGradient, x::AbstractVector)
    hessian!(N.r, N.f, x)
    ∇ = DiffResults.gradient(N.r)
    H = DiffResults.hessian(N.r)
    Δ = view(H, diagind(H))
    ε = 1e-10
    @. N.d = -1 * ∇ / max(abs(Δ), ε)
    return DiffResults.value(N.r), N.d
end

################################################################################
# Gauss-Newton algorithm, minimizes sum of squares of vector-valued function f w.r.t. x
# n, m = length(f(x)), length(x) # in case we want to pre-allocate the jacobian
struct GaussNewton{T, F, D, R} <: Direction{T}
    f::F
    d::D
    r::R
    function GaussNewton(f::F, x::V, y::V) where {T, V<:AbstractVector{T}, F<:Function}
        r = DiffResults.JacobianResult(y, x)
        new{T, F, V, typeof(r)}(f, similar(x), r)
    end
end
objective(G::GaussNewton) = x -> sum(abs2, f(x))
function valdir(G::GaussNewton, x::AbstractVector)
    jacobian!(G.r, G.f, x)
    y = DiffResults.value(G.r)
    J = DiffResults.jacobian(G.r)
    J = qr(J)
    ldiv!(N.d, J, ∇)
    N.d .*= -1
    return sum(abs2, y), N.d # since GaussNewton is for least-squares
end

################################################################################
# Broyden-Fletcher-Goldfarb-Shanno algorithm
# TODO: move first iteration into constructor, and so too with LBFGS
struct BFGS{T, D, M, X} <: Direction{T}
    direction::D
    H⁻¹::M # approximation to inverse Hessian
    x::X # old x
    ∇::X # old gradient
    d::X # direction
    s::X
    y::X # pre-allocate temporary storage
    Hy::X # storage for H⁻¹*y in bfgs_update, can alias with d
    check::Bool # check strong convexity
    function BFGS(dir::Direction, x::AbstractVector; check::Bool = true)
        n = length(x)
        H⁻¹ = zeros(eltype(x), (n, n))
        ∇ = -dir(x) # recording initial gradient
        D = DecreasingStep(dir, x) # first step is chosen with linesearch and gradient D only
        xold = copy(x)
        update!(D, x) # updates x
        s, y = (x - xold), (-dir(x) -∇)
        d = copy(s)         # d = -H⁻¹*∇
        H⁻¹[diagind(H⁻¹)] .= (s'y) / sum(abs2, y) # takes care of scaling issues
        Hy = d # we can alias memory here
        T, X = eltype(x), typeof(x)
        new{T, typeof(dir), typeof(H⁻¹), X}(dir, H⁻¹, xold, ∇, d, s, y, Hy, check)
    end
end
BFGS(f::Function, x) = BFGS(Gradient(f, x), x)
BFGS(f::Function, ∇::Function, x) = BFGS(CustomDirection(f, x->(f(x), -∇(x)), x), x)
objective(L::BFGS) = objective(L.direction)

function valdir(N::BFGS, x::AbstractVector)
    value, ∇ = valdir(N.direction, x) # this should maybe be valdir?
    ∇ .*= -1 # since we get the direction, (negative gradient)
    @. N.s = x - N.x; copy!(N.x, x) # update s and save current x
    @. N.y = ∇ - N.∇; copy!(N.∇, ∇) # if f is stochastic, have to be careful about consistency here
    bfgs_update!(N.H⁻¹, N.s, N.y, N.Hy, N.check)
    mul!(N.d, N.H⁻¹, ∇)
    N.d .*= -1
    return value, N.d
end
# updates inverse hessian with bfgs strategy
# could also use recursive algorithm here to reduce memory footprint (see below)
function bfgs_update!(H⁻¹::AbstractMatrix, s::AbstractVector, y::AbstractVector,
                    Hy::AbstractVector = similar(y), check::Bool = true)
    ρ = 1/(s'y)
    if ρ > 0 # function strongly convex -> update inverse Hessian
        mul!(Hy, H⁻¹, y)
        d = (ρ * dot(y, Hy) + 1) # rank 1
        @. H⁻¹ += ρ*(d*s*s' - s*Hy' - Hy*s')
    elseif check
        println("Warning: Skipping BFGS update because function is not strongly convex.")
    end
    return H⁻¹
end
# old update (allocating)
# A = (I - ρ*s*y') # TODO: pre-allocate?
# H⁻¹ .= A * H⁻¹ * A' .+ ρ*s*s' # TODO: woodbury?
# else # re-initialize inverse hessian to diagonal, if not strongly convex
#     N.H⁻¹ .= I(length(x))
# BFGS update corresponds to updating the Hessian as: (use for woodbury)
# Hs = N.H*s
# N.H .= N.H + y*y'*ρ - Hs*Hs' / dot(s, N.H, s)
################################################################################
# limited memory Broyden-Fletcher-Goldfarb-Shanno algorithm
struct LBFGS{T, D, X, V, A} <: Direction{T}
    direction::D
    x::X # old x
    ∇::X # old gradient
    d::X # direction
    s::V # previous m descent increments (including step size)
    y::V # previous m gradient differences
    m::Int # maximum recursion depth
    α::A
    check::Bool
    function LBFGS(dir::D, x::X, m::Int; check::Bool = true) where {T,
                                            D<:Direction, X<:AbstractVector{T}}
        m ≥ 1 || error("recursion depth m cannot be smaller than 1")
        n = length(x)
        s = CircularBuffer{X}(m) # TODO: elements could be static arrays
        y = CircularBuffer{X}(m)
        V = typeof(s)
        for i in 1:m
            s.buffer[i] = zeros(n)
            y.buffer[i] = zeros(n)
        end
        xold = copy(x)
        ∇ = -dir(x)
        d = -1e-6*∇
        x .+= d # TODO: initial stepsize selection?
        α = zeros(eltype(x), m)
        new{T, D, X, V, typeof(α)}(dir, xold, ∇, d, s, y, m, α, check)
    end
end
LBFGS(f::Function, x, m::Int) = LBFGS(Gradient(f, x), x, m)
function LBFGS(f::Function, ∇::Function, x, m::Int)
    LBFGS(CustomDirection(f, x->(f(x), -∇(x)), x), x, m)
end
objective(L::LBFGS) = objective(L.direction)

function valdir(N::LBFGS, x::AbstractVector)
    value, ∇ = valdir(N.direction, x)
    copy!(N.d, ∇)
    ∇ .*= -1 # since we get the direction, (negative gradient)
    s = difference(x, N.x) # difference to avoid temporary
    y = difference(∇, N.∇)
    ρ = 1/(s'y)
    if ρ > 0 # only update inverse Hessian approximation if ρ > 0, like in BFGS!
        copyfirst!(N.s, s); copy!(N.x, x) # lbfgs_update
        copyfirst!(N.y, y); copy!(N.∇, ∇)
    elseif N.check
        println("Warning: Skipping LBFGS update because function is not strongly convex.")
    end
    t = length(N.s)
    α = t < N.m ? view(N.α, 1:t) : N.α # limit recursion if we haven't stepped enough
    lbfgs_recursion!(N.d, N.s, N.y, α)
    return value, N.d
end

# recursive algorithm to compute gradient hessian product
# WARNING: d has to be initialized with the negative gradient
function lbfgs_recursion!(d::AbstractVector, s, y, α)
    m = length(α)
    for i in 1:m
        α[i] = dot(s[i], d) / dot(s[i], y[i])
        d .-= α[i] * y[i]
    end
    d .*= dot(s[1], y[1]) / sum(abs2, y[1]) # apply scaling
    for i in reverse(1:m)
        β_i = dot(y[i], d) / dot(y[i], s[i])
        d .+= (α[i] - β_i) * s[i]
    end
    return d
end

################################################################################
# natural gradient descent, see e.g. "A Stochastic Quasi-Newton Method for Online Convex Optimization"
# instead of G, keep track of inverse(G)?
struct NaturalGradient{T, F, M, D, R} <: Direction{T}
    f::F
    G::M # Riemanian metric (fisher information)
    d::D
    r::R # diff result
end
function valdir(N::NaturalGradient, x::AbstractVector)
    gradient!(N.r, N.f, x)
    ∇ = DiffResults.gradient(N.r)
    # TODO: can use Woodbury here
    # TODO: running average is only meaningful if the information geometry does not change
    # otherwise, use exponential average
    N.G .= t\((t-1) * N.G + ∇*∇') # should this rather be a step size???
    copy!(N.d, inverse(N.G) * ∇)
    N.d .*= -1
    return DiffResults.value(N.r), N.d
end

################################################################################
# fd returns both value and direction
# allow pre-allocation, using fd!
struct CustomDirection{T, F, FD} <: Direction{T}
    f::F # calculates objective value
    valdir::FD # calculates objective value and gradient
    function CustomDirection(f::F, valdir::FD, x::T) where {T<:AbstractArray, F, FD}
        new{T, F, FD}(f, valdir)
    end
end
valdir(D::CustomDirection, x) = D.valdir(x)

# valgrad(f, x) = f(x), -gradient(f, x)
# Gradient(f, x) = CustomDirection(f, valgrad, x)
# Newton(f, x) = CustomDirection(f, valnewton, x)

############################ update rules ######################################
# function newton(f)
#     d(x::AbstractVector) = - (hessian(f, x) \ gradient(f, x))
# end

# function LevenbergMarquart(f, λ::Real)
#     d(x::AbstractVector) = - (hessian(f, x) + λ*I) \ gradient(f, x)
# end

# using ForwardDiff: derivative
# # finds root of f(x) = 0, for minimization, input f'(x) ...
# function newton(f, x::Number; maxiter::Int = 16)
#     for i in 1:maxiter
#         x -= f(x) / derivative(f, x)
#     end
#     return x
# end
#
# # descent direction definition
# # TODO: specialize factorization of jacobian
# # technically it's the left pseudoinverse, not sure if we gain anything with that
# # d(x::AbstractVector) = pseudoinverse(jacobian(f, x)) * f(x)
# function gaussnewton(f)
#     d(x::AbstractVector) = -jacobian(f, x) \ f(x)
# end
