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
# TODO: more pre-allocation (s, y)
struct BFGS{T, F, M, D, R, N} <: Direction{T}
    f::F
    # gradient::G
    B::M # approximation to inverse Hessian
    d::D
    r::R
    x::N # old x
    function BFGS(f::F, x::V) where {T, F, V<:AbstractVector{T}}
        B = Matrix((1e-10I)(length(x)))
        r = GradientResult(x)
        d = -B*gradient(f, x) # TODO: figure out descent in deterministic case should be chosen with line search
        xold = copy(x)
        x .+= d
        new{T, F, typeof(B), V, typeof(r), V}(f, B, d, r, xold)
    end
end
# for Directions which need initialization, have initialize! ?
# would get rid of special case t == 1, similar with LBFGS below
# TODO: limit memory allocations
function valdir(N::BFGS, x::AbstractVector, t::Int)
    gradient!(N.r, N.f, x)
    ∇ = DiffResults.gradient(N.r)
    y = ∇ - FD.gradient(N.f, N.x) #
    # ∇ = N.gradient(N.r)
    # y = ∇ - N.gradient(N.f, N.x) # if f is stochastic, have to be careful about consistency here
    # s = N.d # this relies on step size selector to modify d, instead:
    s = x - N.x
    if t == 1
        N.B[diagind(N.B)] .= (s'y) / sum(abs2, y) # put in constructor?
    else
        ρ = 1/(s'y)
        A = (I - ρ*s*y') # pre-allocate?
        # mul!(N.B, A, N.B) # can't alias N.B
        N.B .= A * N.B * A' + ρ*s*s' # woodbury?
    end
    copy!(N.x, x) # store old parameter # TODO: maybe shift this to update!
    # this is the only line to be replaced for LBFGS
    mul!(N.d, N.B, ∇)
    N.d .*= -1
    return DiffResults.value(N.r), N.d
end

################################################################################
using DataStructures: CircularBuffer

# convenience for circular buffer
# does in place copy of x in front of buffer, important if cb is buffer of arrays
function copyfirst!(cb::CircularBuffer, x)
    cb.first = (cb.first == 1 ? cb.capacity : cb.first - 1)
    if length(cb) < cb.capacity
        cb.length += 1 # this probably shouldn't happen in this context
    end
    copy!(cb.buffer[cb.first], x)
end

################################################################################
# limited memory Broyden-Fletcher-Goldfarb-Shanno algorithm
struct LBFGS{T, F, D, V, R} <: Direction{T}
    f::F
    d::D
    s::V # previous m descent increments (including step size)
    y::V # previous m gradient differences
    m::Int # maximum recursion depth
    r::R # diff result
    x::D # old x
    function LBFGS(f::F, x::V, m::Int) where {T, F, V<:AbstractVector{T}}
        m ≥ 1 || error("recursion depth m cannot be smaller than 1")
        n = length(x)
        s = CircularBuffer{Vector{T}}(m) # TODO: these could be static arrays
        y = CircularBuffer{Vector{T}}(m)
        for i in 1:m
            push!(s, zeros(n))
            push!(y, zeros(n))
        end
        r = GradientResult(x)

        # first step, need this to compute differences in direction!
        xold = copy(x)
        G = Gradient(f, x) #ArmijoStep(Gradient(f, x)) # in stochastic setting should NOT be chosen with line search
        update!(G, x)
        d = x - xold #direction(G, x)
        new{T, F, V, typeof(s), typeof(r)}(f, d, s, y, m, r, xold)
    end
end

# TODO: limit memory allocations
function valdir(N::LBFGS, x::AbstractVector, t::Int)
    gradient!(N.r, N.f, x)
    ∇ = DiffResults.gradient(N.r)
    copyfirst!(N.y, difference(∇, gradient(N.f, N.x))) # difference to avoid temporary
    # copyfirst!(N.s, N.d) # this relies on step size selector to modify d, instead:
    copyfirst!(N.s, difference(x, N.x))
    copy!(N.x, x) # store old parameter # TODO: maybe shift this to update!

    # approximate hessian (lbfgs direction update)
    copy!(N.d, ∇) # TODO: figure out descent
    p = N.d
    s = N.s; y = N.y;
    m = min(t, N.m)
    α = zeros(eltype(x), m)
    for i in 1:m
        α[i] = (s[i]'p) / (s[i]'y[i])
        p .-= α[i] * y[i]
    end
    if t > 1
        p .*= (s[1]'y[1]) / (y[1]'y[1])
    end
    for i in reverse(1:m)
        β_i = (y[i]'p) / (y[i]'s[i])
        p .+= (α[i] - β_i) * s[i]
    end
    N.d .*= -1
    return DiffResults.value(N.r), N.d
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
    function CustomDirection(f::F, valdir::FD, x::V) where {T<:Number,
                                                V<:AbstractArray{<:T}, F, FD}
        new{V, F, FD}(f, valdir)
    end
end
valdir(D::CustomDirection, x) = D.valdir(x)
valdir(D::CustomDirection, x, t::Int) = D.valdir(x, t)

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
