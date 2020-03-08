module Optimization

using LinearAlgebra
using Metrics
using MyLazyArrays: difference

# define euclidean metric
const euclidean = Metrics.EuclideanMetric()

# TODO: projections
# TODO: Frank-Wolfe algorithm for linearly constrained convex problems
# TODO: Lagrange multiplier methods
# TODO: Proximal methods, proximal operators, e.g. for l1 regularized minimization

# DESIGN: instead or in addition of direction, can also have update strategies
# dependent on x and d (for example including projection, constrained quadratic programming)

# principle of this package:
# reduce optimization problems to fixed point iterations
# to this end, design fixed point operator f! and stopping rule isfixed

# somthing to consider:
# "Runge–Kutta-like scaling techniques for first-order methods in convex optimization"

########################## Auto-Differentiation Setup ##########################
# using ForwardDiff for most algorithms if no explicit descent direction is given
import ForwardDiff: derivative, derivative!, gradient, gradient!,
    jacobian, jacobian!, hessian, hessian!

using DiffResults
using DiffResults: DiffResult, GradientResult, HessianResult

gradient(f) = x -> gradient(f, x)
jacobian(f) = x -> jacobian(f, x)
hessian(f) = x -> hessian(f, x)

######################## General Fixed Point Algorithm #########################
# attempts to find a fixed point of f! by iterating f! on x
function fixedpoint!(f!, x::AbstractVector, isfixed)
    t = 1
    while !isfixed(x, t)
        f!(x, t)
        t += 1
    end
    x, t
end
# using default stopping criterion
fixedpoint!(f!, x::AbstractVector) = fixedpoint!(f!, x, StoppingCriterion(x))

########################### Stopping Criterion ##################################
# abstract type StoppingCriterion{T} end
mutable struct StoppingCriterion{T, V<:AbstractVector} #<: StoppingCriterion{T}
    δ::T # minimum absolute change in parameters x for termination
    x::V # holds last value of parameters
    maxiter::Int # maximum iterations
    # ε::T # minimum absolute change in function f for termination
    # y::T # holds last function value
end
StoppingCriterion(x) = StoppingCriterion(1e-6, fill(Inf, size(x)), 128)
function (T::StoppingCriterion)(x::AbstractVector, t)
    val = euclidean(x, T.x) < T.δ || t > T.maxiter # || abs(y - T.y) < ε
    typeof(x) <: AbstractVector ? copy!(T.x, x) : (T.x = x)
    return val
end

######################## Canonical Descent Directions ##########################
abstract type FixedPointIterator{T} end
abstract type Update{T} <: FixedPointIterator{T} end
abstract type Direction{T} <: Update{T} end
# NOTE: all directions are increasing unless a negative step size is chosen

# TODO:
# include("submodular.jl")

# TODO: take away restriction of x to be AbstractVector

# Direction objects D have to define a
# D(x::AbstractVector, f::Function) method which computes the direction
# and stores it in a field d
(D::Update)(x::AbstractVector, t::Int) = update!(D, x, t)
update!(D::Update, x::AbstractVector, t::Int) = update!(D, x)

function update!(D::Direction, x::AbstractVector)
    x .+= direction!(D, x)
end
# defined in case the Update is independent of the iteration counter t
function update!(D::Direction, x::AbstractVector, t::Int)
    x .+= direction!(D, x, t)
end
direction!(D::Direction, x::AbstractVector, t::Int) = direction!(D, x)

# value(D::Update) = DiffResults.value(D.r)
objective(d::Direction) = d.f
value(D::Direction) = DiffResults.value(D.r) # do all directions have a value?
direction(D::Direction) = D.d

function valdir(D::Direction, x, t::Int...)
    direction!(D, x, t...)
    value(D), direction(D)
end

################################################################################
# to make a descent direction out of a ascent direction
# struct Descent{T, D} <: Direction{T}
#     d::D
# end
# direction!(D::Descent, x::AbstractVector) = -direction!(D.d, x)
#
# function update!(D::Descent, x::AbstractVector)
#     x .-= direction!(D.d, x)
# end

################################################################################
struct Gradient{T, F, V, R} <: Direction{T}
    f::F # function to be optimized
    d::V # direction
    r::R # differentiation result
    function Gradient(f::F, x::V) where {T, V<:AbstractVector{T}, F}
        r = DiffResults.GradientResult(x)
        new{T, F, V, typeof(r)}(f, similar(x), r)
    end
end
# ? direction(G::Gradient) = DiffResults.gradient(G.r)
function direction!(G::Gradient, x::AbstractVector)
    gradient!(G.r, G.f, x) # strangely, this is still allocating
    G.d .= DiffResults.gradient(G.r)
    G.d .*= -1
end

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
# d(x) = - cholesky(hessian(f, x))) \ gradient(f, x))
function direction!(N::Newton, x::AbstractVector)
    hessian!(N.r, N.f, x)
    ∇ = DiffResults.gradient(N.r)
    H = DiffResults.hessian(N.r)
    H = cholesky(H)
    ldiv!(N.d, H, ∇)
    N.d .*= -1
end

# 1D Newton's method for optimization
function direction(N::Newton, x::Real)
    g(x) = derivative(N.f, x)
    r = derivative!(N.r, g, x)
    -DiffResults.value(r) / DiffResults.derivative(r)
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
function direction!(N::ScaledGradient, x::AbstractVector)
    hessian!(N.r, N.f, x)
    ∇ = DiffResults.gradient(N.r)
    H = DiffResults.hessian(N.r)
    Δ = view(H, diagind(H))
    ε = 1e-10
    @. N.d = -1 * ∇ / max(abs(Δ), ε)
end

################################################################################
# fd returns both value and direction
# allow pre-allocation, using fd!
struct CustomDirection{T, F, FD, V} <: Direction{T}
    f::F # calculates objective value
    valgrad::FD # calculates objective value and gradient
    d::V # TODO: store direction in place
    function CustomDirection(f::F, valgrad::FD, x::V) where {T<:Number,
                                                V<:AbstractVector{<:T}, F, FD}
        new{T, F, FD, V}(f, valgrad, zero(x))
    end
end
valdir(D::CustomDirection, x::AbstractVector) = D.gradient(D, x)

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
function direction!(G::GaussNewton, x::AbstractVector)
    jacobian!(G.r, G.f, x)
    y = DiffResults.value(G.r)
    J = DiffResults.jacobian(G.r)
    J = qr(J)
    ldiv!(N.d, J, ∇)
    N.d .*= -1
end
# Gauss-Newton minimizes sum of squares
value(G::GaussNewton) = sum(abs2, DiffResults.value(G.r))

################################################################################
# Broyden-Fletcher-Goldfarb-Shanno algorithm
# TODO: more pre-allocation (s, y)
struct BFGS{T, F, M, D, R, N} <: Direction{T}
    f::F
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
function direction!(N::BFGS, x::AbstractVector, t::Int)
    gradient!(N.r, N.f, x)
    ∇ = DiffResults.gradient(N.r)
    y = ∇ - gradient(N.f, N.x) # if f is stochastic, have to be careful about consistency here
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
end

################################################################################
using DataStructures: CircularBuffer

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
        G = ArmijoStep(Gradient(f, x)) # in stochastic setting should NOT be chosen with line search
        update!(G, x)
        d = direction(G)

        new{T, F, V, typeof(s), typeof(r)}(f, d, s, y, m, r, xold)
    end
end
# TODO: limit memory allocations
function direction!(N::LBFGS, x::AbstractVector, t::Int)
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
    return N.d
end

# convenience for circular buffer
# does in place copy of x in front of buffer
function copyfirst!(cb::CircularBuffer, x)
    cb.first = (cb.first == 1 ? cb.capacity : cb.first - 1)
    if length(cb) < cb.capacity
        cb.length += 1 # this probably shouldn't happen in this context
    end
    copy!(cb.buffer[cb.first], x)
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
function direction!(N::NaturalGradient, x::AbstractVector)
    gradient!(N.r, N.f, x)
    ∇ = DiffResults.gradient(N.r)
    # TODO: can use Woodbury here
    # TODO: running average is only meaningful if the information geometry does not change
    # otherwise, use exponential average
    N.G .= t\((t-1) * N.G + ∇*∇') # should this rather be a step size???
    copy!(N.d, inverse(N.G) * ∇)
    N.d .*= -1
end

################################################################################
struct PolyakMomentum{T, D<:Direction, F, V} <: Direction{T}
    d::D
    β::T
    x::V # old state variable
end
objective(P::PolyakMomentum) = objective(P.d)
value(P::PolyakMomentum) = value(P.d)
direction(P::PolyakMomentum) = direction(P.d)
function direction!(P::PolyakMomentum, x::AbstractVector)
    direction!(P.d, x) # update direction
    direction(P.d) .= direction(P.d) .+ P.β .* (x .- P.x)
end
function update!(P::PolyakMomentum, x::AbstractVector)
    direction!(P, x) # update direction
    copy!(P.x, x) # update old value
    x .+= direction(P)
end

################################################################################
# Nesterov Accelerated Gradients, Nesterov Momentum
# e.g. https://jlmelville.github.io/mize/nesterov.html
# In conjunction with stochastic methods, variance reduction,
# look at https://arxiv.org/pdf/1910.07939.pdf
# and https://arxiv.org/pdf/1910.09494.pdf
struct Nesterov{T, F, V} <: Update{T}
    f::F
    α::T # A
    μ::T # M # technically this should also be a step size
    θ₁::V # keeping old θ vector around
    θ₂::V
end
function update!(N::Nesterov, x::AbstractVector, dx = gradient(N.f, x))
    @. N.θ₁ = x - N.α * dx # classical gradient step
    @. x = N.θ₁ + μ * (N.θ₁ - Ν.Θ₂)
    copy!(N.Θ₂, Ν.θ₁) # update θ₁ vector
    return x
end

################################################################################
# Adam implementation will work by setting step size selector to constant
# technically it's a stochastic descent direction, make f stochastic
# based on "ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION"
# TODO: different method where second moments are replaced with second derivatives
# (stochastic version of scaled gradient)
# make this a descent direction
struct Adam{T, F, V} <: Update{T} # think of it as direction
    f::F
    α::T # basic step size
    β₁::T # "memory" parameter for m
    β₂::T # "memory" parameter for v
    m::V # first moment estimate
    v::V # second moment estimate
    ε::T
    # r::R # DiffResults
end
function Adam(f, α::Real, β₁::Real, β₂::Real, x::AbstractVector, ε::Real = 1e-8)
    m, v = one(x), one(x)
    Adam(f, α, β₁, β₂, m, v, ε)
end
# updates x in place
function update!(A::Adam, x::AbstractVector, t::Int, dx = gradient(A.f, x))
    m, v, β₁, β₂ = A.m, A.v, A.β₁, A.β₂
    @. m = β₁ * m + (1-β₁) * dx
    @. v = β₂ * v + (1-β₂) * dx^2 # update biased moment estimates
    α_t = A.α * √(1-β₂^t)/(1-β₁^t) # update step-size to de-bias moments
    @. x = -α_t * m / (√v + A.ε) # TODO: ascent vs descent
end

################################################################################
# TODO: preconditioning, could be external by running cg on
# EAE'x = Eb where inv(E)inv(E)' = M where M is preconditioner
# TODO: nonlinear
# β = polyakribiere(dx₁, dx₂) = dx₁'(dx₁-dx₂) / (dx₂'dx₂)
# Careful: does not like ill-conditioned systems
struct ConjugateGradient{T, M, V} <: Update{T}
    # f::F # nonlinear conjugate gradient
    A::M # linear conjugate gradient
    b::V # target
    r₁::V # last two residuals
    r₂::V
    d::V # direction
    Ad::V # product of A and direction
    function ConjugateGradient(A::M, b::U, x::V) where {T<:Number,
                    M<:AbstractMatrix{T}, U<:AbstractVector, V<:AbstractVector}
        r₁ = copy(b)
        mul!(r₁, A, x, -1., 1.) # r = b - Ax
        d = copy(r₁)
        r₂ = similar(b)
        Ad = similar(b)
        new{T, M, V}(A, b, r₁, r₂, d, Ad)
    end
end
# no allocations :)
function update!(C::ConjugateGradient, x::AbstractVector, t::Int)
    if t > 1
        # this changes for non-linear problems
        β = sum(abs2, C.r₁) / sum(abs2, C.r₂) # new over old residual norm
        @. C.d = β*C.d + C.r₁  # axpy!
    elseif t ≥ length(x) # algorithm does not have guarantees after n iterations
        return x
    end
    # this is the only line that is explicitly linear
    mul!(C.Ad, C.A, C.d) # C.Ad = C.A * C.d
    α = sum(abs2, C.r₁) / dot(C.d, C.Ad)
    @. x += α * C.d # this directly updates x -> <:Update
    copy!(C.r₂, C.r₁) # could get around doing this copy with circular buffer
    @. C.r₁ -= α * C.Ad
    x
end
# TODO: more iterative methods: GMRES, Arnoldi, Lanczos
include("compressedsensing.jl")

include("ode.jl")
############################ Step Size Selectors ###############################
# generalizes search for an α such that (x - α * direction) leads to minimizer

# TODO: if we want to keep track of last α, could have 0D array and still have immutable object
# α = zeros(), or Array{Float64, 0}(undef)
# TODO: employ local / global optimization for optimal step size selection?

# if f isa StepSize, should be able to call with f(x, f, d, α, n)
# where x is the current parameter vector,
# f is the function to be minimized
# d(x) is the direction to be taken
# α is the first guess at a step size
# n is the iteration number in the optimization routine
struct ConstantStep{T, D<:Direction} <: Direction{T}
    d::D
    α::T
end
# decide if I want to represent the multiplication lazily
# direction(C::ConstantStep) = C.α * direction(C.d) # introduces temporary
function direction!(C::ConstantStep, x::AbstractVector)
    direction!(C.d, x) # update direction vector
    direction(C.d) .*= C.α # multiply direction with step-size
end
# function update!(C::ConstantStep, x::AbstractVector)
#     direction!(C.d, x) # update direction vector
#     x .+= C.α .* direction(C.d)
# end

################################################################################
# TODO: specialize for 1D input
mutable struct ArmijoStep{T, D<:Direction} <: Direction{T}
    d::D # direction κατεύθυνση
    c::T # coefficient in Armijo rule
    α::T # last α, has to be 0D array, or armijo is mutable (went with mutable)
    decrease::T # μείωση factor
    increase::T # αύξηση factor
    maxbacktrack::Int
end
function ArmijoStep(d::D, c::T = 0, α = 1., decrease = 3., increase = 2.,
                            maxbacktrack::Int = 16) where {T, D<:Direction}
    ArmijoStep{T, D}(d, c, T(α), T(decrease), T(increase), maxbacktrack)
end
objective(A::ArmijoStep) = objective(A.d)
direction(A::ArmijoStep) = direction(A.d)

# ε is smallest allowable step size
function update!(A::ArmijoStep, x::AbstractVector, t::Int...)
    f = objective(A)
    y, d = valdir(A.d, x)
    dg = typeof(A.d) <: Gradient ? d'd : d'gradient(f, x)
    α = A.α # initial α
    xn = x + α*d # could pre-allocate space for x + α * d
    for i in 1:A.maxbacktrack
        if (f(xn) ≤ y + α * A.c * dg) #|| α < ε # rule for minimization
            copy!(x, xn)  # only if we jump out with a good step size, reassign x to xn
            break
        else
            α /= A.decrease
            @. xn = x + α*d
        end
    end
    # A.α = α * A.increase
    return x
end
# function update!(A::ArmijoStep, x::AbstractVector)
#     x .+= A.α *
# end

# TODO: Could rename Damped, so type reads e.g. Damped{Newton}
# 0 < c₁ < c₂ < 1
mutable struct WolfeStep{T, D<:Direction{T}} <: Update{T}
    d::D
    c₁::T
    c₂::T
    α::T
    decrease::T # μείωση factor
    increase::T # αύξηση factor
    maxbacktrack::Int
end
function WolfeStep(d::D, c::T = .99, α = 1., decrease = 7., increase = 2.,
                            maxbacktrack::Int = 16) where {T, D<:Direction}
    ArmijoStep{T, D}(d, c, T(α), T(decrease), T(increase), maxbacktrack)
end
# solves for α s.t. x - αd satisfies Wolfe conditions
function update!(W::WolfeStep, x::AbstractVector)
    f = objective(W)
    direction!(W.d, x) # get direction and value
    y, d = value(W.d), direction(W.d) # should we get the value like this?
    dg = typeof(W.d) <: Gradient ? d'd : d'gradient(f, x)
    α = W.α # initial α
    xn = x + α*d # could pre-allocate space for x + α * d
    for i in 1:W.maxbacktrack
        armijo = f(xn) ≤ y + α * W.c₁ * dg
        curv = abs(d'gradient(f, xn)) ≤ abs(W.c₂ * dg)
        if armijo && curv # rule for minimization
            break
        else
            α /= W.decrease
            @. xn = x + α*d
        end
    end
    W.α = α * W.increase
    copy!(x, xn)
end

################################################################################
# struct TimeDependentStep{T, D<:Direction} <: Direction{T}
#     d::D
#     α::T
# end
# function update!(C::TimeDependentStep, x::AbstractVector, t::Int)
#     direction!(C.d, x)
#     C.d .*= C.α(t)
#     x .+= direction(C.d)
# end

########################### Stopping Criteria ###################################

# could generally be function of f, x, (d), α, t

# compositie stopping critera
# struct Any{T, A<:Tuple{Vararg{StoppingCriterion}}} <: StoppingCriterion{T}
#     args::A
# end
# (T::Any)(x, f, g, t) = any(c(x, f, g, t) for c in T.args)
#
# struct All{T, A<:Tuple{Vararg{StoppingCriterion}}} <: StoppingCriterion{T}
#     args::A
# end
# (T::All)(x, f, g, t) = all(c(x, f, g, t) for c in T.args)

# this is closest to the definition of fixed point finding
# struct XStop{T, V} <: StoppingCriterion{T}
#     ε::T
#     x::V
# end
# (T::XStop)(x::AbstractVector) = norm(difference(x, T.x)) < ε
# update!(T::XStop, x) = (T.x .= x) # update the stopping criterion
#
# # if function value does not decay enough (could be done in step size selection?)
# # should this be mutable since old_y is scalar but has to change (same with TStop)
# mutable struct YStop{T} <: StoppingCriterion{T}
#     δ::T
#     old_y::T
# end

# if we reach maximum iterations
# struct TStop{T} <: StoppingCriterion
#     max::Int
#    i::Int
# end

# if gradient falls below zero (more generally increment?)
# struct GradientStop{T} <: StoppingCriterion{T}
#     ε::T
# end

########################## Stochastic Approximations ###########################
# f should be dependent on data x and parameters θ, f(θ, x)
# possibly extend to f(θ, x, y) for regression
# Idea: stochastic and online could be subsumed if we have an infinite vector
# whose getindex method always returns random elements (maybe that is not a good definition)
# computes stochastic approximation to f by randomly samling from x with replacement
# TODO: add option to sample without replacement
function stochastic(f, x::AbstractVector)
    return function sf(θ, batchsize::Int = 1)
        subset = sort!(rand(1:length(x), batchsize)) # without replacement # with replacement: sample(1:length(x), batchsize, replace = false)
        f(θ, x[subset])
    end
end

# stochastic approximation of f(θ, x) by going through x sequentially (in online fashion)
# combine this with PeriodicVector in MyLinearAlgebra to get infinite stream of data
function online(f, x::AbstractVector)
    return function sf(θ, t::Int, batchsize::Int = 1)
        subset = t*batchsize:(t+1)*batchsize
        f(θ, x[subset])
    end
end

############################ update rules ######################################
# function newton(f)
#     d(x::AbstractVector) = - (hessian(f, x) \ gradient(f, x))
# end

# function LevenbergMarquart(f, λ::Real)
#     d(x::AbstractVector) = - (hessian(f, x) + λ*I) \ gradient(f, x)
# end

using ForwardDiff: derivative
# finds root of f(x) = 0, for minimization, input f'(x) ...
function newton(f, x::Number; maxiter::Int = 16)
    for i in 1:maxiter
        x -= f(x) / derivative(f, x)
    end
    return x
end

# descent direction definition
# TODO: specialize factorization of jacobian
# technically it's the left pseudoinverse, not sure if we gain anything with that
# d(x::AbstractVector) = pseudoinverse(jacobian(f, x)) * f(x)
function gaussnewton(f)
    d(x::AbstractVector) = -jacobian(f, x) \ f(x)
end

######################### Convenience functions ################################
# intervalProjection(x, a, b) = max(min(x, b), a)
# intervalProjection!(y, x, a, b) = (@. y = max(min(x, b), a))
# # exponential average convenience
# expave!(z, α, x, y) = (@. z = α * x + (1-α) * y)

# two most important rules to consider when selecting step sizes
# function armijo_rule(x, f::Function, d, ∇, c, α)
#     f(x + α*d) ≤ f(x) + α*c*d'gradient(f, x) # RHS could be computed independently of α
# end
# # implements the strong wolfe
# function curvature_rule(x, f, d, c, α)
#     abs(d'gradient(f, x + α*d)) ≤ abs(c*d'gradient(f, x)) # weak rule: -d'∇(x + α*d) ≤ -c*d'∇(x)
# end
# function armijo_rule(xn, y, f::Function, dg, c, α)
#     f(xn) ≤ y + α * c * dg # RHS could be computed independently of α
# end
# function curvature_rule(xn, f, dg, c, α)
#     abs(d'gradient(f, xn)) ≤ abs(c*dg) # weak rule: -d'∇(x + α*d) ≤ -c*d'∇(x)
# end

end

# allocating
# function descent(x, f, direction;
#                 maxiter::Int = 128, update!,
#                 α::AbstractStepSize, # initial step size
#                 ε::Real = 1e-8, # minimum absolute change in function f for termination
#                 δ::Real = 1e-8, # minimum absolute change in parameters x for termination
#                 projection = identity) # think about the projection step
#     old_x = zero(x)
#     y = f(x)
#     for t in 1:maxiter
#         old_x, old_y = x, y
#         x += α(t) * direction(x)
#         if euclidean(x, old_x) < δ || abs(y - old_y) < ε
#             break
#         end
#     end
#     return x, val, iter
# end
