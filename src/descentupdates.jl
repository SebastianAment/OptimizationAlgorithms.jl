################################################################################
# this has to be an update, since the updating of the x variable does not occur in valdir
struct PolyakMomentum{T, D, S<:Real} <: Update{T}
    direction::D
    x::T # old state variable
    α::S # stepsize
    β::S # momentum variable
end
const Polyak = PolyakMomentum
objective(P::PolyakMomentum) = objective(P.d)
function valdir(P::PolyakMomentum, x)
    v, d = valdir(P.direction, x)
    @. d += P.β * (x - P.x)
    return v, d
end
function update!(P::PolyakMomentum, x)
    d = P.direction(x) # update direction
    d += P.β * (x - P.x)
    copy!(P.x, x) # update old value
    @. x += P.α * d
end

################################################################################
# Nesterov Accelerated Gradients, Nesterov Momentum
# e.g. https://jlmelville.github.io/mize/nesterov.html
# In conjunction with stochastic methods, variance reduction,
# look at https://arxiv.org/pdf/1910.07939.pdf
# and https://arxiv.org/pdf/1910.09494.pdf
struct NesterovMomentum{T, G, S<:Real} <: Update{T}
    direction::G # should be negative gradient
    x::T # keeping old θ vector around
    t::T # need temporary
    α::S # A
    μ::S # M # technically this should also be a step size
end
const Nesterov = NesterovMomentum
function Nesterov(D, x::AbstractArray, α::Real, μ::Real)
    Nesterov(D, copy(x), copy(x), α, μ)
end
function update!(N::Nesterov, x)
    d = N.direction(x)
    @. x = x + N.α * d # classical gradient step
    copy!(N.t, x)
    @. x = x + N.μ * (x - N.x)
    copy!(N.x, N.t) # update old θ vector
    return x
end

################################################################################
# ADAM implementation will work by setting step size selector to constant
# technically it's a stochastic descent direction, make f stochastic
# based on "ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION"
# make this a descent direction
struct ADAM{T, F, V} <: Update{T} # think of it as direction
    d::F # direction
    α::T # basic step size
    β₁::T # "memory" parameter for m
    β₂::T # "memory" parameter for v
    m::V # first moment estimate
    v::V # second moment estimate
    ε::T
end
function ADAM(d, x::AbstractArray, α::Real, β₁::Real, β₂::Real, ε::Real = 1e-8)
    m, v = zero(x), zero(x)
    ADAM(d, α, β₁, β₂, m, v, ε)
end
# updates x in place
function update!(A::ADAM, x::AbstractArray, t::Int)
    dx = A.d(x)
    @. A.m = expave(A.m, dx, A.β₁) # update biased moment estimates
    @. A.v = expave(A.v, dx^2, A.β₂)
    α_t = stepsize(A, t)
    @. x += α_t * A.m / (√A.v + A.ε)
end

# exponential average
function expave(a::Real, b::Real, γ::Real)
    γ * a + (1-γ) * b
end

# AMSGRAD https://openreview.net/pdf?id=ryQu7f-RZ
struct AMSGRAD{T, F, V} <: Update{T} # think of it as direction
    d::F # direction
    α::T # basic step size
    β₁::T # "memory" parameter for m
    β₂::T # "memory" parameter for v
    m::V # first moment estimate
    v::V # second moment estimate
    maxv::V # maximum second moment in optimization history
    ε::T
end
function AMSGRAD(d, x::AbstractArray, α::Real, β₁::Real, β₂::Real, ε::Real = 1e-8)
    m, v, maxv = zero(x), zero(x), zero(x)
    AMSGRAD(d, α, β₁, β₂, m, v, maxv, ε)
end
# updates x in place
function update!(A::AMSGRAD, x::AbstractArray, t::Int)
    dx = A.d(x)
    @. A.m = expave(A.m, dx, A.β₁) # update biased moment estimates
    @. A.v = expave(A.v, dx^2, A.β₂)
    @. A.maxv = max(A.maxv, A.v)
    α_t = stepsize(A, t)
    @. x += α_t * A.m / (√A.maxv + A.ε)
end

# stepsize to debias moments
function stepsize(A::Union{ADAM, AMSGRAD}, t::Int)
    A.α * sqrt(1-A.β₂^t)/(1-A.β₁^t)
end
