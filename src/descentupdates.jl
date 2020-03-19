################################################################################
# this has to be an update, since the updating of the x variable does not occur in valdir
struct PolyakMomentum{T, D<:Direction, S<:Real} <: Update{T}
    direction::D
    x::T # old state variable
    β::S
end
objective(P::PolyakMomentum) = objective(P.d)
function valdir(P::PolyakMomentum, x)
    v, d = valdir(P.direction, x)
    @. d += P.β * (x - P.x)
    return v, d
end
function update!(P::PolyakMomentum, x)
    d = P.direction(x) # update direction
    copy!(P.x, x) # update old value
    @. x += d
end

################################################################################
# Nesterov Accelerated Gradients, Nesterov Momentum
# e.g. https://jlmelville.github.io/mize/nesterov.html
# In conjunction with stochastic methods, variance reduction,
# look at https://arxiv.org/pdf/1910.07939.pdf
# and https://arxiv.org/pdf/1910.09494.pdf
struct Nesterov{T, G, S<:Real} <: Update{T}
    direction::G # should be negative gradient
    x::T # keeping old θ vector around
    α::S # A
    μ::S # M # technically this should also be a step size
end
function update!(N::Nesterov, x)
    d = N.direction(x)
    @. x = x + N.α * d # classical gradient step
    @. x = x + N.μ * (x - N.x)
    copy!(N.x, x) # update old x vector
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
