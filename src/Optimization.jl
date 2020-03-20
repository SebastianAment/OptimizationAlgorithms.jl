module Optimization

using LinearAlgebra
using Metrics
using LinearAlgebraExtensions: difference

# define euclidean metric
const euclidean = Metrics.EuclideanMetric()

# TODO: abstract away gradient computation for at least BFGS, LBFGS,
# TODO: all directions / updates seem to be time independent, except BFGS, LBFGS, ADAM
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

######################## General Fixed Point Algorithm #########################
# TODO: allocating (for immutable types), via 0D array?
# attempts to find a fixed point of f! by iterating f! on x
# in light of ode's, this could be a gradient flow
# general enough: iterate! - ?
function fixedpoint!(f!, x, isfixed)
    t = 1
    while !isfixed(x, t)
        f!(x, t)
        t += 1
    end
    x, t
end
# using default stopping criterion
fixedpoint!(f!, x) = fixedpoint!(f!, x, StoppingCriterion(x))

########################### Stopping Criterion ##################################
# abstract type StoppingCriterion{T} end
struct StoppingCriterion{T, S}
    x::T # holds last value of parameters
    δ::S # minimum absolute change in parameters x for termination
    maxiter::Int # maximum iterations
    # ε::T # minimum absolute change in function f for termination
    # y::T # holds last function value
    function StoppingCriterion(x, δ::Real = 1e-6, maxiter::Int = 128)
        new{typeof(x), typeof(δ)}(fill(Inf, size(x)), δ, maxiter)
    end
end
function (T::StoppingCriterion)(x, t)
    val = euclidean(x, T.x) < T.δ || t > T.maxiter # || abs(y - T.y) < ε
    T.x .= x
    return val
end

######################## Canonical Descent Directions ##########################
# abstract type FixedPointIterator{T} end # makes sense as long as we stick to optimization
abstract type Update{T} end # <: FixedPointIterator{T} end
abstract type Direction{T} <: Update{T} end
# NOTE: all directions are decreasing unless a negative step size is chosen

# update functor
(D::Update)(x, t::Int) = update!(D, x, t)
(D::Update)(x) = update!(D, x)

# direction functor
(D::Direction)(x, t::Int) = direction(D, x, t)
(D::Direction)(x) = direction(D, x)

update!(D::Update, x, t::Int) = update!(D, x)
update!(D::Direction, x) = (x .+= D(x))
update!(D::Direction, x, t::Int) = (x .+= D(x, t))
update!(D::Direction) = (x, t::Int) -> (x .+= D(x, t))

function objective end # returns objective function
objective(d::Direction, x) = d.f(x)
objective(d::Direction) = x->objective(d, x)

const value = objective

function direction end # returns direction
direction(D::Direction, x) = valdir(D, x)[2]
direction(D::Direction, x, t::Int) = valdir(D, x, t)[2]

# calculates direction and value at the same time
valuedirection(D::Direction, x, t::Int) = valdir(D::Direction, x)
valuedirection(D::Direction) = (x, t) -> valdir(D, x, t)
const valdir = valuedirection

################################################################################
include("util.jl")
include("descentdirections.jl")
include("descentupdates.jl")
include("stochastic.jl")
include("linearsolvers.jl")
include("compressedsensing.jl")
include("ode.jl")
include("stepsize.jl")
include("submodular.jl")

end # Optimization

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

######################### Convenience functions ################################
# intervalProjection(x, a, b) = max(min(x, b), a)
# intervalProjection!(y, x, a, b) = (@. y = max(min(x, b), a))
# # exponential average convenience
# expave!(z, α, x, y) = (@. z = α * x + (1-α) * y)
