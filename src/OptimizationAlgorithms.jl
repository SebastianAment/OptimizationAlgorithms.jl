module OptimizationAlgorithms

using LinearAlgebra
using LinearAlgebraExtensions
using LinearAlgebraExtensions: difference, LazyDifference

# principle of this package:
# reduce optimization problems to fixed point iterations
# to this end, design fixed point operator f! and stopping rule isfixed

######################## General Fixed Point Algorithm #########################
# IDEA: allocating (for immutable types), via 0D array?
# attempts to find a fixed point of f! by iterating f! on x
# in light of ode's, this could be a gradient flow
# general enough: iterate! - ?
# f! should be a fixed point operator, so ∃x s.t. x = f!(x)
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

# TODO:
# f!(x, t) should update x and return the value y
function optimize!(f!, x, isfixed)
    y = f!(x, t)
    t = 1
    while !isfixed(x, y, t)
        y = f!(x, t)
        t += 1
    end
    x, y, t
end
optimize!(f!, x) = optimize!(f!, x, StoppingCriterion(x))

########################### Stopping Criterion ##################################
# TODO: add verbose option
# TODO: enable x to be scalar?
# abstract type StoppingCriterion{T} end
mutable struct StoppingCriterion{T, S}
    x::T # holds last value of parameters
    dx::S # minimum absolute change in parameters x for termination
    rx::S # minimum relative change in parameters x for termination

    y::S # holds last function value
    dy::S
    ry::S

    maxiter::Int # maximum iterations
    function StoppingCriterion(x::AbstractVector, y::Real = Inf;
                dx = 1e-6, rx = 0., dy = 1e-6, ry = 0., maxiter::Int = 1024)
        new{typeof(x), typeof(y)}(fill(Inf, size(x)), dx, rx, y, dy, ry, maxiter)
    end
end

(T::StoppingCriterion)(x::AbstractVector, t::Integer) = T(x) || T(t)
(T::StoppingCriterion)(y::Real, t::Integer) = T(y) || T(t)
function (T::StoppingCriterion)(x::AbstractVector)
    any(isnan, x) && error("x contains NaN: x = $x")
    dx = norm(difference(x, T.x))
    rx = dx / norm(x)
    isfixed = dx < T.dx
    isfixed |= rx < T.rx
    T.x .= x
    return isfixed
end
function (T::StoppingCriterion)(y::Real)
    isnan(y) && error("y = NaN")
    dy = abs(y - T.y)
    ry = dy / abs(y)
    isfixed = dy < T.dy
    isfixed |= ry < T.ry
    isfixed |= T.maxiter < t
    T.y = y
    return isfixed
end
(T::StoppingCriterion)(t::Integer) = T.maxiter < t

######################## Canonical Descent Directions ##########################
# abstract type FixedPointIterator{T} end # makes sense as long as we stick to optimization
abstract type Update{T} end # <: FixedPointIterator{T} end
abstract type Direction{T} <: Update{T} end
# NOTE: all directions are decreasing unless a negative step size is chosen

# update functor
(D::Update)(x, t::Int) = update!(D, x, t)
(D::Update)(x) = update!(D, x)

# direction functor
(D::Direction)(x, t::Int) = direction(D, x) # since directions do not depend on time
(D::Direction)(x) = direction(D, x)

update!(D::Update, x, t::Int) = update!(D, x)
update!(D::Direction, x) = (x .+= D(x))
update!(D::Direction, x, t::Int) = (x .+= D(x, t))

# converting a direction to an update
update!(D::Direction) = (x, t::Int) -> (x .+= D(x, t))
fixedpoint!(d::Direction, x, isfixed) = fixedpoint!(update!(d), x, isfixed)

function objective end # returns objective function
objective(d::Direction, x) = d.f(x)
objective(d::Direction) = x->objective(d, x)

const value = objective

function direction end # returns direction
direction(D::Direction, x) = valdir(D, x)[2]
# direction(D::Direction, x, t::Int) = valdir(D, x)[2]

# calculates direction and value at the same time
# value_direction(D::Direction, x, t::Int) = valdir(D::Direction, x)
value_direction(D::Direction) = x -> valdir(D, x)
const valdir = value_direction

################################################################################
include("util.jl")
include("descentdirections.jl")
include("gaussnewton.jl")
include("descentupdates.jl")
include("stochastic.jl")
include("ode.jl")
include("stepsize.jl")
include("submodular.jl")

end # Optimization
