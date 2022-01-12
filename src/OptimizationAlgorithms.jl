module OptimizationAlgorithms

using LinearAlgebra
# using LinearAlgebraExtensions
# using LinearAlgebraExtensions: difference, LazyDifference

difference(x, y) = x-y

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
    while true
        x = f!(x, t)
        t += 1
        isfixed(x, t) && break
    end
    x, t
end

# using default stopping criterion
fixedpoint!(f!, x) = fixedpoint!(f!, x, StoppingCriterion(x))

# # TODO:
# # f!(x, t) should return both x and y for generality with scalar case
# this is not a problem for Directions, but for Updates
# either rewrite update! to do this, thereby breaking it's use in fixedpoint!,
# or write new function update_evaluate!()
# function optimize!(f!, x, isfixed)
#     t = 0
#     while true
#         x, y = f!(x, t)
#         t += 1
#         isfixed(x, y, t) && break
#     end
#     x, y, t
# end
# optimize!(f!, x) = optimize!(f!, x, StoppingCriterion(x))

########################### Stopping Criterion ##################################
# IDEA: have input output be single argument as tuple?
# abstract type StoppingCriterion{T} end
mutable struct StoppingCriterion{T, S}
    x::T # holds last value of parameters
    dx::S # minimum absolute change in parameters x for termination
    rx::S # minimum relative change in parameters x for termination

    y::S # holds last function value
    dy::S
    ry::S

    maxiter::Int # maximum iterations
    verbose::Bool
    function StoppingCriterion(x, y::Real = Inf; dx::Real = 1e-6, rx::Real = 0.,
        dy::Real = 1e-6, ry::Real = 0., maxiter::Int = 1024, verbose::Bool = false)
        new{typeof(x), typeof(y)}(copy(x), dx, rx, y, dy, ry, maxiter, verbose)
    end
end

(T::StoppingCriterion)(x) = input_criterion!(T, x)
function (T::StoppingCriterion)(x, t::Integer)
    T.verbose && println("in iteration $t:")
    1 < t && (input_criterion!(T, x) || T.maxiter < t)
end
function (T::StoppingCriterion)(x, y::Real, t::Integer)
    T.verbose && println("in iteration $t:")
    1 < t && (input_criterion!(T, x) || output_criterion!(T, y) || T.maxiter < t)
end

function input_criterion!(T::StoppingCriterion, x::Union{Real, AbstractVector})
    any(isnan, x) && error("x contains NaN: x = $x")
    dx = norm(difference(x, T.x))
    rx = dx / norm(x)
    if T.verbose
        println("absolute change in x = $dx")
        println("relative change in x = $rx")
    end
    isfixed = dx < T.dx || rx < T.rx
    if x isa Real
        T.x = x
    else # AbstractVector
        T.x .= x
    end
    return isfixed
end
function output_criterion!(T::StoppingCriterion, y::Real)
    isnan(y) && error("y = NaN")
    dy = abs(y - T.y)
    ry = dy / abs(y)
    if T.verbose
        println("absolute change in y = $dy")
        println("relative change in y = $ry")
    end
    isfixed = dy < T.dy || ry < T.ry
    T.y = y
    return isfixed
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
(D::Direction)(x, t::Int) = direction(D, x) # since directions do not depend on time
(D::Direction)(x) = direction(D, x)

update!(D::Update, x, t::Int) = update!(D, x)
update!(D::Direction, x) = (x .+= D(x))
update!(D::Direction, x, t::Int) = (x .+= D(x, t))

# converting a direction to an update
update!(D::Direction) = (x, t::Int) -> (x .+= D(x, t))
fixedpoint!(D::Direction, x, isfixed) = fixedpoint!(update!(D), x, isfixed)

function objective end # returns objective function
objective(D::Direction, x) = D.f(x)
objective(D::Direction) = x->objective(D, x)

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
