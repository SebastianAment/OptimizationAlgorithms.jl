# TODO: quasi-exact line search via interpolation, GPs
# IDEA: g .*= -g'g / (g'A*g) # optimal stepsize for quadratic problems

################################################################################
# inputs: searchcondition function, i.e. Armijo, Wolfe
# stepupdate is step size updating policy
function linesearch!(xn::AbstractVector, x::AbstractVector, α::Real,
                     direction::AbstractVector, searchcondition, maxbacktrack::Int, decrease::Real = 3.)
    for i in 1:maxbacktrack
        @. xn = x + α * direction # update
        if searchcondition(α, xn)
            return true
        end
        α /= decrease
    end
    return false
end
# linesearch for scalar parameters
function linesearch(x::Real, α::Real, direction::Real, searchcondition,
                    maxbacktrack::Int, decrease::Real = 3.)
    for i in 1:maxbacktrack
        xn = x + α * direction # update
        if searchcondition(α, xn)
            return xn
        end
        α /= decrease
    end
    return x
end
########################### linesearch types ###################################
struct DecreasingStep{T, D, A<:Real, S<:Real} <: Update{T}
    direction::D # αξία και κατεύθυνση
    x::T # storage for value
    α::A # stepsize
    c::S # coefficient in decreasing rule
    decrease::S # μείωση factor
    # increase::S # αύξηση factor
    maxbacktrack::Int
    function DecreasingStep(direction::Direction, x, α::Real = 1., c::Real = 1.0,
                            decrease::Real = 3.0, maxbacktrack::Int = 16) # increase::Real = 2,
        T, D, A, S = typeof(x), typeof(direction), typeof(α), promote_type(typeof(c), typeof(decrease)) #, typeof(increase))
        new{T, D, A, S}(direction, copy(x), α, c, decrease, maxbacktrack) # makes sure that we use a copy of x
    end
end

objective(D::DecreasingStep) = objective(D.direction)
function update!(D::DecreasingStep, x::AbstractVector)
    val, dir = valdir(D.direction, x)
    descent = DescentCondition(objective(D), val, x, D.c)
    success = linesearch!(D.x, x, D.α, dir, descent, D.maxbacktrack, D.decrease)
    return success ? copyto!(x, D.x) : x
end
# update for scalar parameter
function update!(D::DecreasingStep, x::Real)
    val, dir = valdir(D.direction, x)
    descent = DescentCondition(objective(D), val, x, D.c)
    xn = linesearch(x, D.α, dir, descent, D.maxbacktrack, D.decrease)
    success = xn != x
    return success ? xn : x
end

# combines descent condition and local minimum w.r.t. step size α condition
mutable struct DescentCondition{T, F, X}
    f::F # objective function
    value::T # value f(x)
    new_value::T # value at most recently proposed point f(xn)
    xn::X
    c::T # descent factor
end
function DescentCondition(objective::Function, value, x, c::Real = 1.)
    DescentCondition(objective, value, typeof(value)(Inf), zero(x), c)
end
function (D::DescentCondition)(α, xn)
    decreasing = D.new_value ≤ D.c * D.value # was the last proposal decreasing
    new_value = D.f(xn)
    alpha_minimum = D.new_value ≤ new_value # was the last proposal a stationary point w.r.t. α?
    if decreasing && alpha_minimum
        if xn isa Real
            xn = D.xn
        else
            xn .= D.xn # assign last proposal to current one
        end
        return true
    else
        if xn isa Real
            D.xn = xn
        else
            D.xn .= xn
        end
        D.new_value = new_value
        return false
    end
end

# weakest rule, no good theoretical guarantees, but works in practice and is fast
function descent_condition(objective::Function, value, c::Real = 1.)
    return descent(α, xn) = (objective(xn) ≤ c * value)
end

################################################################################
# TODO: what if this instead had a direction field?
struct ArmijoStep{T, D, G, A, S} <: Update{T}
    direction::D # αξία και κατεύθυνση
    x::T # storage for value
    gradient::G # need gradient for Armijo rule
    α::A # last α, has to be 0D array, or armijo is mutable (went with mutable)
    c::S # coefficient in Armijo rule
    decrease::S # μείωση factor
    maxbacktrack::Int
    function ArmijoStep(dir::D, x::T, grad::G = x->FD.gradient(objective(dir), x),
            α::S = 1., c = 0., decrease = 3., maxbacktrack::Int = 16) where {T, D, G, S}
            new{T, D, G, S, S}(dir, copy(x), grad, α, S(c), S(decrease), maxbacktrack)
    end
end
# function ArmijoStep(x, obj, valdir,
#         grad = x->FD.gradient(obj, x), α::S = 1., c = 0., decrease = 3.,
#         increase = 2., maxbacktrack::Int = 16) where {S}
#     # α = fill(α) # creating 0D array
#     x = fill(zero(eltype(x)), size(x))
#     ArmijoStep(copy(x), obj, valdir, grad, α,
#                 S(c), S(decrease), S(increase), maxbacktrack)
# end
objective(A::ArmijoStep) = objective(A.direction)
function update!(A::ArmijoStep, x)
    val, dir = valdir(A.direction, x)
    grad = A.gradient(x)
    armijo = armijo_condition(objective(A), val, dir, grad, A.c)
    success = linesearch!(A.x, x, A.α, dir, armijo, A.maxbacktrack, A.decrease)
    return success ? copyto!(x, A.x) : x
end

# returns a function which calculates the armijo condition
# if direction = gradient, pass it to the function to avoid duplicate calculation
# value = objective(x)
function armijo_condition(objective::Function, value, direction::T, ∇::T,
                                                    c::Real = 1e-4) where {T}
    direction_gradient = dot(direction, ∇) # just to make sure its eager
    return armijo(α, xn) = (objective(xn) ≤ value + α * c * direction_gradient)
end

################################################################################
# could also abstract away step update:
# backtrack(α, decrease) = α * decrease
# backtrack(decrease) = α -> backtrackstep(α, decrease)
# integral control?
# AKA InexactLinesearch, could have separate type for exact
# struct LineSearch{T, D<:Direction, S, C} <: Update{T}
#     direction::D
#     x::T # intermediate storage for value
#     # stepupdate::S
#     searchcondition::C
#     decrease::S
#     maxbacktrack::Int
#     function LineSearch(dir::D, x::T, α::S = 1., c = 1., decrease = 3.,
#                     maxbacktrack::Int = 16) where {T, D<:Direction, S}
#         # makes sure that we use a copy of x
#         new{T, D, S, C}(dir, copy(x), S(decrease), maxbacktrack)
#     end
# end
# function update!(L::LineSearch, x)
#     val, dir = valdir(L.direction, x)
#
#     d = L.direction(x)
#     # cond = L.searchcondition(x, t)
#     descent = L.search_condition(objective(D), val, dir, D.c)
#     success = linesearch!(L.x, x, dir, L.searchcondition, L.maxbacktrack, L.decrease) #L.stepupdate)
#     return success ? copy!(x, D.x) : x
# end
# needs to compute function which returns searchcondition for each x
# function armijo(d::Direction)
#     value, direction = A.valdir(x, t...)
#     grad = A.gradient(x)
#     cond = armijo_condition(A.objective, value, direction, grad, A.c)
#     success = linesearch!(A.x, x, A.α, direction, armijo, A.maxbacktrack, A.decrease)
#     return success ? copy!(x, A.x) : x
# end

################################################################################
# strongest condition, theoretical guarantees
function wolfe_condition(x::T, objective::Function, direction::T, ∇::T,
                            c₁::Real = 1e-4, c₂::Real = .99) where {T}
    fx = objective(x)
    direction_gradient = dot(direction, ∇) # just to make sure its eager
    function wolfe(α, xn, ∇n = gradient(f, xn))
        armijo = (objective(xn) ≤ fx + α * c₁ * direction_gradient)
        curv = abs(dot(direction, ∇n)) ≤ abs(c₂*direction_gradient) # weak rule: -d'∇(x + α*d) ≤ -c*d'∇(x)
        return armijo && curv
    end
    return wolfe
end

# TODO: Could rename Damped, so type reads e.g. Damped{Newton}
# 0 < c₁ < c₂ < 1
# mutable struct WolfeStep{T, D<:Direction{T}} <: Update{T}
#     d::D
#     c₁::T
#     c₂::T
#     α::T
#     decrease::T # μείωση factor
#     increase::T # αύξηση factor
#     maxbacktrack::Int
# end
# function WolfeStep(d::D, c::T = .99, α = 1., decrease = 7., increase = 2.,
#                             maxbacktrack::Int = 16) where {T, D<:Direction}
#     ArmijoStep{T, D}(d, c, T(α), T(decrease), T(increase), maxbacktrack)
# end

############################ Step Size Selectors ###############################
# generalizes search for an α such that (x - α * direction) leads to minimizer
# struct ConstantStep{T, D<:Direction} <: Direction{T}
#     d::D
#     α::T
# end
# # decide if I want to represent the multiplication lazily
# # direction(C::ConstantStep) = C.α * direction(C.d) # introduces temporary
# function direction!(C::ConstantStep, x)
#     direction!(C.d, x) # update direction vector
#     direction(C.d) .*= C.α # multiply direction with step-size
# end
# function update!(C::ConstantStep, x::AbstractVector)
#     direction!(C.d, x) # update direction vector
#     x .+= C.α .* direction(C.d)
# end

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

################################################################################
# two most important rules to consider when selecting step sizes
# function armijo_rule(x, f::Function, d, c, α, ∇ = gradient(f, x))
#     f(x + α*d) ≤ f(x) + α * c * (d'∇) # RHS could be computed independently of α
# end

# direction_gradient is dot product of the two
# function armijo_rule(xn, value::Real, f, direction_gradient, c::Real, α::Real)
#     f(xn) ≤ value + α * c * direction_gradient
# end
# implements the strong wolfe
# function curvature_rule(x, f, d, c, α)
#     abs(d'gradient(f, x + α*d)) ≤ abs(c*d'gradient(f, x)) # weak rule: -d'∇(x + α*d) ≤ -c*d'∇(x)
# end
# function curvature_rule(xn, f, dg, c, α)
#     abs(d'gradient(f, xn)) ≤ abs(c*dg) # weak rule: -d'∇(x + α*d) ≤ -c*d'∇(x)
# end

################################## WIP #########################################
# the minimization is expensive as we have to reevaluate the gradient of the objective
# better with interpolation?
# step size calculated with approximate minimization via secant method
# struct SecantStep{T, D<:Direction{T}} <: Update{T}
#     direction::D
# end
#
# objective(D::SecantStep) = objective(D.direction)
# function update!(D::SecantStep, x)
#     val, dir = valdir(D.direction, x)
#     descent = descent_condition(objective(D), val, dir, D.c)
#     success = linesearch!(D.x, x, D.α, dir, descent, D.maxbacktrack, D.decrease)
#     # α, success = linesearch!(D.x, x, D.α, dir, descent, D.maxbacktrack, D.decrease)
#     # D.α = α * D.increase
#     function value(α)
#         @. xn = x + α * direction # update
#         value(D, xn)
#     end
#     function value_direction(α)
#         @. xn = x + α * direction # update
#         val, dir = value_direction(D, xn)
#         val, dot(dir, direction)
#     end
#     return success ? copy!(x, D.x) : x
# end
