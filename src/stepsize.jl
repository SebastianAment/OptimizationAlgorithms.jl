# TODO: quasi-exact line search via interpolation, GPs
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
# inputs: searchcondition function, i.e. Armijo, Wolfe
# stepupdate is step size updating policy
function linesearch!(xn::T, x::T, α::Real, direction::T, searchcondition,
                    maxbacktrack::Int, decrease::Real = 3.) where {T}
    for i in 1:maxbacktrack
        @. xn = x + α * direction # update
        if searchcondition(α, xn)
            return true
        end
        α /= decrease
    end
    return false
end

# could also abstract away step update:
# backtrack(α, decrease) = α * decrease
# backtrack(decrease) = α -> backtrackstep(α, decrease)
# integral control?
# struct LineSearch{T, D<:Direction, S, C} <: Update{T}
#     x::T # intermediate storage for value
#     direction::D
#     # stepupdate::S
#     decrease::S
#     searchcondition::C
#     maxbacktrack::Int
# end
#
# function (L::LineSearch)(x)
#     d = direction(x)
#     # cond = L.searchcondition(x, t)
#     linesearch!(L.x, x, d, L.searchcondition, L.maxbacktrack, L.decrease) #L.stepupdate)
# end
# # needs to compute function which returns searchcondition for each x
# # function armijo(d::Direction)
# #     value, direction = A.valdir(x, t...)
# #     grad = A.gradient(x)
# #     cond = armijo_condition(A.objective, value, direction, grad, A.c)
# #     success = linesearch!(A.x, x, A.α, direction, armijo, A.maxbacktrack, A.decrease)
# #     return success ? copy!(x, A.x) : x
# # end

################################################################################
# TODO: what if this instead had a direction field?
struct DecreasingStep{T, V, D, A, S} <: Update{T}
    x::T # storage for value
    objective::V # value function αξία
    valdir::D # value and direction κατεύθυνση
    α::A # last α, has to be 0D array, or armijo is mutable (went with mutable)
    c::S # coefficient in decreasing rule
    decrease::S # μείωση factor
    increase::S # αύξηση factor
    maxbacktrack::Int
end
# function DecreasingStep(x, obj::Function, valdir::Function,
#             α::S = 1., c = 0., decrease = 3.,
#             increase = 2., maxbacktrack::Int = 16) where {S<:Real}
#     # α = fill(α) # creating 0D array
#     x = fill(zero(eltype(x)), size(x))
#     DecreasingStep(x, obj, valdir, α, S(c), S(decrease), S(increase), maxbacktrack)
# end

function DecreasingStep(x, dir::Direction, α::S = 1., c = 1.,
        decrease = 3., increase = 2., maxbacktrack::Int = 16) where {S<:Real}
        DecreasingStep(copy(x), objective(dir), valdir(dir),
                α, S(c), S(decrease), S(increase), maxbacktrack)
end

function update!(D::DecreasingStep, x)
    value, direction = D.valdir(x)
    descent = descent_condition(D.objective, value, direction, D.c)
    success = linesearch!(D.x, x, D.α, direction, descent, D.maxbacktrack, D.decrease)
    return success ? (x .= D.x) : x #copy!(x, D.x) : x
end

# weakest rule, no good theoretical guarantees, but works in practice and is fast
function descent_condition(objective::Function, value, direction::T,
                                                        c::Real = 1.) where {T}
    return descent(α, xn) = (objective(xn) ≤ c * value)
end

################################################################################
# TODO: what if this instead had a direction field?
struct ArmijoStep{T, V, D, G, A, S} <: Update{T}
    x::T # storage for value
    objective::V # value function αξία
    valdir::D # value and direction κατεύθυνση
    gradient::G # need gradient for Armijo rule
    α::A # last α, has to be 0D array, or armijo is mutable (went with mutable)
    c::S # coefficient in Armijo rule
    decrease::S # μείωση factor
    increase::S # αύξηση factor
    maxbacktrack::Int
end
function ArmijoStep(x, obj, valdir,
        grad = x->FD.gradient(obj, x), α::S = 1., c = 0., decrease = 3.,
        increase = 2., maxbacktrack::Int = 16) where {S}
    # α = fill(α) # creating 0D array
    x = fill(zero(eltype(x)), size(x))
    ArmijoStep(copy(x), obj, valdir, grad, α,
                S(c), S(decrease), S(increase), maxbacktrack)
end

function ArmijoStep(x, dir::Direction,
        grad = x->FD.gradient(objective(dir), x), α::S = 1., c = 0.,
        decrease = 3., increase = 2., maxbacktrack::Int = 16) where {S}
        ArmijoStep(copy(x), objective(dir), valdir(dir), grad,
                    α, S(c), S(decrease), S(increase), maxbacktrack)
end

function update!(A::ArmijoStep, x)
    value, direction = A.valdir(x)
    grad = A.gradient(x)
    armijo = armijo_condition(A.objective, value, direction, grad, A.c)
    success = linesearch!(A.x, x, A.α, direction, armijo, A.maxbacktrack, A.decrease)
    return success ? copy!(x, A.x) : x
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

# ε is smallest allowable step size
# function update!(A::ArmijoStep, x, t::Int...)
#     f = A.val
#     y, d = A.valdir(x, t...) # only t dependence
#     dg = typeof(A.d) <: Gradient ? d'd : d'gradient(f, x) # TODO: generalize FD gradient?
#     for i in 1:A.maxbacktrack
#         @. A.x = x + A.α * d
#         if (f(A.x) ≤ y + A.α * A.c * dg) #|| α < ε # rule for minimization
#             copy!(x, A.x)  # only if we jump out with a good step size, reassign A.x to x
#             break
#         end
#         A.α ./= A.decrease
#     end
#     # A.α = α * A.increase
#     return x
# end
