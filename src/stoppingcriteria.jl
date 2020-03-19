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
