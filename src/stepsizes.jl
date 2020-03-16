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
function direction!(C::ConstantStep, x)
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
