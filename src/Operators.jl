module Operators
using LinearAlgebra
import TaylorSeries # importing avoids name conflict with ∇
const TS = TaylorSeries
import ForwardDiff
const FD = ForwardDiff

# TODO: integral operators
# TODO: famous differential equations, Maxwell, Schrödinger, Heat,

# traits for function output types
abstract type OutputValue end
struct ScalarValued <: OutputValue end
struct VectorValued <: OutputValue end
struct FunctionValued <: OutputValue end

# defaults to scalar value
outputvalue(::Function) = ScalarValued()
outputvalue(::AbstractVector) = VectorValued()
outputvalue(::Number) = ScalarValued()
# might include input types

# outputvalue(::curl) = VectorValued()

# do we need T? is it the input type to the function?
abstract type Operator{T} end # vector -> vector
abstract type Functional{T} <: Operator{T} end # vector -> scalar
function islinear end
# operator traits
islinear(::Operator) = false
isfunctional(::Operator) = false
isfunctional(::Functional) = true

############################ Evaluation Functional #############################
struct Evaluation{T} <: Functional{T}
    x::T
end
Evaluation() = Evaluation(0.)
(L::Evaluation)(f) = delta(f, L.x)
islinear(::Evaluation) = true

# functional design
delta(f::Function, x) = f(x)
delta(x) = f -> f(x)
const δ = delta
islinear(::typeof(δ)) = true

######################## linear algebra of operators ###########################
struct Sum{T, AT<:Tuple{Vararg{Operator}}} <: Operator{T}
    args::AT
    function Sum(args::Tuple{Vararg{Operator}})
        T = Float64 #promote_type(args)
        new{T, typeof(args)}(args)
    end
end
Sum(L::Operator...) = Sum(L)
(S::Sum)(f) = isfunctional(S) ? sum(L->L(f), S.args) : x -> sum(L->L(f)(x), S.args)
islinear(S::Sum) = all(islinear, S.args)
isfunctional(S::Sum) = all(isfunctional, S.args)
Base.:+(L::Operator...) = Sum(L)

# TODO: input can be function? should this instead scale an operator?
struct Scaling{T, F, AT <: Operator{T}} <: Operator{T}
    α::F
    L::AT
end
# (S::Scaling)(f) = isfunctional(S) ? S.α * S.L(f) : x -> S.α * S.L(f)(x)
(S::Scaling{<:Any, <:Number})(f) = isfunctional(S.L) ? S.α * S.L(f) : x -> S.α * S.L(f)(x)
(S::Scaling{Any, Function})(f) = isfunctional(S.L) ? x -> S.α(x) * S.L(f) : x -> S.α(x) * S.L(f)(x)
Base.:*(α::Union{Real, Function}, L::Operator) = Scaling(α, L)
islinear(S::Scaling) = islinear(S.L)
isfunctional(S::Scaling) = isfunctional(S.L) && typeof(S.α) <: Number

# AT is either <:Tuple{Vararg{Operator}} or AbstractVector{Operator}
struct Dot{T, V<:AbstractVector{T}, AT} <: Operator{T}
    α::V
    args::AT
end
# TODO: make vector operator broadcasting work
(D::Dot{Number})(f) = x -> dot(D.α, D.args.(f)(x))
(D::Dot{Function})(f) = x -> dot(D.α(x), D.args.(f)(x))
islinear(D::Dot) = all(islinear, D.args)
isfunctional(D::Dot) = all(isfunctional, S.args)
Base.:*(α::AbstractVector, L::AbstractVector{Operator}) = Dot(α, L)

# Base.:+(L::Operato, α::Real)= Sum(_->α, f) # works for scalars
# this acts on functions
struct Shift{T} <: Operator{T}
    s::T
end
(L::Shift)(f) = x->f(x-L.s)
islinear(::Shift) = true
isfunctional(::Shift) = false

# technically, could do it without own type, just using Base.:∘
# however, we would not be able to check for properties of composition
# TODO: if first is Functional, enforce all intermediate ones to be Operators?
# TODO: could also convert this to a single functional upon construction
# and store properties in object, meh
struct Composition{T, AT} <: Operator{T}
    args::AT
    function Composition(args::Tuple{Vararg{Operator}})
        T = Float64 #promote_type(args)
        new{T, typeof(args)}(args)
    end
end
Composition(L::Operator...) = Composition(L)
(C::Composition)(f) = ∘(C.args...)(f)
islinear(C::Composition) = all(islinear, C.args)
isfunctional(C::Composition) = isfunctional(C.args[1])
# Base.:∘(L::Operator...) = Composition(L)

########################## Differential Operators ##############################
abstract type DifferentialOperator{T} <: Operator{T} end
abstract type IntegralOperator{T} <: Operator{T} end
# IntegroDifferentialOperator
# could add: adjoint, Schwarzian derivative
# TODO: Zygote implementation of differential operators
####################### scalar differential operators ##########################
# TODO: write vector versions of del, laplacian, derivatives, partials
# add dimension?
struct Del{T} <: DifferentialOperator{T} end
Del() = Del{Float64}()
islinear(::Del) = true
(::Del)(f) = del(f)

# FD implementation of del
function del end
const ∇ = del

# trait design
del(f, x::Real) = FD.derivative(f, x)
del(f, x::AbstractVector) = del(f, x, outputvalue(f))
del(f, x::AbstractVector, ::ScalarValued) = FD.gradient(f, x)
del(f, x::AbstractVector, ::VectorValued) = FD.jacobian(f, x)

del(f) = x -> del(f, x)
islinear(::typeof(del)) = true

# TODO: figure out correct definition of vector laplacian
# scalar laplacian of each component ...
# this incurs quadratic cost in dimension
# could have vector of 1D taylor variables instead!
struct Laplacian{T} <: DifferentialOperator{T} end
Laplacian() = Laplacian{Float64}()
islinear(::Laplacian) = true
(::Laplacian)(f) = laplacian(f)

# FD implementation
laplacian(f, x) = laplacian(f, x, outputvalue(f))
laplacian(f, x::Real, ::ScalarValued) = del(del(f), x)
laplacian(f, x, ::ScalarValued) = div(del(f), x)

laplacian(f, x::AbstractVector, ::ScalarValued) = div(del(f), x)


# trait design
# laplacian(f, x) = del(f, x, outputvalue(f))
# laplacian(f, x::Real, ::ScalarValued) = FD.derivative(f, x)
# laplacian(f, x, ::ScalarValued) = FD.gradient(f, x)
# laplacian(f, x::Real, ::VectorValued) = FD.derivative.(f, x)
# laplacian(f, x, ::VectorValued) = FD.jacobian(f, x)

laplacian(f) = x -> laplacian(f, x)
islinear(::typeof(laplacian)) = true
const Δ = laplacian
const ∇² = laplacian

####################### vector differential operators ##########################
# TODO: add divergence in cylindrical and spherical coordinates?
# TODO: divergence of tensor fields
# divergence: sum of component-wise derivative
struct Divergence{T} <: DifferentialOperator{T} end
const Div = Divergence
Div() = Div{Float64}()
islinear(::Divergence) = true
(::Divergence)(f) = divergence(f)

function divergence end
const div = divergence
islinear(::typeof(div)) = true
# overloading dot to make (∇ ⋅ f) = div(f)
LinearAlgebra.dot(::Union{typeof(del), Del}, f) = x -> div(f, x)

# FD implementation
function div(f, x::AbstractVector)
    J = FD.jacobian(f, x)
    size(J, 1) == size(J, 2) || error("divergence only defined for equal input and output dimension, but Jacobian is not square: $(size(J))")
    tr(J)
end
div(f, x::Real, ::ScalarValued) = del(f, x, ScalarValued())
div(f) = x -> div(f, x)


# look at Maxwell-Faraday equation
struct Curl{T} <: DifferentialOperator{T} end
Curl() = Curl{Float64}()
const Rot = Curl
islinear(::Curl) = true
(::Curl)(f) = curl(f)

function curl end
const rot = curl
# overloading cross to make (∇ × f) = rot(f) work
LinearAlgebra.cross(::Union{typeof(del), Del}, f) = x -> curl(f, x)

# could instead or also have, Zygote would allow for most efficient implementation
function curl(f, x::Number, y::Number, z::Number)
    curl(f, [x, y, z]) # would need to replace this
end
# FD implementation
# TODO: could allow 2D input
function curl(f, x::AbstractVector) # NTuple{3, Real}}
    J = FD.jacobian(f, x)
    size(J) == (3, 3) || error("curl only defined for three dimensional input and output,
                                but Jacobian is of size $(size(J))")
    y = similar(x)
    # y = zeros(3)
    y[1] = J[3, 2] - J[2, 3]
    y[2] = J[1, 3] - J[3, 1]
    y[3] = J[2, 1] - J[1, 2]
    return y
end
curl(f) = x -> curl(f, x)
islinear(::typeof(curl)) = true

################### more general linear differential operators #################
# BUG: TaylorSeries suffers from perturbation confusion if used in a nested way
# represents all derivatives up to order N, including the value of scalar valued function
struct Derivatives{T, N} <: Operator{T} end
(::Derivatives{T, N})(f) where {T, N} = x -> derivatives(f, x, Val(N))
Base.length(::Derivatives{T, N}) where {T, N} = N+1
function derivatives(f, x::Real, ::Val{N}) where {N}
    df = f(x + TS.Taylor1(N))
    [derivative(i, df) for i in 0:N]
end
islinear(::Derivatives) = true
# TODO: specialize for kernels
# function derivatives(k::AbstractKernel, x::Real, y::Real, ::Val{N}) where {N}
#     t = set_variables("t", order = N, numvars = 2)
#     kxy = k(x + t[1], y + t[2])
#     return -1
# end

# represents all partial derivatives of order up to N, including the value
struct Partials{T, N} <: Operator{T} end
# TODO: this is more complex
# d is dimension of the input
Base.length(::Partials{T, N}, d::Int) where {T, N} = sum(d^n for n in 0:N)

# if input is one-dimensional, reduces to derivatives (requires scalar-valued function)
partials(f, x::Real, n::Val{N}) where N = derivatives(f, x, n)

# first and second derivatives are probably most efficient with FD
# TODO: overcome problem with vector valued functions, which we need for kernels!
# to reduce allocations, could use view trick as in multi.jl
function partials(f, x, ::Val{1})
    result = DiffResults.GradientResult(x)
    FD.gradient!(result, f, x)
    vcat(result.value, result.derivs[1]) # d+1
end
# function partials(f, x, ::Val{N}) where N
#     numvars = length(x)
#    df = f(x .+ set_variables("x", order = order(L), numvars = numvars))  # laziness could be advantageous here
#    factorial(id) # for each dimension
#    sum(i->df.coeffs[i+1] * factorial(i) * L.α[i+1], 0:order(L)) # α[i](x)
# end

# ordinary differential operator
# struct ODE{T, V<:AbstractVector{T}} <: DifferentialOperator{T}
#     α::V
#     # could add temporary storage
# end
#
# function (L::ODE)(f, x::Real)
#     df = x + f(TS.Taylor1(order(L))) # laziness could be advantageous here
#     sum(i->df.coeffs[i+1] * factorial(i) * L.α[i+1], 0:order(L)) # α[i](x)
# end
#
# struct PDE{T, V} <: DifferentialOperator{T}
#     α::V
#     # could add temporary storage
# end
#
# function (L::PDE)(f, x::AbstractVector)
#     numvars = length(x)
#     df = f(x .+ set_variables("x", order = order(L), numvars = numvars))  # laziness could be advantageous here
#     factorial(id) # for each dimension
#     sum(i->df.coeffs[i+1] * factorial(i) * L.α[i+1], 0:order(L)) # α[i](x)
# end
#
# order(L::Union{ODE, PDE}) = length(L.α)-1
# islinear(::Union{ODE, PDE}) = true

end # Operator

# TaylorSeries implementations of differential operators

# del(f, x::Real) = 2f(x + Taylor1(1)).coeffs[2]
# function del(f, x::AbstractVector)
#     df = f(x .+ TS.set_variables("x", order = 2, numvars = length(x)))
#     g = TS.gradient(df)
#     TS.evaluate.(TS.derivative.(g, 1:length(x)))
# end

# Taylor Series implementation
# function laplacian(f, x::Real)
#     df = f(x + TS.Taylor1(2))
#     2*df.coeffs[3]
# end
# function laplacian(f, x)
#     df = f(x .+ TS.set_variables("x", order = 2, numvars = length(x)))
#     g = TS.gradient(df)
#     TS.evaluate.(TS.derivative.(g, 1:length(x)))
# end

# function divergence(f, x)
#     # sum of component-wise derivative
#     df = L.f(x .+ TS.set_variables("x", order = 1, numvars = length(x)))
#     sum(k->TS.derivative(df[k], k), 1:length(x))
# end

# TS implementation
# function curl(f, x) # NTuple{3, Real}}
#     length(x) == 3 || error("curl needs three dimensional input, but length(x) = $(length(x))")
#     df = f(x .+ TS.set_variables("x", order = 1, numvars = length(x)))
#     y = similar(x)
#     @inline cross(i::Int, j::Int) = evaluate(TS.derivative(df[i], j))
#     y[1] = cross(3, 2) - cross(2, 3)
#     y[2] = cross(1, 3) - cross(3, 1)
#     y[3] = cross(2, 1) - cross(1, 2)
#     return y
# end
