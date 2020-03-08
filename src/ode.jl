############################### ODE Integrators ################################
################################# Runge Kutta ##################################
# TODO: adaptive step size, have to make them mutable?
struct Euler{T, F} <: Update{T}
    f::F
    Δ::T
end
const RK1 = Euler
(E::Euler)(x, t) = update(E, x, t)
function update(R::Euler, x, t::Real)
    x += R.Δ * R.f(t, x)
    t += R.Δ
    return x, t
end

struct Midpoint{T, F} <: Update{T}
    f::F
    Δ::T
end
const RK2 = Midpoint
(R::RK2)(x, t) = update(R, x, t)
function update(R::Midpoint, x, t::Real)
    x += R.Δ * R.f(t + R.Δ/2, x + R.Δ/2 * R.f(t, x))
    t += R.Δ
    return x, t
end

struct RungeKutta4{T, F} <: Update{T}
    f::F
    Δ::T
end
const RK4 = RungeKutta4
(R::RK4)(x, t) = update(R, x, t)
function update(R::RK4, x, t::Real)
    k1 = R.Δ * R.f(t, x) # explitic euler
    k2 = R.Δ * R.f(t + R.Δ/2, x + k1/2) # midpoint
    k3 = R.Δ * R.f(t + R.Δ/2, x + k2/2) # midpoint
    k4 = R.Δ * R.f(t + R.Δ, x + k3) # midpoint
    x += (k1 + 2k2 + 2k3 + k4)/6 # round-off lower if we factor out R.Δ?
    t += R.Δ
    return x, t
end

# generalized Runge-Kutta method where A, b, c constitute Butcher tableau
# make N part of type?
struct RungeKutta{T, F, AT<:AbstractMatrix{T}, B, C} <: Update{T}
    f::F
    A::AT # coefficient matrix
    b::B # vector of weightings for intermediate steps, for adaptive, can be matrix (with lower order method embedded)
    c::C # vector of fractional time-steps
    Δ::T
end
const RK = RungeKutta
# general implicit scheme
function update(R::RK, x::Real, t::Real)
    g(k) = sum(abs2, k - @. R.f(t + R.Δ * c, x + R.Δ * A*k)) # equation for k
    k = zero(R.b)
    fixedpoint!(ArmijoStep(Newton(g, k)), k) # if f is linear in k, this is one step
    x += R.Δ * dot(R.b, k)
    t += R.Δ
    return x, t
end
# general explicit scheme (i.e. Butcher tableau is lower triangular)
function update(R::RK{T, F, <:LowerTriangular}, x::Real, t::Real) where {T, F}
    k = zero(R.b)
    for i in eachindex(k)
        k[i] = R.f(t + R.Δ * c[i], x + R.Δ * dot(view(A, i, :), k))
    end
    x += R.Δ * dot(R.b, k)
    t += R.Δ
    return x, t
end

using StaticArrays
# TODO: add second b vector to embed lower order method for error estimation
# TODO: step sizes
function RK(f, Δ::Real, n::Int, explicit = true)
    if explicit
        A = zeros(n, n)
        if n == 1
            b = SVector(1)
            c = SVector(0)
        elseif n == 2
            A[2, 2] = 1/2
            b = SVector(0, 1)
            c = SVector(0, 1/2)
        elseif n == 4
            A[2, 1] = 1/2
            A[3, 2] = 1/2
            A[4, 3] = 1
            A = SMatrix{n,n}(0)
            b = SVector(1/6, 1/3, 1/3, 1/6)
            c = SVector(0, 1/2, 1/2, 1)
        else
            error("order $n RK method not implemented")
        end
        A = LowerTriangular(SMatrix{n,n}(A))
        RK(f, A, b, c, Δ)
    else
        error("implicit RK not implemented")
    end
end
