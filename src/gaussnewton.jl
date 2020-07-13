abstract type AbstractGaussNewton{T} <: Direction{T} end

function objective(GN::AbstractGaussNewton, x::AbstractVector)
    y = DiffResults.value(GN.r)
    GN.f(y, x)
    sum(abs2, y)
end

function get_value_jacobian(GN::AbstractGaussNewton)
    return DiffResults.value(GN.r), DiffResults.jacobian(GN.r)
end

function update_jacobian!(GN::AbstractGaussNewton, x::AbstractVector)
    y, J = get_value_jacobian(GN)
    jacobian!(GN.r, GN.f, y, x, GN.cfg) # constant allocation
    return y, J
end

# update_jacobian! needs to be called prior to update_direction for correctness
function update_direction!(GN::AbstractGaussNewton, λ::Real, min_diagonal::Real = 1e-12)
    y, J = get_value_jacobian(GN)
    mul!(GN.d, J', y)
    mul!(GN.JJ, J', J)
    lm_scaling!(GN.JJ, λ, min_diagonal)
    C = cholesky!(GN.JJ) # constant allocation
    ldiv!(C, GN.d)
    GN.d .*= -1
    return sum(abs2, y), GN.d
end

function lm_scaling!(JJ::AbstractMatrix, λ::Real, min_diagonal::Real = 1e-12)
    for i in diagind(JJ)
        JJ[i] = max(JJ[i] * (1+λ), min_diagonal)
    end
    JJ
end

################################################################################
# Gauss-Newton algorithm, minimizes sum of squares of vector-valued function f w.r.t. x
struct GaussNewton{T, F, D, R, C, JT} <: AbstractGaussNewton{T}
    f::F
    d::D
    r::R
    cfg::C
    JJ::JT
    function GaussNewton(f::F, x::V, y::V) where {T, V<:AbstractVector{T}, F<:Function}
        d = similar(x)
        r = DiffResults.JacobianResult(y, x)
        cfg = JacobianConfig(f, y, x)
        J = DiffResults.jacobian(r)
        JJ = J'J
        new{T, F, V, typeof(r), typeof(cfg), typeof(JJ)}(f, d, r, cfg, JJ)
    end
end

# using qr factorization instead of cholesky of normal equations
function valdir(GN::GaussNewton, x::AbstractVector, qr::Val{true})
    y, J = update_jacobian!(GN, x)
    ldiv!(GN.d, qr!(J), y)
    GN.d .*= -1
    return sum(abs2, y), GN.d
end

function valdir(GN::GaussNewton, x::AbstractVector, min_diagonal::Real = 1e-12)
    update_jacobian!(GN, x)
    update_direction!(GN, 0, min_diagonal)
end

################################################################################
# TODO: geodesic acceleration (see https://en.wikipedia.org/wiki/Levenberg–Marquardt_algorithm)
struct LevenbergMarquart{T, F, D, R, C, JT} <: AbstractGaussNewton{T}
    f::F
    d::D
    r::R
    cfg::C # Jacobian config
    JJ::JT
    function LevenbergMarquart(f::F, x::V, y::V) where {T, V<:AbstractVector{T}, F<:Function}
        d = similar(x)
        r = DiffResults.JacobianResult(y, x)
        cfg = JacobianConfig(f, y, x)
        J = DiffResults.jacobian(r)
        JJ = J'J
        new{T, F, V, typeof(r), typeof(cfg), typeof(JJ)}(f, d, r, cfg, JJ)
    end
end

function valdir(LM::LevenbergMarquart, x::AbstractVector, λ::Real, min_diagonal::Real = 1e-12)
    update_jacobian!(LM, x)
    update_direction!(LM, λ, min_diagonal)
end

# TODO: put settings in struct?
function optimize!(LM::LevenbergMarquart, x::AbstractVector; maxiter::Int = 128,
                min_decrease::Real = 1e-8, max_step::Real = Inf, λ::Real = 1.,
                increase_factor::Real = 3, decrease_factor::Real = 2,
                min_diagonal::Real = 1e-12)
    oldx = copy(x)
    update_jacobian!(LM, oldx)
    for i in 1:maxiter
        val, dx = update_direction!(LM, λ, min_diagonal)
        @. x = oldx + dx
        newval = objective(LM, x)
        if newval > val || isnan(newval) || any(x->abs(x) > max_step, dx) # might be easier to read with nested loop
            λ *= increase_factor
            continue
        elseif abs(newval - val) < min_decrease
            break
        end
        val = newval
        copy!(oldx, x)
        update_jacobian!(LM, oldx)
        λ /= decrease_factor
    end
    return x
end
