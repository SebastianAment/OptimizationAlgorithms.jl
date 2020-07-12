################################################################################
# Gauss-Newton algorithm, minimizes sum of squares of vector-valued function f w.r.t. x
# n, m = length(f(x)), length(x) # in case we want to pre-allocate the jacobian
struct GaussNewton{T, F, D, R, C} <: Direction{T}
    f::F
    d::D
    r::R
    cfg::C
    function GaussNewton(f::F, x::V, y::V) where {T, V<:AbstractVector{T}, F<:Function}
        r = DiffResults.JacobianResult(y, x)
        cfg = JacobianConfig(f, y, x)
        new{T, F, V, typeof(r), typeof(cfg)}(f, similar(x), r, cfg)
    end
end

function valdir(G::GaussNewton, x::AbstractVector)
    y = DiffResults.value(G.r)
    jacobian!(G.r, G.f, y, x, G.cfg)
    J = DiffResults.jacobian(G.r)
    ldiv!(G.d, qr!(J), y)
    G.d .*= -1
    return sum(abs2, y), G.d
end

################################################################################
# TODO: geodesic acceleration (see https://en.wikipedia.org/wiki/Levenberg–Marquardt_algorithm)
struct LevenbergMarquart{T, F, D, R, C, JT} <: Direction{T}
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

function objective(LM::Union{GaussNewton, LevenbergMarquart}, x::AbstractVector)
    y = DiffResults.value(LM.r)
    LM.f(y, x)
    sum(abs2, y)
end

function valdir(LM::LevenbergMarquart, x::AbstractVector, λ::Real = 1e-6)
    y = DiffResults.value(LM.r)
    jacobian!(LM.r, LM.f, y, x, LM.cfg) # constant allocation
    J = DiffResults.jacobian(LM.r)
    mul!(LM.d, J', y)
    mul!(LM.JJ, J', J)
    lm_scaling!(LM.JJ, λ)
    C = cholesky!(LM.JJ) # constant allocation
    ldiv!(C, LM.d) # using y as a temporary
    LM.d .*= -1
    return sum(abs2, y), LM.d # since LevenbergMarquart is for least-squares
end
# const LM = LevenbergMarquart
function lm_scaling!(JJ::AbstractMatrix, λ::Real)
    for i in diagind(JJ)
        JJ[i] += λ * JJ[i]
    end
    JJ
end

# TODO: put settings in struct?
function optimize!(LM::LevenbergMarquart, x::AbstractVector; maxiter::Int = 128,
                min_decrease::Real = 1e-8, max_step::Real = Inf, λ::Real = 1.,
                increase_factor::Real = 3, decrease_factor::Real = 2)
    oldx = copy(x)
    for i in 1:maxiter
        val, dx = valdir(LM, oldx, λ)
        @. x = oldx + dx
        newval = objective(LM, x)
        if newval > val || isnan(newval) || any(x->abs(x) > max_step, dx)
            λ *= increase_factor
            continue
        elseif abs(newval - val) < min_decrease
            break
        end
        val = newval
        copy!(oldx, x)
        λ /= decrease_factor
    end
    return x
end
