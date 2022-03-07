abstract type AbstractGaussNewton{T} <: Direction{T} end

function objective!(GN::AbstractGaussNewton, y::AbstractVector, x::AbstractVector)
	norm(GN.f(y, x))
end
# WARNING overwrites value vector
function objective!(GN::AbstractGaussNewton, x::AbstractVector)
	y = DiffResults.value(GN.r)
    norm(GN.f(y, x))
end
function objective(GN::AbstractGaussNewton, x::AbstractVector)
	y = similar(DiffResults.value(GN.r))
	norm(GN.f(y, x))
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
    !(isinf(λ) || isnan(λ)) || throw(DomainError("$λ: has to be finite"))
    y, J = get_value_jacobian(GN)
    mul!(GN.d, J', y)
    mul!(GN.JJ, J', J)
    lm_scaling!(GN.JJ, λ, min_diagonal)
    C = cholesky!(GN.JJ) # constant allocation
    ldiv!(C, GN.d)
    GN.d .*= -1
    return y, GN.d, J, C
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
    d::D # pre-allocation for parameter direction vector
    # y::Y # pre-allocation for output / residual vector
    r::R # diff results
    cfg::C # Jacobian config
    JJ::JT # pre-allocation for Hessian approximation
    function GaussNewton(f::F, x::U, y::V) where {T, U<:AbstractVector{T},
                                                V<:AbstractVector, F<:Function}
        d = similar(x)
        r = DiffResults.JacobianResult(y, x)
        cfg = JacobianConfig(f, y, x)
        J = DiffResults.jacobian(r)
        JJ = J'J
        new{T, F, U, typeof(r), typeof(cfg), typeof(JJ)}(f, d, r, cfg, JJ)
    end
end

# using qr factorization instead of cholesky of normal equations
function valdir(GN::GaussNewton, x::AbstractVector, qr::Val{true})
    y, J = update_jacobian!(GN, x)
    ldiv!(GN.d, qr!(J), y)
    GN.d .*= -1
    return norm(y), GN.d
end

function valdir(GN::GaussNewton, x::AbstractVector, min_diagonal::Real = 1e-12)
    update_jacobian!(GN, x)
    y, d, _, _ = update_direction!(GN, 0, min_diagonal)
	return norm(y), GN.d
end

################################################################################
# TODO: geodesic acceleration (see https://en.wikipedia.org/wiki/Levenberg–Marquardt_algorithm)
struct LevenbergMarquart{T, F, D, R, C, JT} <: AbstractGaussNewton{T}
    f::F
    d::D # pre-allocation for parameter direction vector
    r::R # DiffResult
    cfg::C # Jacobian config
    JJ::JT # pre-allocation for Hessian approximation
    function LevenbergMarquart(f::F, x::U, y::V) where {T, U<:AbstractVector{T},
                                                V<:AbstractVector, F<:Function}
		d = similar(x)
        r = DiffResults.JacobianResult(y, x)
        cfg = JacobianConfig(f, y, x)
        J = DiffResults.jacobian(r)
        JJ = J'J
        new{T, F, U, typeof(r), typeof(cfg), typeof(JJ)}(f, d, r, cfg, JJ)
    end
end

function valdir(LM::LevenbergMarquart, x::AbstractVector, λ::Real, min_diagonal::Real = 1e-12)
    update_jacobian!(LM, x)
    update_direction!(LM, λ, min_diagonal)
end

################################################################################
# settings for LM optimization algorithm
struct LevenbergMarquartSettings{T<:Real}
	max_iter::Int # maximum number of iterations
	min_iter::Int # see min_decrease below
	max_backtrack::Int # maximum number of consecutive backtracks (λ increases)

	min_resnorm::T # minimum norm of residual vector for termination
	min_res::T # elementwise minimum of residual vector for termination

	min_decrease::T # minimum decrease in residual norm for termination
	 # ... after at least miniter iterations have been executed

	max_step::T # maximum norm of step proposal
	min_diagonal::T # minimum diagonal entries of Hessian approximation

	min_λ::T
	max_λ::T

	increase_factor::T # factor by which λ is increased
	decrease_factor::T
end

function LevenbergMarquartSettings(;max_iter::Int = 128, max_backtrack::Int = 32,
	min_resnorm::Real = 1e-6, min_res::Real = 1e-6,
	min_decrease::Real = 1e-12, min_iter::Int = 1,
	max_step::Real = Inf, min_diagonal::Real = 1e-12,
	min_lambda::Real = 1e-12, max_lambda::Real = 1e12,
	increase_factor::Real = 10, decrease_factor::Real = 7)

	LevenbergMarquartSettings(max_iter, min_iter, max_backtrack,
								promote(min_resnorm, min_res,
								min_decrease, max_step, min_diagonal,
								min_lambda, max_lambda,
								increase_factor, decrease_factor)...)
end

################################################################################

function optimize!(LM::LevenbergMarquart{<:Real},
			x::AbstractVector, y::AbstractVector,
			stn::LevenbergMarquartSettings = LevenbergMarquartSettings(),
			λ::Real = 1e-6, verbose::Union{Val{true}, Val{false}} = Val(false), geodesic::Bool=false)
    oldx = copy(x)
	val, newval = objective!(LM, y, x), Inf
    for i in 1:stn.max_iter
		update_jacobian!(LM, oldx)
		newval, λ = lm_backtrack!(LM, x, oldx, y, λ, stn, geodesic)
		verbose isa Val{false} || lm_verbose(i, newval, val, λ, y)
		if converged(stn, newval, val, y, i)
			return x, i
		end
		val = newval
        copy!(oldx, x)
	end
	return x, stn.max_iter
end

function lm_backtrack!(LM::LevenbergMarquart,
				x::AbstractVector, oldx::AbstractVector, y::AbstractVector,
	 								λ::Real, stn::LevenbergMarquartSettings, geodesic::Bool = false)
	val = 0.
	h = 0.01
	α = 0.75
	temp = zero(y)
	temp2 = zero(y)
	r = zero(y)
	dx2 = zero(x)
	
	for j in 1:stn.max_backtrack
		w, dx, J, C = update_direction!(LM, λ, stn.min_diagonal)
		val = norm(w)
		if geodesic
            r = 2/h * ( (LM.f(temp2, oldx + h.*dx)- w)/h - mul!(temp, J, dx) )
		    mul!(dx2, J', r)
			ldiv!(C, dx2)
		end
		if !geodesic || (geodesic && 2(norm(dx2))/norm(dx) < α)
			@. x = oldx + dx - 0.5dx2
			newval = objective!(LM, y, x) # norm(LM.f(y, x))
			if !(newval ≥ val || isnan(newval) || any(x->abs(x) > stn.max_step, dx - 0.5dx2))
				λ = max(λ / stn.decrease_factor, stn.min_λ)
				return newval, λ
			end
		end
		λ = min(λ * stn.increase_factor, stn.max_λ)
	end
	copy!(x, oldx) # if backtracking wasn't successful
	return val, λ
end

function converged(stn::LevenbergMarquartSettings, newval::Real, val::Real,
						y::AbstractVector, i::Int)
	newval ≤ stn.min_resnorm && return true
	(abs(newval-val) < stn.min_decrease && i ≥ stn.min_iter) && return true
	all(x -> abs(x) < stn.min_res, y) && return true
	return false
end

function lm_verbose(i::Int, newval::Real, val::Real, λ::Real, y::AbstractVector)
	println("iteration $i:
		residual norm = $(norm(y)),
		maximum residual = $(maximum(y)),
		λ = $λ,
		improvement = $(val-newval)")
end
