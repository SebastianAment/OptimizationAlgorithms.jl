abstract type AbstractGaussNewton{T} <: Direction{T} end

function objective!(GN::AbstractGaussNewton, y::AbstractVector, x::AbstractVector)
	sum(abs2, GN.f(y, x))
end
# WARNING overwrites value vector
function objective!(GN::AbstractGaussNewton, x::AbstractVector)
	y = DiffResults.value(GN.r)
    sum(abs2, GN.f(y, x))
end
function objective(GN::AbstractGaussNewton, x::AbstractVector)
	y = similar(DiffResults.value(GN.r))
	sum(abs2, GN.f(y, x))
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

# TODO: put settings in struct?
# min_decrease is a minimum decrease in the residual norm below which the
# algorithm terminates if also min_iter iterations have been executed
function optimize!(LM::LevenbergMarquart{<:Real},
			x::AbstractVector, y::AbstractVector,
			verbose::Union{Val{true}, Val{false}} = Val(false);
			maxiter::Int = 128, maxbacktrack::Int = 32,
            min_val::Real = 1e-6, min_decrease::Real = 1e-12, min_iter::Int = 1,
            max_step::Real = Inf, min_diagonal::Real = 1e-12,
			lambda::Real = 1e-3, min_lambda::Real = 1e-12, max_lambda::Real = 1e12,
            increase_factor::Real = 3, decrease_factor::Real = 2)
    oldx = copy(x)
	λ = lambda
	val, newval = Inf, Inf
    for i in 1:maxiter
		update_jacobian!(LM, oldx)
		for j in 1:maxbacktrack
			val, dx = update_direction!(LM, λ, min_diagonal)
			@. x = oldx + dx
			newval = sum(abs2, LM.f(y, x))
			if newval > val || isnan(newval) || any(x->abs(x) > max_step, dx)
				λ = min(λ * increase_factor, max_lambda)
				if !(j < maxbacktrack)
					return x .= oldx # copy!(x, oldx) # if backtracking wasn't successful
				end
			else
				λ = max(λ / decrease_factor, min_lambda)
				break
			end
		end
		if newval < min_val || (abs(newval-val) < min_decrease && i ≥ min_iter) # if convergence criteria are met,
			if λ > min_lambda # make sure that we at least take one GaussNewton step at the optimum for maximum accuracy
				λ = min_lambda
			else
				break
			end
		end
		verbose isa Val{false} || lm_verbose(i, newval, val, λ)
		val = newval
        copy!(oldx, x)
	end
	return x
end

function lm_verbose(i, newval, val, λ)
	println("iteration $i:
		residual norm = $newval,
		λ = $λ,
		improvement = $(val-newval)")
end
