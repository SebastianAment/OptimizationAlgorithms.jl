# Compressed Sensing algorithms

include("pursuit.jl")

########################## Sparse Bayesian Learning ############################
# sparse bayesian learning for regression
# TODO: investigate direct optimization of nlml, not via em
using LinearAlgebraExtensions: AbstractMatOrUni, AbstractMatOrFac
using LazyInverse: inverse
struct SBL{T, AT, BT, GT} <: Update{T}
    AΣA::AT
    AΣb::BT
    Γ::GT
    # B # pre-allocate
    # Σ::ST
    function SBL(A::AbstractMatrix, b::AbstractVector, Σ::AbstractMatOrUni,
                Γ::Diagonal = (1.0I)(size(A, 2)))
        AΣ = A'inverse(Σ)
        AΣb = AΣ * b
        AΣA = AΣ * A
        T = eltype(A)
        new{T, typeof(AΣA), typeof(AΣb), typeof(Γ)}(AΣA, AΣb, Γ)
    end
end
# TODO: prune indices?
# TODO: for better conditioning
# C = (sqrt(Γ)*AΣA*sqrt(Γ) + I)
# B = inverse(sqrt(Γ)) * C * inverse(sqrt(Γ))
# TODO: try CG? could work well, because we have good initial x's
# TODO: implement solve! for different matrix types
# TODO: for L0 norm: B = sqrt(Γ)*pseudoinverse(A * sqrt(Γ))
# try non-linear optimization directly on marginal likelihood
function update!(S::SBL, x::AbstractVector = zeros(size(S.Γ, 2)))
    AΣA, AΣb, Γ = S.AΣA, S.AΣb, S.Γ
    B = AΣA + inverse(Γ) # woodury makes sense if number of basis functions is larger than x
    B = factorize(B) # could use in-place cholesky!(B)
    B isa Number ? (x .= AΣb ./ B) : ldiv!(x, B, AΣb) # x = B \ AΣb, could generalize with functional solve! input
    # update rules
    # @. Γ.diag = x^2 + $diag($inverse(B)) + 1e-16 # provably convergent
    @. Γ.diag = x^2 / (1 - $diag($inverse(B)) / Γ.diag) + 1e-14 # good heuristic
    return x
end
# updating noise variance (if scalar)
# B⁻¹ = inverse(B)
# σ² = sum(abs2, b-A*x) / sum(i->B[i]/Γ[i], diagind(Γ)) # could replace with euclidean norm

# TODO: greedy marginal likelihood optimization
# using SparseArrays
# function greedy_sbl(A, x::AbstractVector, y, σ)
#     n, m = size(A)
#     # initialization
#     i = 1
#     x[i] = 1 # or something better
#     α = fill(Inf, m)
#     a2 = sum(abs2, A[:,i])
#     α[i] = a2 / (dot(A[:,i], y)^2 / a2 - σ^2)
#     isactive = .!isinf.(α) # active basis patterns
#
#     Σ = σ^2*Ι(length(y))
#     B = inv(Σ)
#     ind = i
#     Φ = @view A[:, isactive]
#     # AΣ = A'inverse(Σ)
#     for i in 1:maxiter
#         P = inverse(Φ'inverse(Σ)*Φ + Diagonal(α[isactive])) # posterior covariance
#         P = factorize(P)
#         update_sq!(s, q, A, B, Φ, P, y)
#         update_α!(α, s, q)
#         @. isactive = !isinf(α) # active basis patterns
#     end
#     x = P * (Φ' * (B * y))
#     return x
# end
#
# # both updates could work on a single basis k
# # updates set of active basis functions
# # greedy strategy: chooses update which increases marginal likelihood most
# function update_α!(α, s, q)
#     δ = zero(α)
#     for k in eachindex(α)
#         if α[k] == Inf && s[k] < q[k]^2 # out of model and high quality factor
#             δ[k] = (q[k]^2 - s[k]) / s[k] + log(s[k]) - 2log(q[k]) # add k
#         elseif α[k] < Inf && q[k]^2 ≤ s[k] # in model but not high enough quality
#             δ[k] = q[k]^2 / (s[k] - α[k]) - log(1 - (s[k] / a[k])) # delete k
#         else # α re-estimation
#             β = s[k]^2 / (q[k]^2 - s[k])
#             d =  1/β - 1/α[k]
#             δ[k] = q[k]^2 / (s[k] + 1/d) - log(1 + s[k]*d)
#         end
#     end
#     k = argmax(δ)
#     α[k] = optimal_α(s[k], q[k])
#     return α
# end
# optimal_α(s::Real, q::Real) = (s < q^2) ? (s^2 / (q^2 - s)) : Inf
#
# # updates the sparsity and quality factors s and q
# # could be parallelized
# # TODO: should have separate update if B has not changed since last update
# # (see Tipping 2003, Appendix)
# function update_sq!(s, q, A, B, Φ, P, y, α)
#     # W = Woodbury(B, BΦ', P, BΦ, -1.) # potentially useful
#     By = B*y
#     m = size(A, 2)
#     for k in 1:m
#         φ = view(A, :, k) # kth basis function
#         Bφ = B*φ # TODO: mul!, and could collapse into single temporary vector
#         ΦBφ = Φ*Bφ
#         s[k] = dot(φ, Bφ) - dot(ΦBφ, P, ΦBφ)
#         q[k] = dot(φ, By) - dot(ΦBφ, P, By)
#         if α[k] < Inf
#             s[k] = α[k]*s[k] / (α[k]-s[k]) # equation (23) in Tipping 2003
#             q[k] = α[k]*q[k] / (α[k]-s[k])
#         end
#     end
#     return s, q
# end
