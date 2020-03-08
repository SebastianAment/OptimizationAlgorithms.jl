# Compressed Sensing algorithms
# TODO: move over OMP, etc.

########################## Sparse Bayesian Learning ############################
# sparse bayesian learning for regression
# TODO: investigate direct optimization of nlml, not via em
using LinearAlgebraExtensions: AbstractMatOrUni, AbstractMatOrFac
using MyLazyArrays: inverse
struct SBL{T, AT, BT, GT} <: Update{T}
    AΣA::AT
    AΣb::BT
    Γ::GT
    # B # pre-allocate
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
