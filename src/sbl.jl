########################## Sparse Bayesian Learning ############################
# sparse bayesian learning for linear regression
# TODO: investigate direct optimization of nlml, not via em
using LinearAlgebraExtensions: AbstractMatOrUni, AbstractMatOrFac
using LazyInverse: inverse
using WoodburyIdentity

struct SparseBayesianLearning{T, AT, BT, GT} <: Update{T}
    AΣA::AT
    AΣb::BT
    Γ::GT
    # B # pre-allocate
    # Σ::ST
    function SparseBayesianLearning(A::AbstractMatrix, b::AbstractVector,
                        Σ::AbstractMatOrUni, Γ::Diagonal = (1.0I)(size(A, 2)))
        AΣ = A'inverse(Σ)
        AΣb = AΣ * b
        AΣA = AΣ * A
        T = eltype(A)
        new{T, typeof(AΣA), typeof(AΣb), typeof(Γ)}(AΣA, AΣb, Γ)
    end
end
const SBL = SparseBayesianLearning
# TODO: prune indices?
# TODO: for better conditioning
# C = (sqrt(Γ)*AΣA*sqrt(Γ) + I)
# B = inverse(sqrt(Γ)) * C * inverse(sqrt(Γ))
# CG could work well, because we have good initial x's between iterations
# TODO: for noiseless L0 norm minimization: B = sqrt(Γ)*pseudoinverse(A * sqrt(Γ))
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

##################### greedy marginal likelihood optimization ##################
using SparseArrays
using Base.Threads
struct GreedySparseBayesianLearning{T, AT<:AbstractMatrix{T}, BT, NT, A} <: Update{T}
    A::AT
    b::BT
    Σ::NT
    α::A

    # temporaries
    S::A
    Q::A
    δ::A # holds difference in marginal likelihood for updating an atom
end
const GSBL = GreedySparseBayesianLearning
function GSBL(A::AbstractMatrix, b::AbstractVector, Σ)
    α = fill(Inf, size(A, 2))
    S, Q, δ = (similar(α) for i in 1:3)
    GSBL(A, b, Σ, α, S, Q, δ)
end

function GSBL(A::AbstractMatrix, b::AbstractVector, σ²::Real)
    GSBL(A, b, σ²*I(length(b)))
end

function Base.getproperty(S::GSBL, s::Symbol)
    if s == :x
        isactive = @. !isinf(S.α) # active basis patterns
        x = spzeros(eltype(S.A), size(S.A, 2))
        Ai = @view S.A[:, isactive]
        Σ⁻¹ = inverse(S.Σ)
        P = inverse(*(Ai', Σ⁻¹, Ai) + Diagonal(S.α[isactive]))
        x .= 0
        x[isactive] .= P * (Ai' * (Σ⁻¹ * S.b))
        dropzeros!(x)
    else
        getfield(S, s)
    end
end


function initialize!(sbl::GSBL, α::AbstractVector)
    # nA = sum(abs2, A, dims = 1)[:]
    inner = sbl.A'sbl.b
    k = argmax(abs.(inner))# ./ nA)
    α[k] = inner[k] #nA[k] / (inner[k])
    α
end

# should the input be α?
function update!(sbl::GSBL, α::AbstractVector)
    isactive = @. !isinf(α) # active basis patterns
    A, b, Σ, S, Q = sbl.A, sbl.b, sbl.Σ, sbl.S, sbl.Q
    if !any(isactive)
        return initialize!(sbl, α)
    end
    # get kernel matrix C
    Ai = @view A[:, isactive]
    Γ = Diagonal(inv.(α[isactive]))
    C = Woodbury(Σ, Ai, Γ, Ai')
    C = factorize(C)
    C⁻¹ = inverse(C)
    # update sparsity and quality factors
    @threads for k in 1:size(A, 2)
        Ak = @view A[:, k]
        S[k] = sparsity(C⁻¹, Ak)
        Q[k] = quality(C⁻¹, b, Ak)
    end
    # potential change in marginal likelihood for each atom
    @. sbl.δ = delta(α, S, Q)
    return update_α!(α, sbl.δ, S, Q)
end

function sparsity(C⁻¹::AbstractMatOrFac, a::AbstractVector)
    dot(a, C⁻¹, a)
end
function quality(C⁻¹::AbstractMatOrFac, t::AbstractVector, a::AbstractVector)
    dot(a, C⁻¹, t)
end

# updates set of active basis functions
# greedy strategy: chooses update which increases marginal likelihood most (max δ)
function update_α!(α, δ, S, Q)
    k = argmax(δ)
    sk, qk = sq(S[k], Q[k], α[k])
    α[k] = optimal_α(sk, qk)
    return α
end

# potential change in marginal likelihood for each atom
function delta(α::Real, S::Real, Q::Real)
    s, q = sq(S, Q, α)
    isactive = α < Inf
    isrelevant = s < q^2
    if !isactive && isrelevant # out of model and high quality factor
        δ = δ_add(S, Q)
    elseif isactive && !isrelevant # in model but not high enough quality
        δ = δ_delete(S, Q, α)
    elseif isactive && isrelevant # α re-estimation
        αn = optimal_α(s, q)
        δ = δ_update(S, Q, α, αn)
    else # !isactive && !isrelevant
        δ = 0.
    end
    return δ
end

function δ_add(S, Q)
    (Q^2 - S) / S + log(S) - log(Q^2) # add k
end
function δ_delete(S, Q, α)
    Q^2 / (S - α) - log(1 - (S/α)) # delete k
end
function δ_update(S, Q, α, αn)
    d = 1/αn - 1/α
    Q^2 / (S + 1/d) - log(1 + S*d)
end

# calculate small s, q from S, Q (see Tipping 2003)
function sq(S::Real, Q::Real, α::Real)
    if α < Inf
        s = α*S / (α-S)
        q = α*Q / (α-S)
        s, q
    else
        S, Q
    end
end

function optimal_α(s::Real, q::Real)
    s < q^2 ? s^2 / (q^2 - s) : Inf
end

function greedy_sbl(A, b, σ; maxiter = 128)
    S = GSBL(A, b, σ)
    α = S.α
    for i in 1:maxiter
        S(α)
    end
    S.x
end
