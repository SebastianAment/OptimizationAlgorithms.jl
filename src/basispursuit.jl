using Clp
import JuMP
using JuMP: Model, optimizer_with_attributes, @variable, @objective, @constraint

function basispursuit(A::AbstractMatrix, b::AbstractVector)
    basispursuit(A, b, ones(eltype(A), size(A, 2)))
end

function basispursuit(A::AbstractMatrix, b::AbstractVector, w::AbstractVector)
    model = JuMP.Model(optimizer_with_attributes(Clp.Optimizer, "LogLevel" => 0))
    n, m = size(A)
    @variable(model, x⁺[1:m] ≥ 0)
    @variable(model, x⁻[1:m] ≥ 0)
    @objective(model, Min, dot(w, x⁺) + dot(w, x⁻))
    @constraint(model, constraint, A * (x⁺ .- x⁻) .== b)
    JuMP.optimize!(model)
    x = @. JuMP.value(x⁺) - JuMP.value.(x⁻)
    sparse(x)
end

function basispursuit_reweighting(A::AbstractMatrix, b::AbstractVector,
                                    reweighting!; maxiter = 4)
    x = basispursuit(A, b)
    w = zeros(length(x))
    for i in 2:maxiter
        reweighting!(w, x)
        x = basispursuit(A, b, w)
    end
    x
end

candes_weight(x, ε) = inv(abs(x) + ε)
function candes_weights!(w, x, ε::Real)
    @. w = candes_weight(x, ε)
end
candes_function(ε::Real) = (w, x) -> candes_weights!(w, x, ε)
function candes_reweighting(A::AbstractMatrix, b::AbstractVector,
                                            ε = 1e-6; maxiter = 4)
    basispursuit_reweighting(A, b, candes_function(ε), maxiter = maxiter)
end


function ard_weights!(w, A, x, ε)
    nzind = x.nzind
    Ai = @view A[:, nzind]

    wx = @. abs(x.nzind) / w[nzind]
    WX = Diagonal(wx)
    K = Woodbury(ε*I(size(A, 1)), Ai, WX, Ai')
    K⁻¹ = inverse(factorize(K))

    dK(a) = sqrt(max(dot(a, K⁻¹, a), 0))
    @. w = dK($eachcol(A))
end
ard_function(A::AbstractMatrix, ε::Real) = (w, x) -> ard_weights!(w, A, x, ε)

function ard_reweighting(A::AbstractMatrix, b::AbstractVector,
                                            ε = 1e-6; maxiter = 4)
    basispursuit_reweighting(A, b, ard_function(A, ε), maxiter = maxiter)
end

# W = Diagonal(inv.(w))
# X = diag(abs.(x))
# K = Symmetric(α * I + A*W*X*A') # could get rid of 0. X and use Woodbury
# K⁻¹ = inverse(cholesky(K))
