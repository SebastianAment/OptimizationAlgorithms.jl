module TestCompressedSensing
using Test
using LinearAlgebra
using Optimization
using StatsBase: sample, mean

using SparseArrays
function sparse_data(;n = 32, m = 64, k = 3, σ = 1e-2, normalized = true)
    A = randn(n, m)
    if normalized
        A .-= mean(A, dims = 1)
        A ./= sqrt.(sum(abs2, A, dims = 1))
    end
    x = spzeros(m)
    ind = sort!(sample(1:m, k, replace = false))
    @. x[ind] = randn()

    b = A*x
    @. b += σ^2 * randn()
    A, x, b
end

@testset "Sparse Bayesian Learning" begin
    n, m, k = 32, 64, 3
    σ = 1e-2
    Σ = σ^2*I(n)
    A, x, b = sparse_data(n = n, m = m, k = k, σ = σ)

    sbl = Optimization.SBL(A, b, Σ)
    xsbl = zeros(m)
    _, t = Optimization.fixedpoint!(sbl, xsbl)
    @test findall(abs.(xsbl) .> 1e-3) == x.nzind

    # greedy sparse bayesian learning
    using Optimization: greedy_sbl
    xgsbl = greedy_sbl(A, b, σ)

    @test findall(abs.(xgsbl) .> 1e-3) == x.nzind
    @test isapprox(A*xgsbl, b, atol = 5σ)
end

# @testset "Matching Pursuit" begin
#     using Optimization: MP, mp
#     n, m, k = 32, 64, 3
#     σ = 0.
#     A, x, b = sparse_data(n = n, m = m, k = k, σ = σ, normalized = true)
#
#     xmp = mp(A, b, 2k) # give it more iterations, might re-optimized old atom
#     # noiseless
#     @test xmp.nzind == x.nzind
#     # @test xmp.nzval ≈ x.nzval
# end
#
# @testset "Orthogonal Matching Pursuit" begin
#     using Optimization: OMP, omp
#     n, m, k = 32, 64, 4
#     σ = 0.
#     A, x, b = sparse_data(n = n, m = m, k = k, σ = σ)
#     xomp = omp(A, b, k)
#     # noiseless
#     @test xomp.nzind == x.nzind
#     @test xomp.nzval ≈ x.nzval
#
#     σ = 1e-2 # slightly noisy
#     @. b += σ*randn()
#     xomp = omp(A, b, k)
#     # noiseless
#     @test xomp.nzind == x.nzind
#     @test isapprox(xomp.nzval, x.nzval, atol = 5σ)
# end
#
# # TODO: test which ssp can do but omp can't
# @testset "Subspace Pursuit" begin
#     using Optimization: SSP, ssp
#     n, m, k = 32, 64, 3
#     σ = 0.
#     A, x, b = sparse_data(n = n, m = m, k = k, σ = σ)
#
#     σ = 1e-2 # noisy
#     @. b += σ*randn()
#     xomp = ssp(A, b, k)
#     # noiseless
#     @test xomp.nzind == x.nzind
#     @test isapprox(xomp.nzval, x.nzval, atol = 5σ)
# end

end # TestCompressedSensing
