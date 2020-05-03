module TestCompressedSensing
using Test
using LinearAlgebra
using Optimization
using StatsBase: sample
using SparseArrays
function sparse_data(;n = 32, m = 64, k = 3, σ = 1e-2)
    A = randn(n, m)
    x = spzeros(m)
    ind = sort!(sample(1:m, k, replace = false))
    @. x[ind] = randn()

    b = A*x
    @. b += σ^2 * randn()
    A, x, b
end

@testset "OMP" begin
    using Optimization: OMP, omp
    n, m, k = 32, 128, 8
    σ = 0.
    A, x, b = sparse_data(n = n, m = m, k = k, σ = σ)
    xomp = omp(A, b, k)
    # noiseless
    @test xomp.nzind == x.nzind
    @test xomp.nzval ≈ x.nzval

    σ = 1e-2 # slightly noisy
    @. b += σ*randn()
    xomp = omp(A, b, k)
    # noiseless
    @test xomp.nzind == x.nzind
    @test isapprox(xomp.nzval, x.nzval, atol = σ)
end

# TODO: test which ssp can do but omp can't
@testset "SSP" begin
    using Optimization: SSP, ssp
    n, m, k = 64, 128, 3
    σ = 0.
    A, x, b = sparse_data(n = n, m = m, k = k, σ = σ)

    σ = 1e-1 # noisy
    @. b += σ*randn()
    xomp = ssp(A, b, k)
    # noiseless
    @test xomp.nzind == x.nzind
    @test isapprox(xomp.nzval, x.nzval, atol = σ)
end

@testset "SBL" begin
    n, m, k = 32, 64, 3
    σ = 1e-2
    Σ = σ^2*I(n)
    A, x, b = sparse_data(n = n, m = m, k = k, σ = σ)

    sbl = Optimization.SBL(A, b, Σ)
    xsbl = zeros(m)
    _, t = Optimization.fixedpoint!(sbl, xsbl)
    @test findall(abs.(xsbl) .> 1e-3) == x.nzind
end

end # TestCompressedSensing
