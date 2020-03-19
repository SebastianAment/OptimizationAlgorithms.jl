module TestCompressedSensing
using Test
using LinearAlgebra
using Optimization

@testset "SBL" begin
    n = 32
    m = 64
    A = randn(n, m)

    using StatsBase: sample
    k = 3
    x0 = zeros(m)
    ind = sort!(sample(1:m, k, replace = false))
    x0[ind] = randn(k)

    Σ = 1e-6I(n)
    b = A*x0 + sqrt(Σ) * randn(n)
    sbl = Optimization.SBL(A, b, Σ)
    x = zeros(m)
    # println(x0[ind])
    _, t = Optimization.fixedpoint!(sbl, x)
    # println(t)
    # println(x[ind])
    @test findall(abs.(x) .> 1e-3) == ind
end

end # TestCompressedSensing
