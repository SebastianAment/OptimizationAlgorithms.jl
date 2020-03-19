module TestLinearSolvers

using Test
using LinearAlgebra
using Optimization

@testset "conjugate gradient" begin
    n = 16
    m = n
    A = randn(m, n) / sqrt(n)
    A = A'A + I
    b = randn(n)
    x = zero(b)
    CG = Optimization.ConjugateGradient(A, b, x)
    ε = 1e-10
    for t in 1:n
        Optimization.update!(CG, x, t)
    end
    @test norm(A*x-b) ≤ ε
end

end # TestLinearSolvers
