module TestStepSize
using LinearAlgebra
using Test
using OptimizationAlgorithms
using OptimizationAlgorithms: fixedpoint!, StoppingCriterion
using ForwardDiff: gradient

n = 2
A = randn(n, n)
A = A'A + I
μ = randn(n)
f(x) = (x-μ)'A*(x-μ)

@testset "descent" begin
    x = randn(n)
    G = OptimizationAlgorithms.Gradient(f, x)
    U = OptimizationAlgorithms.DecreasingStep(G, x)
    ε = 1e-6
    fixedpoint!(U, x, StoppingCriterion(x, dx = 1e-2ε, maxiter = 256))
    @test norm(gradient(f, x)) < ε
end

@testset "armijo" begin
    x = randn(n)
    G = OptimizationAlgorithms.Gradient(f, x)
    U = OptimizationAlgorithms.ArmijoStep(G, x)
    ε = 1e-6
    fixedpoint!(U, x, StoppingCriterion(x, dx = 1e-2ε, maxiter = 256))
    @test norm(gradient(f, x)) < ε
end

end # TestStepSize
