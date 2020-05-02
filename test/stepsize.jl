module TestStepSize
using LinearAlgebra
using Test
using Optimization
using Optimization: fixedpoint!, StoppingCriterion
using ForwardDiff: gradient

n = 2
A = randn(n, n)
A = A'A + I
μ = randn(n)
f(x) = (x-μ)'A*(x-μ)

@testset "descent" begin
    x = randn(n)
    G = Optimization.Gradient(f, x)
    U = Optimization.DecreasingStep(G, x)
    ε = 1e-6
    fixedpoint!(U, x, StoppingCriterion(x, dx = 1e-2ε))
    @test norm(gradient(f, x)) < ε
end

@testset "armijo" begin
    x = randn(n)
    G = Optimization.Gradient(f, x)
    U = Optimization.ArmijoStep(G, x)
    ε = 1e-6
    fixedpoint!(U, x, StoppingCriterion(x, dx = 1e-2ε))
    @test norm(gradient(f, x)) < ε
end

end # TestStepSize
