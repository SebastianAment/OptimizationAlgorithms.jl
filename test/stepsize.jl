module TestStepSize
using LinearAlgebra
using Test
using OptimizationAlgorithms
using OptimizationAlgorithms: fixedpoint!, StoppingCriterion
using ForwardDiff: gradient, derivative

n = 2
A = randn(n, n)
A = A'A + I
μ = randn(n)
f(x) = (x-μ)'A*(x-μ)

ε = 1e-6
maxiter = 1024
@testset "descent" begin
    x = randn(n)
    G = OptimizationAlgorithms.Gradient(f, x)
    U = OptimizationAlgorithms.DecreasingStep(G, x)
    fixedpoint!(U, x, StoppingCriterion(x, dx = 1e-2ε, maxiter = maxiter))
    @test norm(gradient(f, x)) < ε

    # also test scalar case
    x = 2rand()-1
    g(x) = -exp(-(x-1)^2)
    G = OptimizationAlgorithms.Gradient(g, x)
    U = OptimizationAlgorithms.DecreasingStep(G, x)
    xn, t = fixedpoint!(U, x, StoppingCriterion(x, dx = 1e-2ε, maxiter = maxiter))
    @test abs(derivative(g, xn)) < ε
end

@testset "armijo" begin
    x = randn(n)
    G = OptimizationAlgorithms.Gradient(f, x)
    U = OptimizationAlgorithms.ArmijoStep(G, x)
    fixedpoint!(U, x, StoppingCriterion(x, dx = 1e-2ε, maxiter = maxiter))
    @test norm(gradient(f, x)) < ε
end


end # TestStepSize
