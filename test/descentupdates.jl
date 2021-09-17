module TestDescentUpdates
using Test
using LinearAlgebra
using OptimizationAlgorithms
using OptimizationAlgorithms: Gradient, ScaledGradient, StoppingCriterion, fixedpoint!
using OptimizationAlgorithms: PolyakMomentum, Nesterov, ADAM, AMSGRAD
using ForwardDiff: gradient

ε = 1e-4
@testset "PolyakMomentum" begin
    n = 2
    A = randn(n, n)
    A = A'A + I
    μ = randn(n)
    f = x -> (x-μ)'A*(x-μ)
    x = randn(n)
    G = Gradient(f, x)
    α = .1
    β = .5
    G = PolyakMomentum(G, copy(x), α, β)
    fixedpoint!(G, x, StoppingCriterion(x, dx = 1e-2ε))
    @test norm(gradient(f, x)) < ε
end

@testset "Nesterov" begin
    # TODO: better test for nesterov
    n = 2
    A = randn(n, n)
    A = A'A + I
    μ = randn(n)
    f = x -> (x-μ)'A*(x-μ)
    x = randn(n)
    α = 5e-2
    β = .5
    G = Gradient(f, x)
    G = Nesterov(G, x, α, β)
    fixedpoint!(G, x, StoppingCriterion(x, dx = 1e-2ε))
    @test norm(gradient(f, x)) < ε
end

@testset "ADAM" begin
    n = 2
    A = randn(n, n)
    A = A'A + I
    μ = randn(n)
    f = x -> (x-μ)'A*(x-μ)
    x = randn(n)
    G = Gradient(f, x)
    α = .1
    β₁ = 0.5
    β₂ = 0.999 # this should be large
    G = ADAM(G, x, α, β₁, β₂)
    fixedpoint!(G, x, StoppingCriterion(x, dx = 1e-2ε, maxiter = 1024))
    @test norm(gradient(f, x)) < ε
end

@testset "AMSGRAD" begin
    n = 2
    A = randn(n, n)
    A = A'A + I
    μ = randn(n)
    f = x -> (x-μ)'A*(x-μ)
    x = randn(n)
    G = Gradient(f, x)
    α = .1
    β₁ = 0.5
    β₂ = 0.999 # this should be large
    G = AMSGRAD(G, x, α, β₁, β₂)
    fixedpoint!(G, x, StoppingCriterion(x, dx = 1e-2ε, maxiter = 1024))
    @test norm(gradient(f, x)) < ε
end

end # TestDescentUpdates
