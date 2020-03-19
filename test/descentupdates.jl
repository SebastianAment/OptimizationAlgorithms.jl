module TestDescentUpdates
using Test
using LinearAlgebra
using Optimization
using Optimization: Gradient, ScaledGradient
using Optimization: PolyakMomentum, Nesterov, Adam
using ForwardDiff: gradient

@testset "PolyakMomentum" begin
    n = 2
    A = randn(n, n)
    A = A'A + I
    μ = randn(n)
    f(x) = (x-μ)'A*(x-μ)
    x = randn(n)
    G = ScaledGradient(f, x)
    G = PolyakMomentum(G, copy(x), .5)
    ε = 1e-6
    for t in 1:64
        G(x)
        if norm(gradient(f, x)) < ε
            break
        end
    end
    @test norm(gradient(f, x)) < ε
end

@testset "Nesterov" begin
    # TODO: better test for nesterov
    n = 2
    A = randn(n, n)
    A = A'A + I
    μ = randn(n)
    f(x) = (x-μ)'A*(x-μ)
    x = randn(n)
    G = ScaledGradient(f, x)
    G = Nesterov(G, copy(x), 1., .01)
    ε = 1e-6
    for t in 1:64
        G(x)
        if norm(gradient(f, x)) < ε
            break
        end
    end
    @test norm(gradient(f, x)) < ε
end

end # TestDescentUpdates
