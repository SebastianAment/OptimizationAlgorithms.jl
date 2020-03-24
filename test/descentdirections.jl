module TestDescentDirections

using LinearAlgebra
using Test
using Optimization
using Optimization: fixedpoint!, StoppingCriterion
using ForwardDiff: derivative, gradient

# TODO: tests for GaussNewton, Momentum ADAM, NaturalGradient, Nesterov
@testset "gradient descent" begin
    n = 2
    A = randn(n, n)
    A = A'A + I
    μ = randn(n)
    f(x) = (x-μ)'A*(x-μ)
    x = randn(n)
    G = Optimization.ScaledGradient(f, x)
    ε = 1e-6
    x, t = fixedpoint!(G, x, StoppingCriterion(x, 1e-2ε))
    @test norm(gradient(f, x)) < ε
end

@testset "Newton's method" begin
    # one step convergence for quadratics
    f(x) = x^2
    x = randn()
    N = Optimization.Newton(f, x)
    x += N(x)
    @test derivative(f, x) ≈ 0

    # non-quadratic problem
    f(x) = x^2 + sin(x)
    x = 1.
    N = Optimization.Newton(f, x)
    for i in 1:8
        x += N(x)
    end
    @test derivative(f, x) ≈ 0

    # one-step convergence for higher-dimensional quadratic
    n = 3
    A = randn(n, n) + I
    A = A'A
    μ = randn(n)
    f(x) = (x-μ)'A*(x-μ)
    ε = 1e-12

    x = randn(n)
    N = Optimization.Newton(f, x)
    x += N(x)
    @test norm(gradient(f, x)) < ε

    # higher-dimensional non-quadratic
    f(x) = 1/2*(x-μ)'A*(x-μ) - sum(x->log(abs2(x)), x)
    x = 2randn(n)
    N = Optimization.Newton(f, x)
    ε = 1e-6
    fixedpoint!(N, x, StoppingCriterion(x, 1e-2ε))
    @test norm(gradient(f, x)) < ε
end


@testset "(L)BFGS" begin
    # convergence for quadratic
    n = 3
    A = randn(n, n)
    A = A'A
    μ = randn(n)
    f(x) = (x-μ)'A*(x-μ)

    x = randn(n)
    B = Optimization.BFGS(f, x)
    ε = 1e-6
    x, t = fixedpoint!(B, x, StoppingCriterion(x, 1e-2ε))
    # println(t)
    @test norm(gradient(f, x)) < ε

    # limited-memory BFGS
    m = 3
    x = randn(n)
    B = Optimization.LBFGS(f, x, m)
    ε = 1e-6
    x, t = fixedpoint!(B, x, StoppingCriterion(x, 1e-2ε))
    # println(t)
    @test norm(gradient(f, x)) < ε

    # higher-dimensional non-quadratic
    f(x) = 1/2*(x-μ)'A*(x-μ) - sum(x->log(abs2(x)), x)
    x = randn(n)
    B = Optimization.BFGS(f, x)
    ε = 1e-6
    x, t = fixedpoint!(B, x, StoppingCriterion(x, 1e-2ε))
    # println(t)
    @test norm(gradient(f, x)) < ε

    x = randn(n)
    B = Optimization.LBFGS(f, x, m)
    ε = 1e-6
    x, t = fixedpoint!(B, x, StoppingCriterion(x, 1e-2ε))
    # println(t)
    @test norm(gradient(f, x)) < ε
end

@testset "CustomDirection" begin
    # convergence for quadratic
    n = 3
    A = randn(n, n)
    A = 1/2*A'A + I
    μ = randn(n)
    f(x) = (x-μ)'A*(x-μ)
    function valdir(x::AbstractVector) # shows how evaluation and gradient computation can be pooled
        g = A*(x-μ)
        v = (x-μ)'g
        g .*= -g'g / (g'A*g) # optimal stepsize for quadratic problems
        v, g
    end
    x = 100randn(n) + μ # normally distributed around μ
    B = Optimization.CustomDirection(f, valdir, x)
    ε = 1e-6
    fixedpoint!(B, x, StoppingCriterion(x, 1e-2ε))
    @test norm(gradient(f, x)) < ε
end

end # TestDescentDirections