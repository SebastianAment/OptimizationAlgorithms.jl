module TestOptimization

using LinearAlgebra
using Test
using Optimization
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
    for t in 1:64
        x .+= Optimization.direction!(G, x, t)
        if norm(G.d) < 1e-2ε
            break
        end
    end
    @test norm(gradient(f, x)) < ε
    # x = randn(n)
    # Optimization.fixedpoint!(x, G)
    # println(x)
end

@testset "BFGS" begin
    # convergence for quadratic
    n = 3
    A = randn(n, n)
    A = A'A
    μ = randn(n)
    f(x) = (x-μ)'A*(x-μ)

    x = randn(n)
    B = Optimization.BFGS(f, x)
    ε = 1e-10
    for t in 1:64
        x .+= Optimization.direction!(B, x, t)
        # println(norm(x-μ))
        if norm(B.d) < 1e-2ε
            break
        end
    end
    @test norm(gradient(f, x)) < ε

    # limited-memory BFGS
    x = randn(n)
    B = Optimization.LBFGS(f, x, 4)
    ε = 1e-10
    for t in 1:64
        x .+= Optimization.direction!(B, x, t)
        if norm(B.d) < 1e-2ε
            break
        end
    end
    @test norm(gradient(f, x)) < ε
    # higher-dimensional non-quadratic
    # f(x) = (x-μ)'A*(x-μ) + sin(sum(x))
    # N = Optimization.Newton(f, x)
    # @time for i in 1:8
    #     x += Optimization.direction!(N, x)
    # end
    # @test all(x -> isapprox(x, 0, atol = 1e-12), gradient(f, x))
end

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

@testset "Newton's method" begin
    # one step convergence for quadratics
    f(x) = x^2
    x = randn()
    N = Optimization.Newton(f, x)
    x += Optimization.direction(N, x)
    @test derivative(f, x) ≈ 0

    # non-quadratic problem
    f(x) = x^2 + sin(x)
    x = 1.
    N = Optimization.Newton(f, x)
    for i in 1:8
        x += Optimization.direction(N, x)
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
    x += Optimization.direction!(N, x)
    @test norm(gradient(f, x)) < ε

    # higher-dimensional non-quadratic
    ε = 1e-10
    f(x) = (x-μ)'A*(x-μ) - sum(x->log(abs2(x)), x)
    x = 2randn(n)
    N = Optimization.Newton(f, x)
    N = Optimization.ArmijoStep(N)
    for t in 1:32
        Optimization.update!(N, x, t)
        N.α = 1
        if norm(gradient(f, x)) < ε
            break
        end
    end
    @test norm(gradient(f, x)) < ε
end

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
end # TestOptimization
