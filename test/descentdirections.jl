module TestDescentDirections

using LinearAlgebra
using Test
using Optimization
using Optimization: fixedpoint!, StoppingCriterion
using ForwardDiff: derivative, gradient

# TODO: separate test problems from algorithms
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
    x, t = fixedpoint!(G, x, StoppingCriterion(x, dx = 1e-2ε))
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
    fixedpoint!(N, x, StoppingCriterion(x, dx = 1e-2ε))
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
    x, t = fixedpoint!(B, x, StoppingCriterion(x, dx = 1e-2ε))
    @test norm(gradient(f, x)) < ε

    # BFGS with ill-conditioned problem
    x = randn(n)
    α = [1e6, 1., 1e-6]
    g(x) = f(α.*x)
    B = Optimization.BFGS(g, x)
    x, t = fixedpoint!(B, x, StoppingCriterion(x, dx = 1e-3ε))
    # test BFGS where inverse Hessian is initialized to Diagonal(1.0./α.^2)
    xs = randn(n)
    BS = Optimization.BFGS(g, xs, Matrix(Diagonal(1.0./α.^2)))
    xs, ts = fixedpoint!(BS, xs, StoppingCriterion(x, dx = 1e-3ε))
    @test norm(gradient(g, x)) < ε
    @test norm(gradient(g, xs)) < ε
    # @test ts < t # scaling outperforms non-scaling for ill-conditioned problem
    # # WARNING: This test could fail, but most times doesn't

    # optimization of 1D non-convex objective (has to forgo BFGS update)
    x = [2.] # fill(2., 2)
    h(x) = 1-exp(-sum(abs2, x))
    B = Optimization.BFGS(h, x, check = false)
    x, t = fixedpoint!(B, x, StoppingCriterion(x, dx = 1e-2ε))
    @test norm(gradient(h, x)) < ε

    x = [2.] # fill(2., 2)
    B = Optimization.LBFGS(h, x, 3, check = false)
    x, t = fixedpoint!(B, x, StoppingCriterion(x, dx = 1e-2ε))
    @test norm(gradient(h, x)) < ε

    ###################### limited-memory BFGS ##################################
    m = 3
    x = randn(n)
    B = Optimization.LBFGS(f, x, m)
    ε = 1e-6
    x, t = fixedpoint!(B, x, StoppingCriterion(x, dx = 1e-2ε))
    # println(t)
    @test norm(gradient(f, x)) < ε

    # higher-dimensional non-quadratic
    f(x) = 1/2*(x-μ)'A*(x-μ) - sum(x->log(abs2(x)), x)
    x = randn(n)
    B = Optimization.BFGS(f, x)
    ε = 1e-6
    x, t = fixedpoint!(B, x, StoppingCriterion(x, dx = 1e-2ε))
    @test norm(gradient(f, x)) < ε

    x = randn(n)
    B = Optimization.LBFGS(f, x, m)
    ε = 1e-6
    x, t = fixedpoint!(B, x, StoppingCriterion(x, dx = 1e-2ε))
    @test norm(gradient(f, x)) < ε

    # ill-conditioned problem, wouldn't work without scaling function
    scaling!(d, s, y) = (d .*= 1.0./α.^2)
    B = Optimization.LBFGS(g, x, 3, scaling!, check = true)
    x, t = fixedpoint!(B, x, StoppingCriterion(x, dx = 1e-2ε))
    @test norm(gradient(g, x)) < ε
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
    fixedpoint!(B, x, StoppingCriterion(x, dx = 1e-2ε))
    @test norm(gradient(f, x)) < ε
end

end # TestDescentDirections
