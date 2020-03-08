module TestOperators
using Test
using Operators
using LinearAlgebra

@testset "basic operators" begin
    # evaluation functional
    f = x->sin(sum(x))
    x = randn()
    δ = Operators.Evaluation(x)
    @test δ(f) ≈ f(x)

    # 2d evaluation functional test
    x = randn(2)
    δ = Operators.Evaluation(x)
    @test δ(f) ≈ f(x)
    @test Operators.islinear(δ)

    # sum
    δ = Operators.Evaluation.(x)
    δ = Operators.Sum(δ...)
    @test δ(f) ≈ sum(f.(x))

    # scaling
    δ = 2δ
    @test Operators.isfunctional(δ)
    @test Operators.islinear(δ)
    @test δ(f) ≈ 2sum(f.(x))

    # dot

    # shift
    x = randn()
    y = randn()
    S = Operators.Shift(x)
    δ = Operators.Evaluation(y)
    @test δ(S(f)) ≈ f(y-x)

# composition
# println(Operators.Composition(δ, S)(f))

end

@testset "scalar differential operators" begin
    f(x) = sin(sum(x))
    # del
    ∇ = Operators.del
    # derivative
    x = randn()
    δ = Operators.δ(x)
    @test δ(∇(f)) ≈ cos(x)
    # gradient
    x = randn(2)
    δ = Operators.Evaluation(x)
    @test  δ(∇(f)) ≈ cos(sum(x)) * ones(2)

    # laplacian
    # Δ = Operators.Laplacian()
    Δ = Operators.Δ
    x = randn()
    δ = Operators.Evaluation(x)
    @test δ(Δ(f)) ≈ -sin(x)

    x = randn(2)
    δ = Operators.Evaluation(x)
    @test  δ(Δ(f)) ≈ sum(-sin(sum(x)) * ones(2))
    # ODE operator
end

@testset "vector differential operators" begin

    g(x) = [sin(sum(x)), cos(sum(x))]
    Operators.outputvalue(::typeof(g)) = Operators.VectorValued()

    # vector del, scalar input
    ∇ = Operators.∇
    @test ∇(g)(0.) ≈ [1, 0]

    # vector del, vector input
    x = randn(2)
    @test ∇(g)(x) ≈ [[cos(sum(x)) cos(sum(x))]; [-sin(sum(x)) -sin(sum(x))]]

    # vector laplacian, scalar input
    Δ = Operators.Δ
    println(Δ(g)(1.))

    # vector laplacian, vector input
    # println(Δ(g)(randn(2)))

    # curl
    f(x, y, z) = [y, -x, zero(z)]
    f(x::AbstractVector) = f(x...)
    x = randn(3)
    @test (∇ × f)(x) ≈ [0, 0, -2]

    f(x, y, z) = [zero(x), -x^2, zero(z)]
    f(x::AbstractVector) = f(x...)
    x = randn(3)
    @test (∇ × f)(x) ≈ [0, 0, -2x[1]]
end

end # TestOperators
