module TestGaussNewton

using Test
using Optimization
using Optimization: GaussNewton, LevenbergMarquart, DecreasingStep, LevenbergMarquartSettings
using Optimization: update!, optimize!

function test_problem(n)
    x = sort!(randn(n))
    g(a, b) = x -> sin(2π*a*x) + b*x^2
    a, b = randn(2)
    y = g(a, b).(x)
    return [a,b], x, y, g, function residual!(r, ab)
        @. r = y - g(ab...).(x)
    end
end

@testset "GaussNewton" begin
    n = 64
    ab0, x, y, g, r = test_problem(n)
    G = GaussNewton(r, ab0, y)
    σ = .1
    ab = ab0 + σ*randn(2)
    for i in 1:16
        update!(G, ab)
    end
    tol = 1e-8
    @test isapprox(ab, ab0, atol = tol)

    D = DecreasingStep(G, ab)
    ab = ab0 + σ*randn(2)
    for i in 1:16
        update!(D, ab)
    end
    tol = 1e-8
    @test isapprox(ab, ab0, atol = tol)
end

@testset "LevenbergMarquart" begin
    n = 64
    ab0, x, y, g, r = test_problem(n)
    LM = LevenbergMarquart(r, ab0, y)
    σ = .1
    ab = ab0 + σ*randn(2)
    tol = 1e-8
    # tesing out intitialization with very large lambda, min_decrease, and min_iter
    res = copy(y)
    stn = LevenbergMarquartSettings(max_iter = 16, min_decrease = 1e-10, min_resnorm = 1e-10, min_res = 0)
    optimize!(LM, ab, res, stn) #, 1e-6, Val(true))
    @test isapprox(ab, ab0, atol = tol)
end

end
