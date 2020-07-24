module TestLinearSolvers

using Test
using LinearAlgebra
using Optimization: cg, CG, CGMatrix

@testset "conjugate gradient" begin
    n = 16
    m = n
    A = randn(m, n) / sqrt(n)
    A = A'A + I
    b = randn(n)

    # early stopping
    ε = 1e-3
    x = cg(A, b, min_res = ε)
    @test 1e-10 < norm(A*x-b) ≤ ε

    # high accuracy
    ε = 1e-10
    x = cg(A, b)
    @test norm(A*x-b) ≤ ε

    ################### CGMatrix
    CA = CGMatrix(A)
    @test CA isa CGMatrix
    @test CA\b ≈ x

    x = randn(n)
    @test CA*x ≈ A*x

    #################### mul!
    y = copy(b)
    @test mul!(y, CA, x, -1, 1) ≈ b - A*x

    y = copy(b)
    α, β = randn(2)
    @test mul!(y, CA, x, α, β) ≈ β*b + α*A*x

    #################### ldiv!
    b = A*x
    @test ldiv!(y, CA, b) ≈ x

    # getindex
    @test CA[:, 1] ≈ A[:, 1]
    @test CA[3, 2] ≈ A[3, 2]

    # setindex!
    v = randn()
    CA[1] = v
    @test CA[1] ≈ v
    @test A[1] ≈ v # mutates original array
end

end # TestLinearSolvers
