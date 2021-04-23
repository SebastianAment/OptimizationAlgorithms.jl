module TestUtil
using Test
using OptimizationAlgorithms: CircularBuffer, _circshift!

@testset "CircularBuffer" begin
    cb = CircularBuffer{Float64}(2)
    push!(cb, 1.)
    push!(cb, 2.)
    @test _circshift!(cb) ≈ [2., 1.]
end

end # TestUtil
