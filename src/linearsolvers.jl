# TODO: more iterative methods: GMRES, Arnoldi, Lanczos
################################################################################
# TODO: preconditioning, could be external by running cg on
# EAE'x = Eb where inv(E)inv(E)' = M where M is preconditioner
# TODO: nonlinear
# β = polyakribiere(dx₁, dx₂) = dx₁'(dx₁-dx₂) / (dx₂'dx₂)
# Careful with ill-conditioned systems
struct ConjugateGradient{T, M, V, R} <: Update{T}
    # f::F # nonlinear conjugate gradient
    A::M # linear conjugate gradient
    b::V # target
    d::V # direction
    Ad::V # product of A and direction
    r::R # cirular buffer, containing last two residuals
    function ConjugateGradient(A::M, b::U, x::V) where {T<:Number,
                    M<:AbstractMatrix{T}, U<:AbstractVector, V<:AbstractVector}
        r₁ = copy(b)
        mul!(r₁, A, x, -1., 1.) # r = b - Ax
        d = copy(r₁)
        r = CircularBuffer{V}(2)
        push!(r, r₁); push!(r, similar(b));
        Ad = similar(b)
        new{T, M, V, typeof(r)}(A, b, d, Ad, r)
    end
end
# no allocations :)
function update!(C::ConjugateGradient, x::AbstractVector, t::Int)
    if t > length(x) # algorithm does not have guarantees after n iterations
        return x # restart?
    elseif t > 1
        # this changes for non-linear problems
        β = sum(abs2, C.r[1]) / sum(abs2, C.r[2]) # new over old residual norm
        @. C.d = β*C.d + C.r[1]  # axpy!
    end
    # this is the only line that is explicitly linear
    mul!(C.Ad, C.A, C.d) # C.Ad = C.A * C.d; could be f(d) if f is nonlinear
    α = sum(abs2, C.r[1]) / dot(C.d, C.Ad)
    @. x += α * C.d # this directly updates x -> <:Update
    circshift!(C.r)
    @. C.r[1] = C.r[2] - α * C.Ad
    return x
end
