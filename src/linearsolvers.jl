# TODO: more iterative methods: GMRES, Arnoldi, Lanczos
################################################################################
# TODO: use circular buffer to get rid of copy
# TODO: preconditioning, could be external by running cg on
# EAE'x = Eb where inv(E)inv(E)' = M where M is preconditioner
# TODO: nonlinear
# β = polyakribiere(dx₁, dx₂) = dx₁'(dx₁-dx₂) / (dx₂'dx₂)
# Careful with ill-conditioned systems
struct ConjugateGradient{T, M, V} <: Update{T}
    # f::F # nonlinear conjugate gradient
    A::M # linear conjugate gradient
    b::V # target
    r₁::V # last two residuals
    r₂::V
    d::V # direction
    Ad::V # product of A and direction
    function ConjugateGradient(A::M, b::U, x::V) where {T<:Number,
                    M<:AbstractMatrix{T}, U<:AbstractVector, V<:AbstractVector}
        r₁ = copy(b)
        mul!(r₁, A, x, -1., 1.) # r = b - Ax
        d = copy(r₁)
        r₂ = similar(b)
        Ad = similar(b)
        new{T, M, V}(A, b, r₁, r₂, d, Ad)
    end
end
# no allocations :)
function update!(C::ConjugateGradient, x::AbstractVector, t::Int)
    if t > 1
        # this changes for non-linear problems
        β = sum(abs2, C.r₁) / sum(abs2, C.r₂) # new over old residual norm
        @. C.d = β*C.d + C.r₁  # axpy!
    elseif t ≥ length(x) # algorithm does not have guarantees after n iterations
        return x
    end
    # this is the only line that is explicitly linear
    mul!(C.Ad, C.A, C.d) # C.Ad = C.A * C.d; could be f(d) if f is nonlinear
    α = sum(abs2, C.r₁) / dot(C.d, C.Ad)
    @. x += α * C.d # this directly updates x -> <:Update
    copy!(C.r₂, C.r₁) # could get around doing this copy with circular buffer
    @. C.r₁ -= α * C.Ad
    x
end
