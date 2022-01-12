################################################################################
# returns dense idenity matrix of eltype T, used in BFGS
identity(T::Type, n::Int) = Matrix(one(T) * I(n))
identity(x::AbstractVecOrMat) = identity(eltype(x), size(x, 1))

################################################################################
using DataStructures: CircularBuffer
# convenience for circular buffer
# does in place copy of x in front of buffer, important if cb is buffer of arrays
# used in LBFGS
function copyfirst!(cb::CircularBuffer, x)
    cb.first = (cb.first == 1 ? cb.capacity : cb.first - 1)
    if length(cb) < cb.capacity
        cb.length += 1
    end
    copy!(cb.buffer[cb.first], x)
end

# TODO: make negative shifts efficient
# TODO: if shifts > c.length ÷ 2,
# used in LBFGS
# Base.circshift
function _circshift!(cb::CircularBuffer, shifts::Int = 1)
    shifts = mod(shifts, cb.length) # since identity if shifts = cb.length
    if shifts > 0
        if cb.length < cb.capacity
            first, last = cb.first, cb.first + cb.length
            for i in 1:shifts
                src = mod1((i-1) + first, cb.capacity)
                dest = mod1(i + last, cb.capacity)
                copy!(cb.buffer[dest], cb.buffer[src])
            end
        end
        cb.first = mod1(cb.first + shifts, cb.capacity)
    end
    return cb
end

######################### Lazy Difference Vector ###############################
# IDEA: is it worth using the LazyArrays package?
# could be replaced by LazyVector(applied(-, x, y)), which is neat.
# lazy difference between two vectors, has no memory footprint
struct LazyDifference{T, U, V} <: AbstractVector{T}
    x::U
    y::V
    function LazyDifference(x, y)
        length(x) == length(y) || throw(DimensionMismatch("x and y do not have the same length: $(length(x)) and $(length(y))."))
        T = promote_type(eltype(x), eltype(y))
        new{T, typeof(x), typeof(y)}(x, y)
    end
end

difference(x::Number, y::Number) = x-y # avoid laziness for scalars
difference(x::AbstractVector, y::AbstractVector) = LazyDifference(x, y)
difference(x::Tuple, y::Tuple) = LazyDifference(x, y)
# this creates a type instability:
Base.size(d::LazyDifference) = (length(d.x),)
Base.getindex(d::LazyDifference, i::Integer) = d.x[i]-d.y[i]

######################### solves with eigen factorization, used for SaddleFreeNewton
function LinearAlgebra.ldiv!(y::AbstractVector, E::Eigen, x::AbstractVector)
     ldiv!!(y, E, copy(x))
end
# WARNING: uses x as a temporary
function ldiv!!(y::AbstractVector, E::Eigen, x::AbstractVector) # temporary
    mul!(y, E.vectors', x)
    @. x = y / E.values
    mul!(y, E.vectors, x)
end
