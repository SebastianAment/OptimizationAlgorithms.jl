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
