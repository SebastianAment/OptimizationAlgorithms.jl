################################################################################
using DataStructures: CircularBuffer

# convenience for circular buffer
# does in place copy of x in front of buffer, important if cb is buffer of arrays
function copyfirst!(cb::CircularBuffer, x)
    cb.first = (cb.first == 1 ? cb.capacity : cb.first - 1)
    if length(cb) < cb.capacity
        cb.length += 1
    end
    copy!(cb.buffer[cb.first], x)
end

# TODO: make negative shifts efficient
# TODO: if shifts > c.length ÷ 2,
function Base.circshift!(cb::CircularBuffer, shifts::Int = 1)
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
