################################################################################
using DataStructures: CircularBuffer

# convenience for circular buffer
# does in place copy of x in front of buffer, important if cb is buffer of arrays
function copyfirst!(cb::CircularBuffer, x)
    cb.first = (cb.first == 1 ? cb.capacity : cb.first - 1)
    if length(cb) < cb.capacity
        cb.length += 1 # this probably shouldn't happen in this context
    end
    copy!(cb.buffer[cb.first], x)
end
