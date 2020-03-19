########################## Stochastic Approximations ###########################
# f should be dependent on data x and parameters θ, f(θ, x)
# possibly extend to f(θ, x, y) for regression
# Idea: stochastic and online could be subsumed if we have an infinite vector
# whose getindex method always returns random elements (maybe that is not a good definition)
# computes stochastic approximation to f by randomly samling from x with replacement
# TODO: add option to sample without replacement
function stochastic(f, x::AbstractVector)
    return function sf(θ, batchsize::Int = 1)
        subset = sort!(rand(1:length(x), batchsize)) # without replacement # with replacement: sample(1:length(x), batchsize, replace = false)
        f(θ, x[subset])
    end
end

# stochastic approximation of f(θ, x) by going through x sequentially (in online fashion)
# combine this with PeriodicVector in MyLinearAlgebra to get infinite stream of data
function online(f, x::AbstractVector)
    return function sf(θ, t::Int, batchsize::Int = 1)
        subset = t*batchsize:(t+1)*batchsize
        f(θ, x[subset])
    end
end
