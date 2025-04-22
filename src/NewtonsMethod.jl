module NewtonsMethod

using LinearAlgebra, Statistics, Plots, ForwardDiff

function newtonroot(f, f_prime, x_0; tolerance = 1E-7, maxiter = 1000)
    # setup the algorithm
    x_old = x_0
    normdiff = Inf
    iter = 1
    while normdiff > tolerance && iter <= maxiter
        x_new = x_old - f(x_old) / f_prime(x_old) # use the passed in map
        normdiff = norm(x_new - x_old)
        x_old = x_new
        iter = iter + 1
    end
    return (; root = x_old, normdiff, iter) # A named tuple
end

function newtonroot(f, x_0; tolerance = 1E-7, maxiter = 1000)
    # setup the algorithm
    x_old = x_0
    normdiff = Inf
    iter = 1
    while normdiff > tolerance && iter <= maxiter
        f_prime = ForwardDiff.derivative(f, x_old)
        x_new = x_old - f(x_old) / f_prime 
        normdiff = norm(x_new - x_old)
        x_old = x_new
        iter = iter + 1
    end
    return (; root = x_old, normdiff, iter) # A named tuple
end

export newtonroot

end
