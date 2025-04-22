module NewtonsMethod

using LinearAlgebra, Statistics, Plots, ForwardDiff

function newtonroot(f, f_prime, x_0; tolerance = 1E-7, maxiter = 1000)
    x_old = x_0
    normdiff = Inf
    iter = 1
    while normdiff > tolerance && iter <= maxiter
        x_new = x_old - f(x_old) / f_prime(x_old)
        normdiff = norm(x_new - x_old)
        x_old = x_new
        iter += 1
    end
    return (; root = x_old, normdiff, iter)
end

function newtonroot(f, x_0; tolerance = 1E-7, maxiter = 1000)
    x_old = x_0
    normdiff = Inf
    iter = 1
    while normdiff > tolerance && iter <= maxiter
        f_prime = ForwardDiff.derivative(f, x_old)
        x_new = x_old - f(x_old) / f_prime
        normdiff = norm(x_new - x_old)
        x_old = x_new
        iter += 1
    end
    return (; root = x_old, normdiff, iter)
end

function newtonroot(f, x_0; f_prime::Function = nothing, tolerance = 1E-7, maxiter = 1000)
    if f_prime === nothing
        # Use automatic differentiation if f_prime is not provided
        return newtonroot(f, x_0; tolerance, maxiter)
    else
        # Use the provided derivative function
        return newtonroot(f, f_prime, x_0; tolerance, maxiter)
    end
end

export newtonroot

end