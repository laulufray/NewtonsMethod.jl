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

function newtonroot(f, x_0; f_prime=nothing, tolerance=1E-7, maxiter=1000)
    x_old = x_0
    normdiff = Inf
    iter = 1

    while normdiff > tolerance && iter <= maxiter
        if f_prime === nothing
           # Use automatic differentiation
           f_prime = ForwardDiff.derivative(f, x_old)
        end
        x_new = x_old - f(x_old) / f_prime
        normdiff = norm(x_new - x_old)
        x_old = x_new
        iter += 1
    end

    return (; root = x_old, normdiff, iter)
end

export newtonroot

end