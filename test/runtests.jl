using NewtonsMethod
using Test

@testset "NewtonsMethod" begin
    # Test 1: Root of a quadratic function
    f(x) = x^2 - 4
    f_prime(x) = 2x
    @test newtonroot(f, f_prime; x_0=1.0) ≈ 2.0
    @test newtonroot(f, f_prime; x_0=-1.0) ≈ -2.0

    # Test 2: Automatic differentiation
    @test newtonroot(f; x_0=1.0) ≈ 2.0
    @test newtonroot(f; x_0=-1.0) ≈ -2.0

    # Test 3: Non-convergence for a function without a root
    g(x) = x^2 + 2
    @test newtonroot(g, x -> 2x; x_0=1.0, maxiter=100) === nothing

    # Test 4: BigFloat precision
    h(x) = BigFloat(x)^2 - BigFloat(4)
    h_prime(x) = 2 * BigFloat(x)
    @test newtonroot(h, h_prime; x_0=BigFloat(1.0)) ≈ BigFloat(2.0)

    # Test 5: Max iterations
    @test newtonroot(f, f_prime; x_0=1.0, maxiter=1) === nothing

    # Test 6: Tolerance
    @test newtonroot(f, f_prime; x_0=1.0, tol=1e-1) ≈ 2.0
end
