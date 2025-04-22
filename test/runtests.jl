using NewtonsMethod
using Test

@testset "NewtonsMethod Tests" begin
    # Test 1: Root of a quadratic function using given derivative
    f(x) = x^2 - 4
    f_prime(x) = 2x
    result1 = newtonroot(f, f_prime, 1.0)
    result2 = newtonroot(f, f_prime, -1.0)
    @test isapprox(result1.root, 2.0; atol=1e-1)
    @test isapprox(result2.root, -2.0; atol=1e-1)
    
    # Test 2: Automatic differentiation
    result3 = newtonroot(f, 1.0)
    result4 = newtonroot(f, -1.0)
    @test isapprox(result3.root, 2.0; atol=1e-1)
    @test isapprox(result4.root, -2.0; atol=1e-1)
    
    # Test 3: Non-convergence for a function without a root
    g(x) = x^2 + 2
    result5 = newtonroot(g, x -> 2x, 1.0; maxiter=100)
    @test result5.normdiff > 1E-7

    # Test 4: BigFloat precision
    h(x) = BigFloat(x)^2 - BigFloat(4)
    h_prime(x) = 2 * BigFloat(x)
    result6 = newtonroot(h, h_prime, BigFloat(1.0))
    @test isapprox(result6.root, BigFloat(2.0); atol=1e-12)

    # Test 5: Max iterations enforced: when maxiter=1 the iteration count should be 2
    result7 = newtonroot(f, f_prime, 1.0; maxiter=1)
    @test result7.iter == 2

    # Test 6: Custom tolerance: using tolerance=1e-1 should converge to approximately 2.0
    result8 = newtonroot(f, f_prime, 1.0; tolerance=1e-1)
    @test isapprox(result8.root, 2.0; atol=1e-1)
end