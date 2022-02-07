@testset "basis_function_regression" begin
    @testset "basis_func_blr $Tx" for Tx in [Matrix, ColVecs, RowVecs]
        @testset "consistency" begin
            rng, N, D, samples = MersenneTwister(123456), 11, 2, 1_000_000
            X, f, Σy = generate_toy_problem(rng, N, D, Tx)

            f_bf = BasisFunctionRegressor(f, ϕ)

            AbstractGPs.TestUtils.test_finitegp_primary_and_secondary_public_interface(
                rng, f_bf(X, Σy)
            )
        end
        @testset "rand" begin
            rng, N, D, samples = MersenneTwister(123456), 11, 2, 10_000_000
            X, f, Σy = generate_toy_problem(rng, N, D, Tx)

            f_bf = BasisFunctionRegressor(f, ϕ)

            # Roughly test the statistical properties of rand.
            Y = rand(rng, f_bf(X, Σy), samples)
            m_empirical = mean(Y; dims=2)
            Σ_empirical = (Y .- mean(Y; dims=2)) * (Y .- mean(Y; dims=2))' ./ samples
            @test mean(f_bf(X, Σy)) ≈ m_empirical atol = 1e-3 rtol = 1e-3
            @test cov(f_bf(X, Σy)) ≈ Σ_empirical atol = 1e-3 rtol = 1e-3
        end
    end
end
