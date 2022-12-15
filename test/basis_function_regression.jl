@testset "basis_function_regression" begin
    @testset "basis_func_blr $Tx" for Tx in [Matrix, ColVecs, RowVecs]
        @testset "self-consistency" begin
            rng, N, D, samples = MersenneTwister(123456), 11, 2, 1_000_000
            X, f, Σy = generate_toy_problem(rng, N, D, Tx)

            f_bf = BasisFunctionRegressor(f, ϕ)

            AbstractGPs.TestUtils.test_finitegp_primary_and_secondary_public_interface(
                rng, f_bf(X, Σy)
            )
        end
        @testset "consistency with BLR" begin
            rng = MersenneTwister(123456)
            N = 11
            D = 2
            X, f, Σy = generate_toy_problem(rng, N, D, Tx)
            f_bf = BasisFunctionRegressor(f, ϕ)

            # Compute logpdf using both the BLR and BFR. Should agree.
            y = rand(rng, f_bf(X, Σy))
            @test logpdf(f(ϕ(X), Σy), y) ≈ logpdf(f_bf(X, Σy), y)

            # Check that posteriors agree.
            f_bf_post = posterior(f_bf(X, Σy), y)
            f_post = posterior(f(ϕ(X), Σy), y)
            @test mean(f_bf_post(X)) ≈ mean(f_post(ϕ(X)))
        end
        @testset "rand" begin
            rng, N, D, samples = MersenneTwister(123456), 11, 2, 1_000_000
            X, f, Σy = generate_toy_problem(rng, N, D, Tx)

            f_bf = BasisFunctionRegressor(f, ϕ)

            # Roughly test the statistical properties of rand.
            Y = rand(rng, f_bf(X, Σy), samples)
            m_empirical = mean(Y; dims=2)
            Σ_empirical = (Y .- mean(Y; dims=2)) * (Y .- mean(Y; dims=2))' ./ samples
            @test mean(f_bf(X, Σy)) ≈ m_empirical atol = 1e-2 rtol = 1e-2
            @test cov(f_bf(X, Σy)) ≈ Σ_empirical atol = 1e-2 rtol = 1e-2
        end
    end
end
