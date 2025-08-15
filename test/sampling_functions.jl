@testset "sampling_functions" begin
    @testset "sampling input $Tx" for Tx in [Matrix, ColVecs, RowVecs]
        @testset "bayesian_linear_regression" begin
            rng, N, D = MersenneTwister(123456), 11, 5
            X, f, Σy = generate_toy_problem(rng, N, D, Tx)

            g = rand(rng, f)
            @test g(X) == g(X)  # check the sample doesn't change between evaluations

            if X isa Matrix
                Xc = ColVecs(X)
                Xr = RowVecs(X')
                @test g(X) == g(Xc)
                @test g(X) == g(Xr)
            end

            # test the Random interface
            @test rand(rng, Random.Sampler(rng, f, Val(Inf))) isa
                BayesianLinearRegressors.BLRFunctionSample

            samples1, samples2 = 1000, 1000
            samples = samples1 * samples2
            gs = rand(rng, f, samples1, samples2)
            @test size(gs) == (samples1, samples2)

            # test statistical properties of the sampled functions
            let
                Y = reduce(hcat, map(h -> h(X), reshape(gs, :)))
                m_empirical = mean(Y; dims=2)
                Σ_empirical = (Y .- mean(Y; dims=2)) * (Y .- mean(Y; dims=2))' ./ samples
                @test mean(f(X, Σy)) ≈ m_empirical atol = 1e-3 rtol = 1e-3
                @test cov(f(X, Σy)) ≈ Σ_empirical + Σy atol = 1e-2 rtol = 1e-2
            end

            # test statistical properties of in-place rand
            let
                A = Array{BayesianLinearRegressors.BLRFunctionSample,2}(
                    undef, samples1, samples2
                )
                A = rand!(rng, A, f)
                Y = reduce(hcat, map(h -> h(X), reshape(gs, :)))
                m_empirical = mean(Y; dims=2)
                Σ_empirical = (Y .- mean(Y; dims=2)) * (Y .- mean(Y; dims=2))' ./ samples
                @test mean(f(X, Σy)) ≈ m_empirical atol = 1e-3 rtol = 1e-3
                @test cov(f(X, Σy)) ≈ Σ_empirical + Σy atol = 1e-2 rtol = 1e-2
            end
        end

        @testset "basis_function_regression" begin
            rng, N, D = MersenneTwister(123456), 11, 2
            X, f, Σy = generate_toy_problem(rng, N, D, Tx)

            f_bf = BasisFunctionRegressor(f, ϕ)

            samples = 1_000_000
            gs = rand(rng, f_bf, samples)

            # test statistical properties of the sampled functions
            let
                Y = reduce(hcat, map(h -> h(X), reshape(gs, :)))
                m_empirical = mean(Y; dims=2)
                Σ_empirical = (Y .- mean(Y; dims=2)) * (Y .- mean(Y; dims=2))' ./ samples
                @test mean(f_bf(X, Σy)) ≈ m_empirical atol = 1e-3 rtol = 1e-3
                @test cov(f_bf(X, Σy)) ≈ Σ_empirical + Σy atol = 1e-2 rtol = 1e-2
            end
        end
    end
end
