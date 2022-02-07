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

            @testset "Zygote (everything dense)" begin
                function test_rand_funcs_adjoints(sample_function)
                    rng, N, D = MersenneTwister(123456), 11, 5
                    X, f, _ = generate_toy_problem(rng, N, D, Tx)
                    mw, A_Λw = f.mw, 0.1 .* randn(rng, D, D)

                    # Run the model forwards and check that output agrees with non-Zygote.
                    z, back = Zygote.pullback(sample_function, X, mw, A_Λw)
                    @test z == sample_function(X, mw, A_Λw)

                    # Compute adjoints using Zygote.
                    z̄ = randn(rng, size(z))
                    dX, dmw, dA_Λw = back(z̄)

                    # Verify adjoints via finite differencing.
                    fdm = central_fdm(5, 1)
                    @test dX ≈ first(j′vp(fdm, X -> sample_function(X, mw, A_Λw), z̄, X))
                    @test dmw ≈ first(j′vp(fdm, mw -> sample_function(X, mw, A_Λw), z̄, mw))
                    @test dA_Λw ≈
                        first(j′vp(fdm, A_Λw -> sample_function(X, mw, A_Λw), z̄, A_Λw))
                end

                function rand_funcs_single(X, mw, A_Λw)
                    Λw = Symmetric(A_Λw * A_Λw' + I)
                    f = BayesianLinearRegressor(mw, Λw)
                    g = rand(MersenneTwister(123456), f)
                    return g(X)
                end

                function rand_funcs_multi(X, mw, A_Λw)
                    Λw = Symmetric(A_Λw * A_Λw' + I)
                    f = BayesianLinearRegressor(mw, Λw)
                    gs = rand(MersenneTwister(123456), f, 1, 1)
                    return reduce(hcat, map(h -> h(X), reshape(gs, :)))
                end

                test_rand_funcs_adjoints(rand_funcs_single)
                test_rand_funcs_adjoints(rand_funcs_multi)
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
