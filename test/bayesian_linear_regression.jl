# Generate a toy problem without any obvious structure in the mean, precision, or noise std.
# Important to ensure that the unit tests don't just pass for a special case by accident.
# Everything should be reasonably well conditioned.
function generate_toy_problem(rng, N, D, ::Type{<:Matrix})
    X = randn(rng, D, N)
    B = randn(rng, D, D)
    C = 0.1 * randn(rng, N, N)
    mw, Λw, Σy = randn(rng, D), B * B' + I, C * C' + I
    return X, BayesianLinearRegressor(mw, Λw), Σy
end

function generate_toy_problem(rng, N, D, ::Type{<:ColVecs})
    X, f, Σy = generate_toy_problem(rng, N, D, Matrix)
    return ColVecs(X), f, Σy
end

function generate_toy_problem(rng, N, D, ::Type{<:RowVecs})
    X, f, Σy = generate_toy_problem(rng, N, D, Matrix)
    return RowVecs(collect(X')), f, Σy
end

# Some type-piracy.
Base.isapprox(dx::NamedTuple{(:X,)}, dy::ColVecs) = isapprox(dx.X, dy.X)

Base.isapprox(dx::NamedTuple{(:X,)}, dy::RowVecs) = isapprox(dx.X, dy.X)

@testset "bayesian_linear_regression" begin
    @testset "blr $Tx" for Tx in [Matrix, ColVecs, RowVecs]
        @testset "consistency" begin
            rng, N, D, samples = MersenneTwister(123456), 11, 3, 1_000_000
            X, f, Σy = generate_toy_problem(rng, N, D, Tx)

            AbstractGPs.TestUtils.test_finitegp_primary_and_secondary_public_interface(
                rng, f(X, Σy)
            )
        end
        @testset "rand" begin
            rng, N, D, samples = MersenneTwister(123456), 11, 3, 10_000_000
            X, f, Σy = generate_toy_problem(rng, N, D, Tx)

            # Roughly test the statistical properties of rand.
            Y = rand(rng, f(X, Σy), samples)
            m_empirical = mean(Y; dims=2)
            Σ_empirical = (Y .- mean(Y; dims=2)) * (Y .- mean(Y; dims=2))' ./ samples
            @test mean(f(X, Σy)) ≈ m_empirical atol = 1e-3 rtol = 1e-3
            @test cov(f(X, Σy)) ≈ Σ_empirical atol = 1e-3 rtol = 1e-3

            @testset "Zygote (everything dense)" begin
                function rand_blr(X, A_Σy, mw, A_Λw)
                    Σy, Λw = Symmetric(A_Σy * A_Σy' + I), Symmetric(A_Λw * A_Λw' + I)
                    f = BayesianLinearRegressor(mw, Λw)
                    return rand(MersenneTwister(123456), f(X, Σy), 3)
                end
                mw, A_Σy, A_Λw = f.mw, 0.1 .* randn(rng, N, N), 0.1 .* randn(rng, D, D)

                # Run the model forwards and check that output agrees with non-Zygote output
                z, back = Zygote.pullback(rand_blr, X, A_Σy, mw, A_Λw)
                @test z == rand_blr(X, A_Σy, mw, A_Λw)

                # Compute adjoints using Zygote.
                z̄ = randn(rng, size(z))
                dX, dA_Σy, dmw, dA_Λw = back(z̄)

                # Verify adjoints via finite differencing.
                fdm = central_fdm(5, 1)
                @test dX ≈ first(j′vp(fdm, X -> rand_blr(X, A_Σy, mw, A_Λw), z̄, X))
                @test dA_Σy ≈
                    first(j′vp(fdm, A_Σy -> rand_blr(X, A_Σy, mw, A_Λw), z̄, A_Σy))
                @test dmw ≈ first(j′vp(fdm, mw -> rand_blr(X, A_Σy, mw, A_Λw), z̄, mw))
                @test dA_Λw ≈
                    first(j′vp(fdm, A_Λw -> rand_blr(X, A_Σy, mw, A_Λw), z̄, A_Λw))
            end
        end
        @testset "logpdf" begin
            rng, N, D = MersenneTwister(123456), 13, 7
            X, f, Σy = generate_toy_problem(rng, N, D, Tx)
            y = rand(rng, f(X, Σy))

            # Construct MvNormal using a naive but simple computation for the mean / cov.
            function naive_normal_stats(X::Matrix)
                return (X' * f.mw, Symmetric(X' * (cholesky(f.Λw) \ X) + Σy))
            end
            naive_normal_stats(X::ColVecs) = naive_normal_stats(X.X)
            naive_normal_stats(X::RowVecs) = naive_normal_stats(collect(X.X'))
            m, Σ = naive_normal_stats(X)

            # Check that logpdf agrees between distributions and BLR.
            @test logpdf(f(X, Σy), y) ≈ logpdf(MvNormal(m, Σ), y)

            @testset "Zygote (everything dense)" begin
                function logpdf_blr(X, A_Σy, y, mw, A_Λw)
                    Σy, Λw = Symmetric(A_Σy * A_Σy' + I), Symmetric(A_Λw * A_Λw' + I)
                    f = BayesianLinearRegressor(mw, Λw)
                    return logpdf(f(X, Σy), y)
                end
                mw, A_Σy, A_Λw = f.mw, 0.1 .* randn(rng, N, N), 0.1 .* randn(rng, D, D)

                z, back = Zygote.pullback(logpdf_blr, X, A_Σy, y, mw, A_Λw)
                @test z == logpdf_blr(X, A_Σy, y, mw, A_Λw)

                # Compute gradients using Zygote.
                z̄ = randn(rng)
                dX, dA_Σy, dy, dmw, dA_Λw = back(z̄)

                # Check correctness via finite differencing.
                fdm = central_fdm(5, 1)
                @test dX ≈ first(j′vp(fdm, X -> logpdf_blr(X, A_Σy, y, mw, A_Λw), z̄, X))
                @test dA_Σy ≈
                    first(j′vp(fdm, A_Σy -> logpdf_blr(X, A_Σy, y, mw, A_Λw), z̄, A_Σy))
                @test dy ≈ first(j′vp(fdm, y -> logpdf_blr(X, A_Σy, y, mw, A_Λw), z̄, y))
                @test dmw ≈ first(j′vp(fdm, mw -> logpdf_blr(X, A_Σy, y, mw, A_Λw), z̄, mw))
                @test dA_Λw ≈
                    first(j′vp(fdm, A_Λw -> logpdf_blr(X, A_Σy, y, mw, A_Λw), z̄, A_Λw))
            end
        end
        @testset "posterior" begin
            @testset "low noise" begin
                rng, N, D = MersenneTwister(123456), 13, 7
                X, f, Σy = generate_toy_problem(rng, N, D, Tx)
                y = rand(rng, f(X, eps()))

                f′_low_noise = posterior(f(X, eps()), y)
                @test mean(f′_low_noise(X, eps())) ≈ y
                @test all(cov(f′_low_noise(X, eps())) .< 1_000 * eps())
            end
            @testset "repeated conditioning" begin
                rng, N, D = MersenneTwister(123456), 13, 7
                X, f, Σy = generate_toy_problem(rng, N, D, Tx)
                X′ = randn(rng, D, N)
                y = rand(rng, f(X, Σy))

                # Chop up the noise because we can't condition on noise that's correlated
                # between things.
                N1 = N - 3
                Σ1, Σ2 = Σy[1:N1, 1:N1], Σy[(N1 + 1):end, (N1 + 1):end]
                Σy′ = vcat(hcat(Σ1, zeros(N1, N - N1)), hcat(zeros(N - N1, N1), Σ2))

                X1 = X isa Matrix ? X[:, 1:N1] : X[1:N1]
                X2 = X isa Matrix ? X[:, (N1 + 1):end] : X[(N1 + 1):end]
                y1, y2 = y[1:N1], y[(N1 + 1):end]

                f′1 = posterior(f(X1, Σ1), y1)
                f′2 = posterior(f′1(X2, Σ2), y2)
                f′ = posterior(f(X, Σy′), y)
                @test mean(f′(X′, Σy)) ≈ mean(f′2(X′, Σy))
                @test cov(f′(X′, Σy)) ≈ cov(f′2(X′, Σy))
            end
        end

        @testset "sampling functions" begin
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
    end
    @testset "unrecognised AbstractVector" begin
        rng = MersenneTwister(123456)
        N, D = 11, 5
        x = collect(eachrow(randn(rng, N, D)))
        _, f, Σy = generate_toy_problem(rng, N, D, ColVecs)
        @test_throws ErrorException rand(f(x, Σy))
    end
end
