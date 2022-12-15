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
            rng, N, D, samples = MersenneTwister(123456), 11, 3, 1_000_000
            X, f, Σy = generate_toy_problem(rng, N, D, Tx)

            # Roughly test the statistical properties of rand.
            Y = rand(rng, f(X, Σy), samples)
            m_empirical = mean(Y; dims=2)
            Σ_empirical = (Y .- mean(Y; dims=2)) * (Y .- mean(Y; dims=2))' ./ samples
            @test mean(f(X, Σy)) ≈ m_empirical atol = 1e-2 rtol = 1e-2
            @test cov(f(X, Σy)) ≈ Σ_empirical atol = 1e-2 rtol = 1e-2

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
            @testset "PDMat weight matrix closure" begin
                # Copied from https://github.com/JuliaLang/julia/pull/39352/files
                @testset "constructing a Cholesky factor from a triangular matrix" begin
                    A = [1.0 2.0; 3.0 4.0]
                    let
                        U = UpperTriangular(A)
                        C = Cholesky(U)
                        @test C isa Cholesky{Float64}
                        @test C.U == U
                        @test C.L == U'
                    end
                    let
                        L = LowerTriangular(A)
                        C = Cholesky(L)
                        @test C isa Cholesky{Float64}
                        @test C.L == L
                        @test C.U == L'
                    end
                end
                rng, N, D = MersenneTwister(123456), 13, 7
                X = randn(rng, D, N)
                X′ = randn(rng, D, N)
                B = randn(rng, D, D)
                U = UpperTriangular(B)
                C = 0.1 * randn(rng, N, N)
                mw, Σy = randn(rng, D), C * C' + I

                Λw_pd = PDMat(Cholesky(U)) + I
                f_pd = BayesianLinearRegressor(mw, Λw_pd)
                fx_pd = f_pd(X, Σy)

                Λw_sym = Symmetric(U'U + I)
                f_sym = BayesianLinearRegressor(mw, Λw_sym)
                fx_sym = f_sym(X, Σy)

                y = rand(rng, fx_pd)
                f′pd = posterior(fx_pd, y)
                f′sym = posterior(fx_sym, y)
                @test f′pd.Λw isa PDMat
                @test f′sym.Λw isa Symmetric
                @test mean(f′pd(X′, Σy)) ≈ mean(f′sym(X′, Σy))
                @test cov(f′pd(X′, Σy)) ≈ cov(f′sym(X′, Σy))
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
