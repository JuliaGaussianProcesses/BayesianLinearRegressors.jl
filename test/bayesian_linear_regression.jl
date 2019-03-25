# Generate a toy problem without any obvious structure in the mean, precision, or noise std.
# Important to ensure that the unit tests don't just pass for a special case by accident.
# Everything should be reasonably well conditioned.
function generate_toy_problem(rng, N, D)
    X, B, C = randn(rng, D, N), randn(rng, D, D), 0.1 * randn(rng, N, N)
    mw, Λw, Σy = randn(rng, D), B * B' + I, C * C' + I
    return X, BayesianLinearRegressor(mw, Λw), Σy
end

function FDM.j′vp(fdm, f, ȳ::Real, X::AbstractArray)
    return reshape(FDM.j′vp(fdm, x->[f(reshape(x, size(X)))], [ȳ], vec(X)), size(X))
end
function FDM.j′vp(fdm, f, Ȳ::AbstractArray, X::AbstractArray)
    return reshape(FDM.j′vp(fdm, x->vec(f(reshape(x, size(X)))), vec(Ȳ), vec(X)), size(X))
end

@testset "blr" begin
    @testset "marginals" begin
        rng, N, D, samples = MersenneTwister(123456), 11, 3, 1_000_000
        X, f, Σy = generate_toy_problem(rng, N, D)

        @test mean.(marginals(f(X, Σy))) == mean(f(X, Σy))
        @test std.(marginals(f(X, Σy))) == sqrt.(diag(cov(f(X, Σy))))
    end
    @testset "rand" begin
        rng, N, D, samples = MersenneTwister(123456), 11, 3, 10_000_000
        X, f, Σy = generate_toy_problem(rng, N, D)

        # Check deterministic properties of rand.
        @test size(rand(rng, f(X, Σy))) == (N,)
        @test size(rand(rng, f(X, Σy), samples)) == (N, samples)

        # Roughly test the statistical properties of rand.
        Y = rand(rng, f(X, Σy), samples)
        m_empirical = mean(Y; dims=2)
        Σ_empirical = (Y .- mean(Y; dims=2)) * (Y .- mean(Y; dims=2))' ./ samples
        @test mean(f(X, Σy)) ≈ m_empirical atol=1e-3 rtol=1e-3
        @test cov(f(X, Σy)) ≈ Σ_empirical atol=1e-3 rtol=1e-3

        @testset "Zygote (everything dense)" begin
            function rand_blr(X, A_Σy, mw, A_Λw)
                Σy, Λw = Symmetric(A_Σy * A_Σy' + I), Symmetric(A_Λw * A_Λw' + I)
                f = BayesianLinearRegressor(mw, Λw)
                return rand(MersenneTwister(123456), f(X, Σy), 3)
            end
            mw, A_Σy, A_Λw = f.mw, 0.1 .* randn(rng, N, N), 0.1 .* randn(rng, D, D)

            # Run the model forwards and check that output agrees with non-Zygote output.
            z, back = Zygote.forward(rand_blr, X, A_Σy, mw, A_Λw)
            @test z == rand_blr(X, A_Σy, mw, A_Λw)

            # Compute adjoints using Zygote.
            z̄ = randn(rng, size(z))
            dX, dA_Σy, dmw, dA_Λw = back(z̄)

            # Verify adjoints via finite differencing.
            fdm = central_fdm(5, 1)
            @test dX ≈ j′vp(fdm, X->rand_blr(X, A_Σy, mw, A_Λw), z̄, X)
            @test dA_Σy ≈ j′vp(fdm, A_Σy->rand_blr(X, A_Σy, mw, A_Λw), z̄, A_Σy)
            @test dmw ≈ j′vp(fdm, mw->rand_blr(X, A_Σy, mw, A_Λw), z̄, mw)
            @test dA_Λw ≈ j′vp(fdm, A_Λw->rand_blr(X, A_Σy, mw, A_Λw), z̄, A_Λw)
        end
    end
    @testset "logpdf" begin
        rng, N, D = MersenneTwister(123456), 13, 7
        X, f, Σy = generate_toy_problem(rng, N, D)
        y = rand(rng, f(X, Σy))

        # Construct MvNormal using a naive but simple computation for the mean / cov.
        m, Σ = X' * f.mw, Symmetric(X' * (cholesky(f.Λw) \ X) + Σy)

        # Check that logpdf agrees between distributions and BLR.
        @test logpdf(f(X, Σy), y) ≈ logpdf(MvNormal(m, Σ), y)

        @testset "Zygote (everything dense)" begin
            function logpdf_blr(X, A_Σy, y, mw, A_Λw)
                Σy, Λw = Symmetric(A_Σy * A_Σy' + I), Symmetric(A_Λw * A_Λw' + I)
                f = BayesianLinearRegressor(mw, Λw)
                return logpdf(f(X, Σy), y)
            end
            mw, A_Σy, A_Λw = f.mw, 0.1 .* randn(rng, N, N), 0.1 .* randn(rng, D, D)

            z, back = Zygote.forward(logpdf_blr, X, A_Σy, y, mw, A_Λw)
            @test z == logpdf_blr(X, A_Σy, y, mw, A_Λw)

            # Compute gradients using Zygote.
            z̄ = randn(rng)
            dX, dA_Σy, dy, dmw, dA_Λw = back(z̄)

            # Check correctness via finite differencing.
            fdm = central_fdm(5, 1)
            @test dX ≈ j′vp(fdm, X->logpdf_blr(X, A_Σy, y, mw, A_Λw), z̄, X)
            @test dA_Σy ≈ j′vp(fdm, A_Σy->logpdf_blr(X, A_Σy, y, mw, A_Λw), z̄, A_Σy)
            @test dy ≈ j′vp(fdm, y->logpdf_blr(X, A_Σy, y, mw, A_Λw), z̄, y)
            @test dmw ≈ j′vp(fdm, mw->logpdf_blr(X, A_Σy, y, mw, A_Λw), z̄, mw)
            @test dA_Λw ≈ j′vp(fdm, A_Λw->logpdf_blr(X, A_Σy, y, mw, A_Λw), z̄, A_Λw)
        end
    end
    @testset "posterior" begin
        @testset "low noise" begin
            rng, N, D = MersenneTwister(123456), 13, 7
            X, f, Σy = generate_toy_problem(rng, N, D)
            y = rand(rng, f(X, eps()))

            f′_low_noise = posterior(f(X, eps()), y)
            @test mean(f′_low_noise(X, eps())) ≈ y
            @test all(cov(f′_low_noise(X, eps())) .< 1_000 * eps())
        end
        @testset "repeated conditioning" begin
            rng, N, D = MersenneTwister(123456), 13, 7
            X, f, Σy = generate_toy_problem(rng, N, D)
            X′ = randn(rng, D, N)
            y = rand(rng, f(X, Σy))

            # Chop up the noise because we can't condition on noise that's correlated
            # between things.
            N1 = N - 3
            Σ1, Σ2 = Σy[1:N1, 1:N1], Σy[N1+1:end, N1+1:end]
            Σy′ = vcat(
                hcat(Σ1, zeros(N1, N - N1)),
                hcat(zeros(N - N1, N1), Σ2),
            )

            X1, X2 = X[:, 1:N1], X[:, N1+1:end]
            y1, y2 = y[1:N1], y[N1+1:end]

            f′1 = posterior(f(X1, Σ1), y1)
            f′2 = posterior(f′1(X2, Σ2), y2)
            f′ = posterior(f(X, Σy′), y)
            @test mean(f′(X′, Σy)) ≈ mean(f′2(X′, Σy))
            @test cov(f′(X′, Σy)) ≈ cov(f′2(X′, Σy))
        end
    end
end
