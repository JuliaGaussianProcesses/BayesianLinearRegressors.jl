# Generate a toy problem without any obvious structure in the mean, precision, or noise std.
# Important to ensure that the unit tests don't just pass for a special case by accident.
# Everything should be reasonably well conditioned.
function generate_toy_problem(rng, N, D)
    X, B = randn(rng, D, N), randn(rng, D, D)
    mw, Λw, σ² = randn(rng, D), B * B' + I, abs(randn(rng) + 0.1)
    return X, BayesianLinearRegressor(mw, Λw, σ²)
end

@testset "blr" begin
    @testset "marginals" begin
        rng, N, D, samples = MersenneTwister(123456), 11, 3, 1_000_000
        X, f = generate_toy_problem(rng, N, D)

        @test mean.(marginals(f(X))) == mean(f(X))
        @test std.(marginals(f(X))) == sqrt.(diag(cov(f(X))))
    end
    @testset "rand" begin
        rng, N, D, samples = MersenneTwister(123456), 11, 3, 1_000_000
        X, f = generate_toy_problem(rng, N, D)

        # Check deterministic properties of rand.
        @test size(rand(rng, f(X))) == (N,)
        @test size(rand(rng, f(X), samples)) == (N, samples)

        # Roughly test the statistical properties of rand.
        Y = rand(rng, f(X), samples)
        m_empirical = mean(Y; dims=2)
        Σ_empirical = (Y .- mean(Y; dims=2)) * (Y .- mean(Y; dims=2))' ./ samples
        @test mean(f(X)) ≈ m_empirical atol=1e-3 rtol=1e-3
        @test cov(f(X)) ≈ Σ_empirical atol=1e-3 rtol=1e-3
    end
    @testset "logpdf" begin
        rng, N, D = MersenneTwister(123456), 13, 7
        X, f = generate_toy_problem(rng, N, D)
        y = rand(rng, f(X))

        # Construct MvNormal using a naive but simple computation for the mean / cov.
        m, Σ = X' * f.mw, Symmetric(X' * (cholesky(f.Λw) \ X) + f.σ² * I)

        # Check that logpdf agrees between distributions and BLR.
        @test logpdf(f(X), y) ≈ logpdf(MvNormal(m, Σ), y)
    end
    @testset "posterior" begin
        @testset "low noise" begin
            rng, N, D = MersenneTwister(123456), 13, 7
            X, f = generate_toy_problem(rng, N, D)
            f_low_noise = BayesianLinearRegressor(f.mw, f.Λw, eps())
            y = rand(rng, f_low_noise(X))

            f′_low_noise = posterior(f_low_noise, X, y)
            @test mean(f′_low_noise(X)) ≈ y
            @test all(cov(f′_low_noise(X)) .< 1_000 * eps())
        end
        @testset "repeated conditioning" begin
            rng, N, D = MersenneTwister(123456), 13, 7
            X, f = generate_toy_problem(rng, N, D)
            X′ = randn(rng, D, N)
            y = rand(rng, f(X))

            N1 = N - 3
            X1, X2 = X[:, 1:N1], X[:, N1+1:end]
            y1, y2 = y[1:N1], y[N1+1:end]

            f′1 = posterior(f, X1, y1)
            f′2 = posterior(f′1, X2, y2)
            f′ = posterior(f, X, y)
            @test mean(f′(X′)) ≈ mean(f′2(X′))
            @test cov(f′(X′)) ≈ cov(f′2(X′))
        end
    end
end
