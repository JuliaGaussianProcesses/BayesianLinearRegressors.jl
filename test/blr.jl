# Generate a toy problem without any obvious structure in the mean, precision, or noise std.
# Important to ensure that the unit tests don't just pass for a special case by accident.
# Everything should be reasonably well conditioned.
function generate_toy_problem(rng, N, D)
    X, B = randn(rng, D, N), randn(rng, D, D)
    mw, Λw, σ² = randn(rng, D), B * B' + I, abs(randn(rng) + 0.1)
    return X, BayesianLinearRegressor(mw, Λw, σ²)
end

@testset "blr.jl" begin
    @testset "rand" begin
        # check that results agree with marginals.
    end
    @testset "logpdf" begin

        # Set up generic toy problem.
        rng, N, D = MersenneTwister(123456), 10, 2
        X, f = generate_toy_problem(rng, N, D)
        y = rand(rng, f(X))

        # Construct MvNormal using a naive but simple computation for the mean / cov.
        m, Σ = X' * f.mw, X' * (cholesky(f.Λw) \ X) + f.σ² * I

        # Check that logpdf agrees between distributions and BLR.
        @test logpdf(f(X), y) ≈ logpdf(MvNormal(m, Σ), y)
    end
    @testset "posterior" begin
        # conditioning with low noise yields low uncetainty and predictions close to obs.
        # multiple conditioning yields results close to single conditioning on everything.
    end
end
