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

# simple nonlinear mapping
ϕ(x::RowVecs) = RowVecs(hcat(ones(length(x)), prod.(x)))
ϕ(x::ColVecs) = ColVecs(hcat(ones(length(x)), prod.(x))')
ϕ(x::AbstractMatrix) = ϕ(ColVecs(x)).X
