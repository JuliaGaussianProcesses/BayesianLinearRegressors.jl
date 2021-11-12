# Bayesian Linear Regression in Julia

[![Build Status](https://github.com/willtebbutt/BayesianLinearRegressors.jl/workflows/CI/badge.svg)](https://github.com/willtebbutt/BayesianLinearRegressors.jl/actions)
[![Codecov](https://codecov.io/gh/willtebbutt/BayesianLinearRegressors.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/willtebbutt/BayesianLinearRegressors.jl)

This is a simple package that does one thing, Bayesian Linear Regression, in around 100 lines of code.

It _is_ actively maintained, but it might appear inactive as it's one of those packages which requires very little maintenance because it's very simple.

## Intended Use and Functionality

The interface sits at roughly the same level as that of [Distributions.jl](https://github.com/JuliaStats/Distributions.jl/). This means that while you won't find a scikit-learn-style `fit` function, you will find all of the primitives that you need to construct such a function to suit your particular problem. In particular, one can:

- Construct a `BayesianLinearRegressor` (BLR) object by providing a mean-vector and precision matrix for the weights of said regressor. This object represents a distribution over (linear) functions.
- A `BayesianLinearRegressor` is a `AbstractGP`, and implements the primary AbstractGP API.
- Think of an instance of `BayesianLinearRegressor` as a very restricted GP, where the time complexity of inference scales linearly in the number of observations `N`.

## Conventions

A `BayesianLinearRegressor` in `D` dimensions works with data where:
- inputs `X` should be a `D x N` matrix of `Real`s where each column is from one data point.
- outputs `y` should be an `N`-vector of `Real`s, where each element is from one data point.

## Example Usage


```julia
# Install the packages if you don't already have them installed
] add AbstractGPs, BayesianLinearRegressors LinearAlgebra Random Optim Plots Zygote
using AbstractGPs, BayesianLinearRegressors, LinearAlgebra, Random, Optim, Plots, Zygote

# Fix seed for re-producibility.
rng = MersenneTwister(123456)

# Construct a BayesianLinearRegressor prior over linear functions of `X`.
mw, Λw = zeros(2), Diagonal(ones(2))
f = BayesianLinearRegressor(mw, Λw)

# Index into the regressor and assume heterscedastic observation noise `Σ_noise`.
N = 10
X = collect(hcat(collect(range(-5.0, 5.0, length=N)), ones(N))')
Σ_noise = Diagonal(exp.(randn(N)))
fX = f(X, Σ_noise)

# Generate some toy data by sampling from the prior.
y = rand(rng, fX)

# Compute the adjoint of `rand` w.r.t. everything given random sensitivities of y′.
_, back_rand = Zygote.pullback(
    (X, Σ_noise, mw, Λw)->rand(rng, BayesianLinearRegressor(mw, Λw)(X, Σ_noise), 5),
    X, Σ_noise, mw, Λw,
)
back_rand(randn(N, 5))

# Compute the `logpdf`. Read as `the log probability of observing `y` at `X` under `f`, and
# Gaussian observation noise with zero-mean and covariance `Σ_noise`.
logpdf(fX, y)

# Compute the gradient of the `logpdf` w.r.t. everything.
Zygote.gradient(
    (X, Σ_noise, y, mw, Λw)->logpdf(BayesianLinearRegressor(mw, Λw)(X, Σ_noise), y),
    X, Σ_noise, y, mw, Λw,
)

# Perform posterior inference. Note that `f′` has the same type as `f`.
f′ = posterior(fX, y)

# Compute `logpdf` of the observations under the posterior predictive.
logpdf(f′(X, Σ_noise), y)

# Sample from the posterior predictive distribution.
N_plt = 1000
X_plt = hcat(collect(range(-6.0, 6.0, length=N_plt)), ones(N_plt))'
f′X_plt = rand(rng, f′(X_plt, eps()), 100) # Samples with machine-epsilon noise for stability

# Compute some posterior marginal statisics.
normals = marginals(f′(X_plt, eps()))
m′X_plt = mean.(normals)
σ′X_plt = std.(normals)

# Plot the posterior. This uses the default AbstractGPs plotting recipes.
posterior_plot = plot();
plot!(posterior_plot, X_plt[1, :], f′(X_plt, eps()); color=:blue, ribbon_scale=3);
sampleplot!(posterior_plot, X_plt[1, :], f′(X_plt, eps()); color=:blue, samples=10);
scatter!(posterior_plot, X[1, :], y; # Observations.
    markercolor=:red,
    markershape=:circle,
    markerstrokewidth=0.0,
    markersize=4,
    markeralpha=0.7,
    label="",
);
display(posterior_plot);
```


## Up For Grabs

- Scikit-learn style interface: it wouldn't be too hard to implement a scikit-learn - style interface to handle basic regression tasks, so please feel free to make a PR that implements this.
- Monte Carlo VI (MCVI): i.e. variational inference using the reparametrisation trick. This could be very useful when working with large data sets and applying big non-linear transformations, such as neural networks, to the inputs as it would enable mini-batching. I would envisage at least supporting both a dense approximate posterior covariance and diagonal (i.e. mean-field), where the latter is for small-moderate dimensionalities and the latter for very high-dimensional problems.

## Bugs, Issues, and PRs

Please do report any bugs you find by raising an issue. Please also feel free to raise PRs, especially if for one of the above `Up For Grabs` items. Raise an issue to discuss the extension in detail before opening a PR if you prefer, though.


## Related Work

[BayesianLinearRegression.jl](https://github.com/cscherrer/BayesianLinearRegression.jl) is closely related, but appears to be a WIP and hasn't been touched in around a year or so (as of 27-03-2019).
