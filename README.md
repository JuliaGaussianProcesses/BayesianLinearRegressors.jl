# Bayesian Linear Regression in Julia

[![Build Status](https://github.com/willtebbutt/BayesianLinearRegressors.jl/workflows/CI/badge.svg)](https://github.com/willtebbutt/BayesianLinearRegressors.jl/actions)
[![Codecov](https://codecov.io/gh/willtebbutt/BayesianLinearRegressors.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/willtebbutt/BayesianLinearRegressors.jl)

This is a simple package that does one thing, Bayesian Linear Regression, in around 100 lines of code.

It _is_ actively maintained, but it might appear inactive as it's one of those packages which requires very little maintenance because it's very simple.

## Intended Use and Functionality

The interface sits at roughly the same level as that of [Distributions.jl](https://github.com/JuliaStats/Distributions.jl/). This means that while you won't find a scikit-learn-style `fit` function, you will find all of the primitives that you need to construct such a function to suit your particular problem. In particular, one can:

- Construct a `BayesianLinearRegressor` (BLR) object by providing a mean-vector and precision matrix for the weights of said regressor. This object represents a distribution over (linear) functions.
- Use a `BayesianLinearRegressor` as an `AbstractGP`, as it implements the primary AbstractGP API.
- Think of an instance of `BayesianLinearRegressor` as a very restricted GP, where the time complexity of inference scales linearly in the number of observations `N`.
- Draw function samples from a `BayesianLinearRegressor` using `rand`.
- Construct a `BasisFunctionRegressor` object which is a thin wrapper around a `BayesianLinearRegressor` to allow a non-linear feature mapping `ϕ` to act on the input.

## Conventions

`BayesianLinearRegressors` is consistent with `AbstractGPs`.
Consequently, a `BayesianLinearRegressor` in `D` dimensions can work with the following input types:
1. `ColVecs` -- a wrapper around an `D x N` matrix of `Real`s saying that each column should be interpreted as an input.
2. `RowVecs`s -- a wrapper around an `N x D` matrix of `Real`s, saying that each row should be interpreted as an input.
3. `Matrix{<:Real}` -- must be `D x N`. Prefer using `ColVecs` or `RowVecs` for the sake of being explicit.

Consult the `Design` section of the [KernelFunctions.jl](https://juliagaussianprocesses.github.io/KernelFunctions.jl/dev/design/) docs for more info on these conventions.

Outputs for a BayesianLinearRegressor should be an `AbstractVector{<:Real}` of length `N`.

## Example Usage


```julia
# Install the packages if you don't already have them installed
] add AbstractGPs, BayesianLinearRegressors LinearAlgebra Random Plots Zygote
using AbstractGPs, BayesianLinearRegressors, LinearAlgebra, Random, Plots, Zygote

# Fix seed for re-producibility.
rng = MersenneTwister(123456)

# Construct a BayesianLinearRegressor prior over linear functions of `X`.
mw, Λw = zeros(2), Diagonal(ones(2))
f = BayesianLinearRegressor(mw, Λw)

# Index into the regressor and assume heterscedastic observation noise `Σ_noise`.
N = 10
X = ColVecs(collect(hcat(collect(range(-5.0, 5.0, length=N)), ones(N))'))
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
X_plt = ColVecs(hcat(collect(range(-6.0, 6.0, length=N_plt)), ones(N_plt))')

# Compute some posterior marginal statisics.
normals = marginals(f′(X_plt, eps()))
m′X_plt = mean.(normals)
σ′X_plt = std.(normals)

# Plot the posterior. This uses the default AbstractGPs plotting recipes.
posterior_plot = plot();
plot!(posterior_plot, X_plt.X[1, :], f′(X_plt, eps()); color=:blue, ribbon_scale=3);
sampleplot!(posterior_plot, X_plt.X[1, :], f′(X_plt, eps()); color=:blue, samples=10);
scatter!(posterior_plot, X.X[1, :], y; # Observations.
    markercolor=:red,
    markershape=:circle,
    markerstrokewidth=0.0,
    markersize=4,
    markeralpha=0.7,
    label="",
);
display(posterior_plot);
```

## Basis Function Regression

Any instance of a `BayesianLinearRegressor` can be replaced by a `BasisFunctionRegressor` (BFR). A `BasisFunctionRegressor` is a thin wrapper around a `BayesianLinearRegressor`, but includes a potentially non-linear feature mapping `ϕ` which is applied to the input before it is passed to the underlying BLR. It is essentially defined as `bfr(X) = blr(ϕ(X))`.

``` julia
using AbstractGPs, BayesianLinearRegressors, LinearAlgebra

X = RowVecs(hcat(range(-1.0, 1.0, length=5)))
blr = BayesianLinearRegressor(zeros(2), Diagonal(ones(2)))

# N.B. ϕ must accept one of the allowed input types and
# must return the same type (in this case RowVecs)
ϕ(x::RowVecs) = RowVecs(hcat(ones(length(x)), prod.(x)))

bfr = BasisFunctionRegressor(blr, ϕ)

# These are equivalent
var(bfr(X)) == var(blr(ϕ(X)))
```

## Up For Grabs

- Scikit-learn style interface: it wouldn't be too hard to implement a scikit-learn - style interface to handle basic regression tasks, so please feel free to make a PR that implements this.
- Monte Carlo VI (MCVI): i.e. variational inference using the reparametrisation trick. This could be very useful when working with large data sets and applying big non-linear transformations, such as neural networks, to the inputs as it would enable mini-batching. I would envisage at least supporting both a dense approximate posterior covariance and diagonal (i.e. mean-field), where the former is for small-moderate dimensionalities and the latter for very high-dimensional problems.

## Bugs, Issues, and PRs

Please do report any bugs you find by raising an issue. Please also feel free to raise PRs, especially if for one of the above `Up For Grabs` items. Raise an issue to discuss the extension in detail before opening a PR if you prefer, though.


## Related Work

[BayesianLinearRegression.jl](https://github.com/cscherrer/BayesianLinearRegression.jl) is closely related, but appears to be a WIP and hasn't been touched in around a year or so (as of 27-03-2019).
