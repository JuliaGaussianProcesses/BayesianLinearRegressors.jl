# Implementes a basis-function (Bayesian) Linear Regressor, where the basis functions are
# provided by an MLP whose parameters are learned via type-2 maximum likelihood
# i.e. maximising the log marginal likelihood of the parameters
# This can be thought of as an MLP where we perform Bayesian inference over only the last
# layer of the network.

# Exact inference is possible in the batch setting. A variational exact that enables mini-
# batching should appear at some point in the future.

# Import a load of things.
using LinearAlgebra, BayesianLinearRegressors, Zygote, Flux, Plots, ProgressMeter, Random
using BayesianLinearRegressors: BayesianLinearRegressor, logpdf, posterior, marginals
using Statistics: mean, std

# Create an MLP that you might have seen in the 90s.
Dlat = 50;
W1, b1 = randn(Dlat, 1), randn(Dlat);
ϕ = Chain(x -> reshape(x, 1, :), x -> tanh.(W1 * x .+ b1))

# Initialise the standard deviation of the observation noise. We will learn this.
logσ = [log(1)]

# Generate toy data.
Ntr = 1000;
x = collect(range(-5.0, 5.0; length=Ntr));
y = sin.(x) + 0.1 * randn(Ntr);

# Construct a simple BLR, with zero mean and identity covariance / precision prior.
blr = BayesianLinearRegressor(zeros(Dlat), Matrix{Float64}(I, Dlat, Dlat))

function nn_blr_training_loop(x, y, pars, Nitr, opt)
    tr_nlml = Vector{Float64}(undef, Nitr)
    p = ProgressMeter.Progress(Nitr)
    for itr in 1:Nitr
        nlml, back = Zygote.forward(Zygote.Params(pars)) do
            -logpdf(blr(ϕ(x), exp(2 * logσ[1])), y)
        end
        tr_nlml[itr] = nlml
        g = back(1.0)
        for par in pars
            Flux.Optimise.update!(opt, par, g[par])
        end
        showvalues = [(:itr, itr), (:nlml, nlml), (:σ_ε, exp(logσ[1]))]
        ProgressMeter.next!(p; showvalues=showvalues)
    end
    return tr_nlml
end

# Gather the parameters and perform 500 iterations of ADAM.
pars = [W1, b1, logσ];
nlmls = nn_blr_training_loop(x, y, pars, 500, ADAM(1e-2, (0.9, 0.999)));

# Compute the posterior BLR, given the optimised network and noise parameters.
blr′ = posterior(blr(ϕ(x), exp(2 * logσ[1])), y);

# Make posterior predictions.
xte = collect(range(-10, 10; length=1000));
ypr_te = marginals(blr′(ϕ(xte), exp(2 * logσ[1])));

# Generate samples from the posterior over the latent function. Note that the observation
# noise is small.
y_samples_te = rand(MersenneTwister(123456), blr′(ϕ(xte), eps()), 25);

# Plot the predicted function vs the ground-truth. Doesn't generalise well, unsurprisingly.
# Does do _really_ well inside the region containing the training data though, and is
# nicely callibrated.
plt = plot();

# Plot the true function with aleatoric uncertainty due to observation noise.
plot!(plt, xte, sin.(xte); linecolor="purple", linewidth=1.5, label="sin");
plot!(plt, xte, sin.(xte) .+ 0.3; linecolor="purple", linewidth=1.5, label="");
plot!(plt, xte, sin.(xte) .- 0.3; linecolor="purple", linewidth=1.5, label="")

# Visualise the posterior marginal uncertainty via 3σ error bars.
m_te, σ_te = mean.(ypr_te), std.(ypr_te)
plot!(
    plt,
    xte,
    [m_te m_te];
    label="",
    fillrange=[m_te .+ 3 .* σ_te, m_te .- 3 .* σ_te],
    linewidth=0,
    fillalpha=0.3,
    fillcolor="red",
);

# Visualise the posterior distribution over the latent function via samples.
plot!(
    plt, xte, y_samples_te[:, 1]; linecolor="blue", linewidth=0.1, label="Posterior Samples"
);
plot!(plt, xte, y_samples_te; linecolor="blue", linewidth=0.1, label="");

# Plot the data.
scatter!(plt, x, y; markercolor="red", label="y", markersize=0.1);
display(plt);

# Compute and display residuals
ypr = marginals(blr′(ϕ(x), exp(2 * logσ[1])));
ε = y .- mean.(ypr);
@show mean(ε), std(ε);
histogram(ε; bins=50)

# Compute and display training curve.
plot(nlmls)
