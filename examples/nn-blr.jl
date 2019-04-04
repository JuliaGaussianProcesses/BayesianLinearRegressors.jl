# Implementes a basis-function (Bayesian) Linear Regressor, where the basis functions are
# provided by an MLP whose parameters are learned via type-2 maximum likelihood
# i.e. maximising the log marginal likelihood of the parameters
# This can be thought of as an MLP where we perform Bayesian inference over only the last
# layer of the network.

# Exact inference is possible in the batch setting. A variational exact that enables mini-
# batching should appear at some point in the future.

# Import a load of things.
using LinearAlgebra, BayesianLinearRegressors, Zygote, Flux, Plots, ProgressMeter
using BayesianLinearRegressors: BayesianLinearRegressor, logpdf, posterior, marginals
using Statistics: mean, std

# Create an MLP that you might have seen in the 90s.
Dlat = 100;
W1, b1 = randn(Dlat, 1), randn(Dlat);
ϕ = Chain(
    x->reshape(x, 1, :),
    x->tanh.(W1 * x .+ b1),
)
logσ = [log(1)]


# Generate toy data.
Ntr = 1000;
x = collect(range(-5.0, 5.0; length=Ntr));
y = sin.(x) + 0.1 * randn(Ntr);

blr = BayesianLinearRegressor(zeros(Dlat), Matrix{Float64}(I, Dlat, Dlat))

function nn_blr_training_loop(x, y, pars, Nitr, opt)
    tr_loss = Vector{Float64}(undef, Nitr)
    p = ProgressMeter.Progress(Nitr)
    for itr  in 1:Nitr
        loss, back = Zygote.forward(Zygote.Params(pars)) do
            -logpdf(blr(ϕ(x), exp(2 * logσ[1])), y)
        end
        tr_loss[itr] = loss
        g = back(1.0)
        for par in pars
            Flux.Optimise.update!(opt, par, g[par])
        end
        showvalues = [(:itr, itr), (:nlml, loss), (:σ_ε, exp(logσ[1]))]
        ProgressMeter.next!(p; showvalues = showvalues)
    end
    return tr_loss
end

Nitr = 500;
pars = [W1, b1, logσ];
nlmls = nn_blr_training_loop(x, y, pars, Nitr, ADAM(1e-2, (0.9, 0.999)));


blr′ = posterior(blr(ϕ(x), exp(2 * logσ[1])), y);

xte = collect(range(-10, 10; length=1000));
ypr_te = marginals(blr′(ϕ(xte), exp(2 * logσ[1])));

# Plot the predicted function vs the ground-truth. Doesn't generalise well, unsurprisingly.
plot(xte, mean.(ypr_te); linecolor="blue",  label="pr", linewidth=2.0);
plot!(xte, mean.(ypr_te) .+ 3 .* std.(ypr_te); linecolor="blue",  label="");
plot!(xte, mean.(ypr_te) .- 3 .* std.(ypr_te); linecolor="blue",  label="");
scatter!(x, y; markercolor="red", label="y", markersize=0.1);
plot!(xte, sin.(xte); linecolor="red", label="sin");
plot!(xte, sin.(xte) .+ 0.3; linecolor="red", label="");
plot!(xte, sin.(xte) .- 0.3; linecolor="red", label="")

# Compute and display residuals
ypr = marginals(blr′(ϕ(x), exp(2 * logσ[1])));
ε = y .- mean.(ypr);
@show mean(ε), std(ε);
histogram(ε; bins=50)

# Compute and display training curve.
plot(nlmls)
