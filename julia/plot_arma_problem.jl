using CSV
using Gadfly
using Colors

mcmc     = CSV.read("..//data_plots//data_arma_problem_mcmc.csv", datarow=1, header=["ψ", "σ²"])
true_par = CSV.read("..//data_plots//data_arma_problem_true.csv", datarow=1, header=["ψ", "σ²"])

psi_true     = true_par[:ψ][1]
sigma2w_true = true_par[:σ²][1]

function var_xi(psi)
    return 1/9*(19 + 32*psi + 20*psi^2 + 8*psi^3 + 2*psi^4)/(1-psi^2)
end

psi_rng     = -0.9:0.01:0.9
n_rng       = length(psi_rng)
var_true    = var_xi(psi_true)
var_rng     = var_xi.(psi_rng)
sigma2w_rng = sigma2w_true*var_true./var_rng;

l1 = layer(
    x=[psi_true], y=[sigma2w_true],
    Geom.point(),
    Theme(default_color="red")
)

l2 = layer(
    x=psi_rng, y=sigma2w_rng,
    Geom.line(),
    Theme(default_color="red", line_style=:dash)
)

l3a = layer(
    x=mcmc[:ψ], y=mcmc[:σ²],
    Geom.density2d(levels = x->maximum(x)*0.5.^collect(1:2:8)),
)

l3b = layer(
    x=mcmc[:ψ], y=mcmc[:σ²],
    Geom.hexbin(xbincount=75, ybincount=50)
)

p1 = plot(
    l1, l2, l3a,
    Scale.color_none(),
    Coord.cartesian(xmin=-0.9, xmax=0.9),
    Guide.xlabel("ψ"), Guide.ylabel("σ²"),
    Guide.xticks(ticks=collect(-0.8:0.2:0.8))
)

p2 = plot(
    l1, l2, l3b,
    Scale.Scale.color_continuous(colormap=x->RGB((1-x)*9/10,(1-x)*9/10,(1-x)*9/10)),
    Coord.cartesian(xmin=-0.9, xmax=0.9, ymin=0, ymax=6),
    Guide.xlabel("ψ"), Guide.ylabel("σ²"),
    Guide.xticks(ticks=collect(-0.8:0.2:0.8))
)

draw(PDF("..//plots//plot_arma_problem_1.pdf", 15cm, 10cm), p1)

draw(PDF("..//plots//plot_arma_problem_2.pdf", 15cm, 10cm), p2)
