using Distributions
using BandedMatrices
using Gadfly

p  = [0.00609, 0.04775, 0.13057, 0.20674,  0.22715,  0.18842,  0.12047,  0.05591,  0.01575,   0.00115]
m  = [1.92677, 1.34744, 0.73504, 0.02266, -0.85173, -1.97278, -3.46788, -5.55246, -8.68384, -14.65000]
s² = [0.11265, 0.17788, 0.26768, 0.40611,  0.62699,  0.98583,  1.57469,  2.54498,  4.16591,   7.33342]

"""
y_t = \exp(h_t/2) \epsilon_t
h_{t+1} = \mu_ + \phi (h_t-\mu_) + \sigma \eta_t
where
\eta_t \sim \mathrm{Normal}(0,1)
\epsilon_t \sim \mathrm{Normal}(0,1)
"""
type StochVol
    μ::Float64
    ϕ::Float64
    σ²::Float64
end

function simulate(sv::StochVol, T)
    h = zeros(T)
    h[1] = sv.μ + sqrt(sv.σ²/(1-sv.ϕ^2))*randn()
    for t=2:T
        h[t] = sv.μ + sv.ϕ*(h[t-1]-sv.μ) + sqrt(sv.σ²)*randn()
    end
    y = exp.(h/2).*randn(T)
    return y, h
end

function draw_h(z, r, T, sv::StochVol)
    c = (z-m[r])/s²[r] + sv.μ*(1-sv.ϕ)/sv.σ²
    Ω = SymBandedMatrix{Float64}(Zeros(T,T),1)
    Ω.data[1,2:end] = sv.ϕ/sv.σ²
    Ω.data[2,:]     = 1/s2[r] + (1+sv.ϕ^2)/sv.σ²


    return h
end

function draw_h(z, h, T, sv::StochVol)


    return r
end

T = 500

sv = StochVol(1, 0.95, 0.25)

y, h_true = simulate(sv, T)

plot(
    layer(x=1:T, y= y,          Geom.line),
    layer(x=1:T, y= 2*exp.(h_true/2), Geom.line, Theme(default_color="orange")),
    layer(x=1:T, y=-2*exp.(h_true/2), Geom.line, Theme(default_color="orange")),
)

z = log.(y.^2)

r = rand(Categorical(p), T)

c = (z-m[r])./s²[r] + sv.μ*(1-sv.ϕ)/sv.σ²
Ω = SymBandedMatrix{Float64}(Zeros(T,T),1)
Ω.data[1,2:end] = sv.ϕ/sv.σ²
Ω.data[2,:]     = 1 ./s²[r] + (1+sv.ϕ^2)/sv.σ²
