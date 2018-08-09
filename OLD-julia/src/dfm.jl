"""
Mixed frequency dynamic (one-)factor model with AR(1) dynamics in factor and idiosyncratic components
y_{i,t} = μ_i + λ_i f_t + u_{i,t}
f_t     = ϕ f_{t-1} + v_t
u_{i,t} = ψ_i u_{i,t-1} + w_{i,t}
where the first variable is assumed to be a quarterly variable such that
y_{1,t} = 3 μ_1 + 1/3 λ_1 f_t + 2/3 λ_1 f_{t-1} + λ_1 f_{t-2} + ...
    2/3 λ_1 f_{t-3} + λ_1 f_{t-4} + 1/3 u_{1,t} + 2/3 u_{1,t-1} + ...
    u_{1,t-2} + 2/3 u_{1,t-3} + 1/3 u_{1,t-4}
for every third time point and y_{1,t} = NaN else
TODO:
- allow for more than one quarterly variable at arbitrary indices
- allow for AR(p) dynamics
- more than one factor
- number of observed variables n as property
- test whether dimensions make sense
- test whether factor dynamics and unobserved component dynamics are stationary
"""
type MixedFreqDFM
    μ::Array{Float64,1}
    λ::Array{Float64,1}
    ϕ::Float64
    σ2v::Float64
    ψ::Array{Float64,1}
    σ2w::Array{Float64,1}
    #quarterly::Array{Bool,1} # simply assume [1, 0, ... 0]
end

"""
Function that transforms a MixedFreqDFM into a StateSpaceModel with state vector
x_t = [f_t ... f_{t-4} u_{1,t} ... u_{1,t-4} u_{2,t} ... u_{n,t}]
"""
function get_state_space_representation(dfm::MixedFreqDFM)

    n = length(dfm.λ)

    μ = [3*dfm.μ[1], dfm.μ[2:end]...]

    A1   = [dfm.ϕ zeros(1,4); eye(4,4) zeros(4,1)]
    A2q  = [dfm.ψ[1] zeros(1,4); eye(4,4) zeros(4,1)]
    A2   = blkdiag(sparse(A2q), spdiagm(dfm.ψ[2:end]))
    A    = blkdiag(sparse(A1), A2)
    A    = Matrix(A)
    # blkdiag does only work with sparse matrices

    b1q  = dfm.λ[1]/3*[1 2 3 2 1]
    b1m  = reshape(dfm.λ[2:end],n-1,1)
    B1   = [sparse(b1q); sparse(b1m) spzeros(n-1,4)]
    b2q  = 1/3*[1 2 3 2 1]
    B2   = blkdiag(sparse(b2q), speye(n-1))
    B    = [B1 B2]
    B    = Matrix(B)

    Rϵ = [1 zeros(1,n); zeros(4,n+1); 0 1 zeros(1,n-1); zeros(4,n+1);
        zeros(n-1,2) eye(n-1,n-1)]

    Σϵ = diagm([dfm.σ2v, dfm.σ2w...])

    Rη = zeros(n,1)

    Ση = reshape([1.],1,1)
    # η does not exit in this state space model which is why I need this ugly hack

    μx1 = zeros(10+n-1)

    part1 = [dfm.ϕ^(abs(i-j)) for i=1:5, j=1:5]
    part1 = part1*dfm.σ2v/(1-dfm.ϕ^2)

    part2 = [dfm.ψ[1]^(abs(i-j)) for i=1:5, j=1:5]
    part2 = part2*dfm.σ2w[1]/(1-dfm.ψ[1]^2)

    Σx1 = zeros(10+n-1,10+n-1)
    Σx1[1:5,1:5] = part1
    Σx1[6:10,6:10] = part2
    for i=1:n-1
        Σx1[10+i,10+i] = dfm.σ2w[1+i]/(1-dfm.ψ[1+i]^2)
    end

    return StateSpaceModel(μ, A, B, Σϵ, Rϵ, Ση, Rη, μx1, Σx1)

end

"""
Function that transforms a MixedFreqDFM into a StateSpaceModel2 with state vector
x_t = [f_t ... f_{t-4} u_{1,t} ... u_{1,t-4}]
"""
function get_state_space_representation2(dfm::MixedFreqDFM)

    n = length(dfm.λ)

    μ  = [3*dfm.μ[1], (1-dfm.ψ[2:end]).*dfm.μ[2:end]...]
    μ1 = [3*dfm.μ[1], dfm.μ[2:end]...]

    A1   = [dfm.ϕ zeros(1,4); eye(4,4) zeros(4,1)]
    A2q  = [dfm.ψ[1] zeros(1,4); eye(4,4) zeros(4,1)]
    A   = blkdiag(sparse(A1), sparse(A2q))
    # blkdiag does only work with sparse matrices

    b1q  = dfm.λ[1]/3*[1 2 3 2 1]
    b1m  = reshape(dfm.λ[2:end],n-1,1)
    b2q  = 1/3*[1 2 3 2 1]
    B    = [b1q b2q; b1m -dfm.ψ[2:end].*b1m zeros(n-1, 3+5)]
    B1   = [b1q b2q; b1m zeros(n-1, 4+5)]

    Rϵ = [1 0; zeros(4,2); 0 1; zeros(4,2)]

    Σϵ = diagm([dfm.σ2v, dfm.σ2w[1]])

    Rη = [zeros(1,n-1); eye(n-1)]

    Ση  = diagm(dfm.σ2w[2:end])
    Ση1 = diagm((1-dfm.ψ[2:end].^2).*dfm.σ2w[2:end])

    μx1 = zeros(10)

    part1 = [dfm.ϕ^(abs(i-j)) for i=1:5, j=1:5]
    part1 = part1*dfm.σ2v/(1-dfm.ϕ^2)

    part2 = [dfm.ψ[1]^(abs(i-j)) for i=1:5, j=1:5]
    part2 = part2*dfm.σ2w[1]/(1-dfm.ψ[1]^2)

    Σx1 = zeros(10,10)
    Σx1[1:5,1:5] = part1
    Σx1[6:10,6:10] = part2

    return StateSpaceModel2(μ, μ1, A, B, B1, Σϵ, Rϵ, Ση, Ση1, Rη, μx1, Σx1)

end
