"""
Mixed frequency dynamic factor model
y_{i,t} = μ_i + λ_i f_t + u_{i,t}
f_t     = ϕ f_{t-1} + v_t
u_{i,t} = ψ_i u_{i,t-1} + w_{i,t}
where the first variable is assumed to be a quarterly variable such that
y_{1,t} = 3 μ_1 + 1/3 λ_1 f_t + 2/3 λ_1 f_{t-1} + λ_1 f_{t-2} + ...
    2/3 λ_1 f_{t-3} + λ_1 f_{t-4} + 1/3 u_{1,t} + 2/3 u_{1,t-1} + ...
    u_{1,t-2} + 2/3 u_{1,t-3} + 1/3 u_{1,t-4}
for every third time point and y_{1,t} = NaN else
TODO: allow for more than one quarterly variable at arbitrary indices
TODO: number of observed variables n as property
TODO: test whether factor dynamics and unobserved component dynamics are stationary
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
brings the Mixed frequency dynamic factor model into the form
y_t = μ + B x_t
x_t = A x_{t-1} + R ϵ_t
where x_t = [f_t, ..., f_{t-4}, u_{1,t}, ..., u_{1,t-4}, u_{2,t}, ... u_{n,t}]
TODO: sparse matrices as output or not ?
"""
function get_state_space_representation(dfm::MixedFreqDFM)

    n = length(dfm.λ)
    # better: n = dfm.n

    μ = [3*dfm.μ[1], dfm.μ[2:end]...]
    # note that μ_1 = 3 μ*_1 where μ* denotes the intercept in the dynamic factor model
    #μ = sparse(μ)

    A1   = [dfm.ϕ spzeros(1,4); speye(4,4) spzeros(4,1)]
    A2q  = [dfm.ψ[1] spzeros(1,4); speye(4,4) spzeros(4,1)]
    A2   = blkdiag(A2q, spdiagm(dfm.ψ[2:end]))
    A    = blkdiag(A1, A2)
    A    = Matrix(A)

    b1q  = dfm.λ[1]/3*[1 2 3 2 1]
    b1m  = reshape(dfm.λ[2:end],n-1,1)
    B1   = [sparse(b1q); sparse(b1m) spzeros(n-1,4)]
    b2q  = 1/3*[1 2 3 2 1]
    B2   = blkdiag(sparse(b2q), speye(n-1))
    B    = [B1 B2]
    B    = Matrix(B)

    R = [1 spzeros(1,n); spzeros(4,n+1); 0 1 spzeros(1,n-1); spzeros(4,n+1);
        spzeros(n-1,2) speye(n-1,n-1)]
    R = Matrix(R)

    Σ = diagm([dfm.σ2v, dfm.σ2w...])
    #Σ = spdiagm([dfm.σ2v, dfm.σ2w...])

    return StateSpaceModel(μ, A, B, Σ, R)

end

"""
calculates E(x_1) and Var(x_1) for the initialization of the Kalman filter
to this end, it uses the unconditional moments E(x_t), Var(x_t)
note that for an AR(1) process f_t = ϕ f_{t-1} + v_t the autocovariances are
Cov(f_t, f_{t-j}) = ϕ^j σ2_v/(1-ϕ^2)
"""
function get_Kalman_filter_initialization(dfm::MixedFreqDFM)

    n = length(dfm.λ)
    # better: n = dfm.n

    xf1 = zeros(10+n-1)

    part1 = spzeros(5,5)
    for i=1:5
        for j=1:5
            part1[i,j] = dfm.ϕ^(abs(i-j))*dfm.σ2v/(1-dfm.ϕ^2)
        end
    end

    part2 = zeros(5,5)
    for i=1:5
        for j=1:5
            part2[i,j] = dfm.ψ[1]^(abs(i-j))*dfm.σ2w[1]/(1-dfm.ψ[1]^2)
        end
    end

    Σxf1 = zeros(10+n-1,10+n-1)
    Σxf1[1:5,1:5] = part1
    Σxf1[6:10,6:10] = part2
    for i=1:n-1
        Σxf1[10+i,10+i] = dfm.σ2w[1+i]/(1-dfm.ψ[1+i]^2)
    end

    return xf1, Σxf1

end
