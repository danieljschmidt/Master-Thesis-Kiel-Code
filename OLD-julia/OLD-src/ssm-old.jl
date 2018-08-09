"""
for example if yt = [NaN, 0.5, 0.7], then St = [0 1 0; 0 0 1]
note that St is a sparse matrix!
"""
function get_selection_matrix(yt)
    n = length(yt)
    observed = .~isnan.(yt)
    St = spzeros(sum(observed), n)
    sum_observed = 0
    for i=1:n
        if observed[i]
            sum_observed = sum_observed + 1
            St[sum_observed, i] = 1
        end
    end
    return St
end

"""
State space model of the type
y_t = μ + B x_t
x_t = A x_{t-1} + R ϵ_t
TODO: include initialization??
"""
type StateSpaceModel
    μ::Array{Float64,1}
    A::Array{Float64,2}
    B::Array{Float64,2}
    Σ::Array{Float64,2}
    R::Array{Float64,2}
end

function simulate(
    ssm::StateSpaceModel,
    xf1::Array{Float64,1},
    Σxf1::Array{Float64, 2},
    T
    )
    x1 = rand(MvNormal(Vector(xf1), Matrix(Σxf1)))
    return simulate(ssm, x1, T)
end

function simulate(
    ssm::StateSpaceModel,
    x1::Array{Float64,1},
    T
    )
    μ, A, B, Σ, R = ssm.μ, ssm.A, ssm.B, ssm.Σ, ssm.R
    n, k = size(B) # include in type StateSpaceModel
    k1, _ = size(Σ)
    x = zeros(T, k)
    ϵ = rand(MvNormal(zeros(k1), Matrix(Σ)),T-1)'     # T - 1 ????
    xc = x1
    x[1,:] = xc
    for t=2:T
        xc = A*xc + R*ϵ[t-1,:]
        x[t,:] = xc # this helps to decrease running time (however, it does not seem to help for fast state smoothing)
    end
    y = ones(T,1)*μ'+x*B'
    return x, y

end

function fast_state_smoothing(
    y::Array{Float64,2},
    ssm::StateSpaceModel,
    xf1::Array{Float64,1},
    Σxf1::Array{Float64,2},
    )

    μ, A, B, Σ, R = ssm.μ, ssm.A, ssm.B, ssm.Σ, ssm.R
    V = R*Σ*R'

    T, _   = size(y)
    n, k = size(B)

    xft  = xf1
    Σxft = Σxf1

    q = zeros(T, k)
    L = zeros(T, k, k)

    for t=1:T

        yt = y[t,:]
        St = get_selection_matrix(yt)
        yt = Vector(St*sparse(yt)) # otherwise NaN because 0 * NaN = NaN

        Bt  = St*B
        μt  = St*μ
        Δyt = yt-μt-Bt*xft
        Ft  = Bt*Σxft*Bt' #+ St*W*St'
        Mt  = Bt'/Ft
        Kt  = A*Σxft*Mt

        Lt  = A-Kt*Bt
        qt = Mt*Δyt

        L[t,:,:] = Lt
        q[t,:] = qt

        xft  = A*xft + Kt*Δyt
        Σxft = A*Σxft*Lt'+ V

    end

    r = zeros(T+1, k) # index shifted by +1 !
    # i.e. also r[T+1,:] = 0

    for t=T:-1:1
        Lt = L[t,:,:]
        r[t,:] = q[t,:] + Lt'*r[t+1,:]
    end

    xs      = zeros(T, k)
    xs[1,:] = xf1 + Σxf1*r[1,:]

    for t=1:T-1
        xs[t+1,:] = A*xs[t,:] + V*r[t+1,:]
    end

    return xs

end

"""
TODO: ϵs instead of vs
TODO: problem with Σxs (see jupyter notebook)
"""
function smooth_states_and_disturbances(
    y::Array{Float64,2},
    ssm::StateSpaceModel,
    xf1::Array{Float64,1},
    Σxf1::Array{Float64,2},
    )

    μ, A, B, Σ, R = ssm.μ, ssm.A, ssm.B, ssm.Σ, ssm.R
    V = R*Σ*R'

    T, _   = size(y)
    n, k = size(B)

    xf  = zeros(T, k)
    Σxf = zeros(T, k, k)
    xft  = xf1
    Σxft = Σxf1

    #dy = zeros(T, n) difficult because n changes if there are NaNs in y[t,:]
    #F  = zeros(T, n, n)
    #K  = zeros(T, k, n)
    #L  = zeros(T, k, k)

    for t=1:T

        xf[t,:]    = xft
        Σxf[t,:,:] = Σxft

        yt = y[t,:]
        St = get_selection_matrix(yt)
        yt = Vector(St*sparse(yt)) # otherwise NaN because 0 * NaN = NaNS

        Bt  = St*B
        μt  = St*μ
        Δyt = yt-μt-Bt*xft
        Ft  = Bt*Σxft*Bt' #+ St*W*St'
        Kt  = A*Σxft*Bt'/Ft
        Lt  = A-Kt*Bt

        xft  = A*xft + Kt*Δyt        # calculate E(x_{t+1}|y_{1:t})
        Σxft = A*Σxft*(A-Kt*Bt)' + V

    end

    xs  = zeros(T, k)
    Σxs = zeros(T, k, k)

    r = zeros(T+1, k)     # index shifted by +1
    N = zeros(T+1, k, k) # index shifted by +1

    #r[T+1] = 0
    #N[T+1] = 0

    for t=T:-1:1

        # ... repeated from above

        yt = y[t,:]
        St = get_selection_matrix(yt)
        yt = Vector(St*sparse(yt)) # otherwise NaN because 0 * NaN = NaNS

        Bt  = St*B
        μt  = St*μ
        Δyt = yt-μt-Bt*xft
        Ft  = Bt*Σxft*Bt' #+ St*W*St'
        Kt  = A*Σxft*Bt'/Ft
        Lt  = A-Kt*Bt

        # ... repeated from above

        r[t,:]     = Bt'/Ft*Δyt + Lt'*r[t+1,:]
        N[t,:,:]   = Bt'/Ft*Bt +Lt'*N[t+1,:,:]*Lt
        xs[t,:]    = xf[t,:] + Σxf[t,:,:]*r[t,:]
        Σxs[t,:,:] = Σxf[t,:,:] - Σxf[t,:,:]*N[t,:,:]*Σxf[t,:,:]

    end

    vs = zeros(T-1, k)      # only until T-1!!
    #ws = zeros(T, n)
    Σvs = zeros(T-1, k, k) # only until T-1!!
    #Σws = zeros(T, n, n)

    for t=1:T

        # TODO: is there still a problem?

        #ws[t,:] = W/F[t,:,:]*dy[t,:] - W*K[t,:,:]'*r[t+1,:] # index of r shifted by +1
        #Σws[t,:,:] = W - W/F[t,:,:]*W - W*K[t,:,:]'*N[t+1,:,:]*K[t,:,:]*W

        if t != T # only until T-1!!
            vs[t,:] = V*r[t+1,:] # index of r shifted by +1
            Σvs[t,:,:] = V - V*N[t+1,:,:]*V
        else
            break
        end

    end

    return xs, Σxs, vs, Σvs #, ws, Σws # xf[1:T,:], Σxf[1:T,:,:],

end

#=
function fast_state_smoothing(
    y::Array{Float64,2},
    μ::SparseVector{Float64,Int64},
    A::SparseMatrixCSC{Float64,Int64},
    B::SparseMatrixCSC{Float64,Int64},
    V::SparseMatrixCSC{Float64,Int64},
    W::SparseMatrixCSC{Float64,Int64},
    xf1::SparseVector{Float64,Int64},
    Σxf1::SparseMatrixCSC{Float64,Int64}
    )

    T, _   = size(y)
    n, k = size(B)

    xft  = xf1
    Σxft = Σxf1

    q = spzeros(T, k)
    L = spzeros(T, k*k)

    for t=1:T

        yt = y[t,:]
        St  = get_selection_matrix(yt)
        yt  = St*sparse(yt)

        Bt  = St*B
        μt  = St*μ
        Δyt = yt-μt-Bt*xft
        Ft  = Bt*Σxft*Bt' + St*W*St'
        Mt = sparse(Matrix(Bt')/Matrix(Ft))
        Kt  = A*Σxft*Mt

        Lt  = A-Kt*Bt
        qt = Mt*Δyt

        L[t,:] = vec(Lt)
        q[t,:] = qt

        xft  = A*xft + Kt*Δyt
        Σxft = A*Σxft*Lt'+ V

    end

    r = zeros(T+1, k) # index shifted by +1 !
    # i.e. also r[T+1,:] = 0

    for t=T:-1:1
        Lt = reshape(L[t,:],k,k)
        r[t,:] = q[t,:] + Lt'*r[t+1,:]
    end

    xs      = zeros(T, k)
    xs[1,:] = xf1 + Σxf1*r[1,:]

    for t=1:T-1
        xs[t+1,:] = A*xs[t,:] + V*r[t+1,:]
    end

    return xs

end
=#
