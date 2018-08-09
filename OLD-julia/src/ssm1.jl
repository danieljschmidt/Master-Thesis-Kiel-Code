using Distributions

"""
TODO: documentation
"""
type StateSpaceModel
    μ::Array{Float64,1}
    A::Array{Float64,2}
    B::Array{Float64,2}
    Σϵ::Array{Float64,2}
    Rϵ::Array{Float64,2}
    Ση::Array{Float64,2}
    Rη::Array{Float64,2}
    μx1::Array{Float64,1}
    Σx1::Array{Float64,2}
end

function simulate(
    ssm::StateSpaceModel,
    T
    )
    n, k = size(ssm.B)
    n1, _ = size(ssm.Ση)
    k1, _ = size(ssm.Σϵ)
    ϵ = rand(MvNormal(zeros(k1), Matrix(ssm.Σϵ)),T-1)'
    η = rand(MvNormal(zeros(n1), Matrix(ssm.Ση)),T)'
    x = zeros(T, k)
    xc = rand(MvNormal(Vector(ssm.μx1), Matrix(ssm.Σx1)))
    x[1,:] = xc
    for t=2:T
        xc = ssm.A*xc + ssm.Rϵ*ϵ[t-1,:]
        x[t,:] = xc # this helps to decrease running time (however, it does not seem to help for fast state smoothing)
    end
    y = ones(T,1)*ssm.μ'+x*ssm.B'+η*ssm.Rη'
    return x, y
end

function fast_state_smoothing(
    y::Array{Float64,2},
    ssm::StateSpaceModel
    )

    μ, A, B = ssm.μ, ssm.A, ssm.B
    Σϵ, Rϵ, Ση, Rη = ssm.Σϵ, ssm.Rϵ, ssm.Ση, ssm.Rη
    μx1, Σx1 = ssm.μx1, ssm.Σx1

    V = Rϵ*Σϵ*Rϵ'
    W = Rη*Ση*Rη'

    T, _ = size(y)
    n, k = size(B)

    xft  = μx1
    Σxft = Σx1

    q = zeros(T, k)
    L = zeros(T, k, k)

    for t=1:T

        yt = y[t,:]
        st = .~isnan.(yt)
        yt = yt[st]

        Bt  = B[st,:]
        μt  = μ[st]
        Wt = W[st, st]

        Δyt = yt-μt-Bt*xft
        Ft  = Bt*Σxft*Bt' + Wt
        Mt  = Bt'/Ft             # computationally most expensive part for large n
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
    xs[1,:] = μx1 + Σx1*r[1,:]

    for t=1:T-1
        xs[t+1,:] = A*xs[t,:] + V*r[t+1,:]
    end

    return xs

end

#TODO: sth is wrong with this function!

#=
function simulation_smoothing_carter_kohn(
    y::Array{Float64,2},
    ssm::StateSpaceModel
    )

    μ, A, B = ssm.μ, ssm.A, ssm.B
    Σϵ, Rϵ, Ση, Rη = ssm.Σϵ, ssm.Rϵ, ssm.Ση, ssm.Rη
    μx1, Σx1 = ssm.μx1, ssm.Σx1

    V = Rϵ*Σϵ*Rϵ'
    W = Rη*Ση*Rη'

    T, _ = size(y)
    n, k = size(B)

    xft  = μx1
    Σxft = Σx1

    xf  = zeros(T, k)
    Σxf = zeros(T, k, k)

    for t=1:T

        yt = y[t,:]
        st = .~isnan.(yt)
        yt = yt[st]

        Bt  = B[st,:]
        μt  = μ[st]
        Wt = W[st, st]

        Δyt = yt-μt-Bt*xft
        Ft  = Bt*Σxft*Bt' + Wt
        Mt = Bt'/Ft

        xft  = xft + Σxft*Mt*Δyt
        Σxft = Σxft - Σxft*Mt*Bt*Σxft

        xf[t,:] = xft
        Σxf[t,:,:] = Σxft

        xft  = A*xft
        Σxft = A*Σxft*A' + V

    end

    #return xf, Σxf

    x = zeros(T, k)

    μt = xf[T,:]
    Σt = Σxf[T,:,:]

    xt    = μt + Matrix(cholfact(Symmetric(Σt + 1e-13*eye(k))))*randn(k,1) # rand(MvNormal(...)) does not work
    x[T,:] = xt

    for t=T-1:-1:1

        xft = xf[t,:]
        Σxft = Σxf[t,:,:]

        Δxt = xt - A*xft
        Ft  = A*Σxft*A' + V
        Mt = A'/Ft
        μt = xft + Σxft*Mt*Δxt
        Σt = Σxft - Σxft*Mt*A*Σxft

        xt = μt + Matrix(cholfact(Symmetric(Σt + 1e-12*eye(k))))*randn(k,1)
        x[t,:] = xt

    end

    return x

end

=#
