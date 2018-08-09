"""
TODO: documentation
"""
type StateSpaceModel2
    μ::Array{Float64,1}
    μ1::Array{Float64,1}
    A::Array{Float64,2}
    B::Array{Float64,2}
    B1::Array{Float64,2}
    Σϵ::Array{Float64,2}
    Rϵ::Array{Float64,2}
    Ση::Array{Float64,2}
    Ση1::Array{Float64,2}
    Rη::Array{Float64,2}
    μx1::Array{Float64,1}
    Σx1::Array{Float64,2}
end

function fast_state_smoothing(
    y::Array{Float64,2},
    ssm::StateSpaceModel2
    )

    A, Σϵ, Rϵ, Rη = ssm.A, ssm.Σϵ, ssm.Rϵ, ssm.Rη
    μx1, Σx1 = ssm.μx1, ssm.Σx1

    V = Rϵ*Σϵ*Rϵ'

    T, _ = size(y)
    k, _ = size(A)
    n, _ = size(Rϵ)

    xft  = μx1
    Σxft = Σx1

    q = zeros(T, k)
    L = zeros(T, k, k)

    μ, B, Ση = ssm.μ1, ssm.B1, ssm.Ση1
    W = Rη*Ση*Rη'

    t = 1

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

    μ, B, Ση = ssm.μ, ssm.B, ssm.Ση
    W = Rη*Ση*Rη'

    for t=2:T

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


function simulation_smoothing_carter_kohn(
    y::Array{Float64,2},
    ssm::StateSpaceModel2
    )

    A, Σϵ, Rϵ, Rη = ssm.A, ssm.Σϵ, ssm.Rϵ, ssm.Rη
    μx1, Σx1 = ssm.μx1, ssm.Σx1

    V = Rϵ*Σϵ*Rϵ'

    T, _ = size(y)
    k, _ = size(A)
    n, _ = size(Rϵ)

    xft  = μx1
    Σxft = Σx1

    xf  = zeros(T, k)
    Σxf = zeros(T, k, k)

    μ, B, Ση = ssm.μ1, ssm.B1, ssm.Ση1
    W = Rη*Ση*Rη'

    t = 1

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

    μ, B, Ση = ssm.μ, ssm.B, ssm.Ση
    W = Rη*Ση*Rη'

    for t=2:T

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

    x = zeros(T, k)

    μt = xf[T,:]
    Σt = Σxf[T,:,:]

    xt    = μt + chol(Symmetric(Σt + 1e-10*eye(k)))'*randn(k,1) # rand(MvNormal(...)) does not work
    x[T,:] = xt

    for t=T-1:-1:1

        xft = xf[t,:]
        Σxft = Σxf[t,:,:]

        Δxt = xt - A*xft
        Ft  = A*Σxft*A' + V
        Mt = A'/Ft
        μt = xft + Σxft*Mt*Δxt
        Σt = Σxft - Σxft*Mt*A*Σxft

        xt = μt + chol(Symmetric(Σt + 1e-10*eye(k)))'*randn(k,1)
        x[t,:] = xt

    end

    return x

end
