type StateSpaceModel
    μ::SparseVector{Float64,Int64}
    A::SparseMatrixCSC{Float64,Int64}
    B::SparseMatrixCSC{Float64,Int64}
    Σϵ::SparseMatrixCSC{Float64,Int64}
    Rϵ::SparseMatrixCSC{Float64,Int64}
    Ση::SparseMatrixCSC{Float64,Int64}
    Rη::SparseMatrixCSC{Float64,Int64}
    μx1::SparseVector{Float64,Int64}
    Σx1::SparseMatrixCSC{Float64,Int64}
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
        Δyt = yt-μt-Bt*xft
        Ft  = Bt*Σxft*Bt' + W[st,st]
        Mt  = Matrix(Bt')/Ft
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
