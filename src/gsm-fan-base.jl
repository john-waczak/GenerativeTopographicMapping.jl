function fit_fan!(gsm::GSMBase, Nv, X; λe = 0.01, λw = 0.1, nepochs=100, tol=1e-3, nconverged=5, verbose=false, make_positive=false)
    # get the needed dimensions
    N,D = size(X)

    K = size(gsm.Z,1)

    M = size(gsm.Φ,2)

    prefac = 0.0
    G = Diagonal(sum(gsm.R, dims=2)[:])

    LnΠ = ones(size(gsm.R))
    for n ∈ axes(LnΠ,2)
        LnΠ[:,n] .= log.(gsm.πk)
    end

    Λ = Diagonal(vcat(λe * ones(Nv), λw * ones(M-Nv)))

    Q = 0.0
    Q_prev = 0.0

    llhs = Float64[]
    Qs = Float64[]
    nclose = 0
    converged = false

    for i in 1:nepochs

        # if desired, force weights to be positive.
        if make_positive
            gsm.W = max.(gsm.W, 0.0)
        end



        # EXPECTATION
        mul!(gsm.Ψ, gsm.W, gsm.Φ')                             # update latent node means
        pairwise!(sqeuclidean, gsm.Δ², gsm.Ψ, X', dims=2)    # update distance matrix
        gsm.Δ² .*= -(1/(2*gsm.β⁻¹))
        softmax!(gsm.R, gsm.Δ² .+ LnΠ, dims=1)

        G .= Diagonal(sum(gsm.R, dims=2)[:])

        # MAXIMIZATION

        # 1. Update the πk values
        # gsm.πk .= (1/N) .* sum(gsm.R, dims=2)
        gsm.πk .= max.((1/N) .* sum(gsm.R, dims=2), eps(eltype(gsm.πk)))
        for n ∈ axes(LnΠ,2)
            LnΠ[:,n] .= log.(gsm.πk)
        end

        # 2. update weight matrix

        # Assume a GSM with Linear and Nonlinear terms + No Bias
        gsm.W = ((gsm.Φ'*G*gsm.Φ + gsm.β⁻¹*Λ)\(gsm.Φ'*gsm.R*X))'

        # if desired, force weights to be positive.
        if make_positive
            gsm.W = max.(gsm.W, 0.0)
        end

        # 3. update precision β
        mul!(gsm.Ψ, gsm.W, gsm.Φ')                         # update means
        pairwise!(sqeuclidean, gsm.Δ², gsm.Ψ, X', dims=2)  # update distance matrix
        gsm.β⁻¹ = sum(gsm.R .* gsm.Δ²)/(N*D)                    # update variance


        # UPDATE LOG-LIKELIHOOD
        prefac = (N*D/2)*log(1/(2* gsm.β⁻¹* π))

        if i == 1
            l = max(prefac + sum(logsumexp(gsm.Δ² .* LnΠ, dims=1)), nextfloat(typemin(1.0)))

            Q = (N*D/2)*log(1/(2* gsm.β⁻¹* π)) + sum(gsm.R .* LnΠ) - sum(gsm.R .* gsm.Δ²)  + (length(D*(M-Nv))/2)*log(λw/(2π)) + (length(D*Nv)/2)*log(λe/(2π))  - (λe/2)*sum(gsm.W[:,1:Nv])  - (λe/2)*sum(gsm.W[:,Nv+1:end])

            push!(llhs, l)
            push!(Qs, Q)
        else
            Q_prev = Q

            l = max(prefac + sum(logsumexp(gsm.Δ² .* LnΠ, dims=1)), nextfloat(typemin(1.0)))

            Q = (N*D/2)*log(1/(2* gsm.β⁻¹* π)) + sum(gsm.R .* LnΠ) - sum(gsm.R .* gsm.Δ²)  + (length(D*(M-Nv))/2)*log(λw/(2π)) + (length(D*Nv)/2)*log(λe/(2π))  - (λe/2)*sum(gsm.W[:,1:Nv])  - (λe/2)*sum(gsm.W[:,Nv+1:end])

            push!(llhs, l)
            push!(Qs, Q)

            # check for convergence
            rel_diff = abs(Q - Q_prev)/min(abs(Q), abs(Q_prev))

            if rel_diff <= tol
                # increment the number of "close" difference
                nclose += 1
            end

            if nclose == nconverged
                converged = true
                break
            end
        end


        if verbose
            println("iter: $(i), Q=$(Qs[end]), log-likelihood = $(llhs[end])")
        end
    end

    AIC = 2*length(gsm.W) - 2*llhs[end]
    BIC = log(size(X,1))*length(gsm.W) - 2*llhs[end]

    return converged, Qs, llhs, AIC, BIC
end


