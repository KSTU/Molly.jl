export KarrKonowalow

function kk_zero_shortcut(atom_i, atom_j)
    return (iszero_value(atom_i.α) || iszero_value(atom_j.α)) &&
           (iszero_value(atom_i.rₘ) || iszero_value(atom_j.rₘ))
end

function α_mixing(atom_i, atom_j)
    return sqrt(atom_i.α * atom_j.α)
end

function rm_mixing(atom_i, atom_j)
    return (atom_i.rₘ + atom_j.rₘ) / 2.0
end

function ε_mixing(atom_i, atom_j)
    return sqrt(atom_i.ε * atom_j.ε)
end

@doc raw"""
    KarrKonowalow(; cutoff, use_neighbors, shortcut, \alpha, r\_m,
               \varepsilon, weight_special)

The Buckingham interaction between two atoms.

The potential energy is defined as
```math
V(r_{ij}) = 
```
and the force on each atom by
```math
\vec{F}_i = 
```
The parameters are derived from the atom parameters according to
```math
\begin{aligned}
\alpha_{ij} &= \sqrt{\alpha_{ii} \alpha_{jj}} \\
{r\_m}_{ij} &= \frac{{r\_m}_{ii} + {r\_m}_{jj}}{2} \\
\varepsilon_{ij} &= \sqrt{\varepsilon_{ii} \varepsilon_{jj}}
\end{aligned}
```
so atoms that use this interaction should have fields `A`, `B` and `C` available.
"""
@kwdef struct KarrKonowalow{C, S, A, B, M, W}
    cutoff::C = NoCutoff()
    use_neighbors::Bool = false
    shortcut::S = kk_zero_shortcut
    α::A = α_mixing
    rₘ::B = rm_mixing
    ε::M = ε_mixing
    weight_special::W = 1
end

use_neighbors(inter::KarrKonowalow) = inter.use_neighbors

function Base.zero(b::KarrKonowalow{C, W}) where {C, W}
    return KarrKonowalow(b.cutoff, b.use_neighbors, b.shortcut, b.α,
                      b.rₘ, b.ε, zero(W))
end

function Base.:+(b1::KarrKonowalow, b2::KarrKonowalow)
    return KarrKonowalow(b1.cutoff, b1.use_neighbors, b1.shortcut, b1.α, b1.rₘ,
                      b1.ε, b1.weight_special + b2.weight_special)
end

@inline function force(inter::KarrKonowalow,
                            dr,
                            atom_i,
                            atom_j,
                            force_units=u"kJ * mol^-1 * nm^-1",
                            special=false,
                            args...)
    if inter.shortcut(atom_i, atom_j)
        return ustrip.(zero(dr)) * force_units
    end
    α = inter.α(atom_i, atom_j)
    rₘ = inter.rₘ(atom_i, atom_j)
    ε = inter.ε(atom_i, atom_j)

    cutoff = inter.cutoff
    r2 = sum(abs2, dr)
    params = (α, rₘ, ε)

    f = force_divr_with_cutoff(inter, r2, params, cutoff, force_units)
    if special
        return f * dr * inter.weight_special
    else
        return f * dr
    end
end

function force_divr(::KarrKonowalow, r2, invr2, (α, rₘ, ε))
    r = sqrt(r2)
    #println("r ", r, " rm ", rₘ, " alpha ", α)
    expvar = exp((1-r/rₘ)*α)
    #println("expvar ", expvar)
    return (6*rₘ^6*(α+6)*((6*expvar)/(α+6)-1)*ε)/(r^8*α)+(6*rₘ^5*expvar*ε)/r^7
end

@inline function potential_energy(inter::KarrKonowalow,
                                    dr,
                                    atom_i,
                                    atom_j,
                                    energy_units=u"kJ * mol^-1",
                                    special=false,
                                    args...)
    if inter.shortcut(atom_i, atom_j)
        return ustrip(zero(dr[1])) * energy_units
    end
    α = inter.α(atom_i, atom_j)
    rₘ = inter.rₘ(atom_i, atom_j)
    ε = inter.ε(atom_i, atom_j)

    cutoff = inter.cutoff
    r2 = sum(abs2, dr)
    params = (α, rₘ, ε)

    pe = potential_with_cutoff(inter, r2, params, cutoff, energy_units)
    if special
        return pe * inter.weight_special
    else
        return pe
    end
end

function potential(::KarrKonowalow, r2, invr2, (α, rₘ, ε))
    r = sqrt(r2)
    return ε / α * (α+6) * (rₘ / r)^6 * (6 / (α + 6) * exp(α * (1 - r/rₘ)) - 1)
end