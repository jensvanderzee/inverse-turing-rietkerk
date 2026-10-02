"""
    sinusoidal_weekly_precip(annual_total; peak_week = 26.0, amplitude_fraction = 0.7,
                             num_weeks = 52) -> Vector{Float64}

Smooth seasonal forcing: weekly rates (mm/day) on a cosine around `annual_total/365`,
rescaled so the year delivers exactly `annual_total` mm at 7 days per week. Port of
`generate_weekly_precipitation`; `amplitude_fraction = 0` gives the uniform rain of
the synthetic spin-up.
"""
function sinusoidal_weekly_precip(annual_total::Real; peak_week::Real = 26.0,
                                  amplitude_fraction::Real = 0.7, num_weeks::Integer = 52)
    mean_rate = annual_total / 365
    # 0-based week index, as Python's `range(num_weeks)`.
    raw = [mean_rate * (1 + amplitude_fraction * cos(2π * (w - peak_week) / num_weeks))
           for w in 0:(num_weeks - 1)]
    raw_total = sum(r * 7 for r in raw)
    scale = raw_total > 0 ? annual_total / raw_total : 1.0
    return raw .* scale
end

"""
    summer_weekly_precip(annual_mm; start_week = 22, end_week = 35, num_weeks = 52,
                         days_per_week = 365/52) -> Vector{Float64}

Monsoonal forcing: all rain in a half-sine pulse over the summer weeks. Port of
`make_weekly_precip` in `bifurcation_parallel.py` / `realdata_simulate_invPDE.py`,
which drives the bifurcation diagram.

`start_week` and `end_week` are 1-based and inclusive: weeks 22–35 here are the
0-based 21–34 of the Python source (roughly June–August).

!!! note "Reproduced normalisation quirk"
    The pulse is normalised with `365/52 ≈ 7.019` days per week while the model
    delivers 7 days per week, so it receives `7/(365/52) ≈ 99.73 %` of `annual_mm`.
    Kept so that a rainfall value on a Julia bifurcation diagram means what it does
    on the Python one; pass `days_per_week = 7` to drop it.
"""
function summer_weekly_precip(annual_mm::Real; start_week::Integer = 22, end_week::Integer = 35,
                              num_weeks::Integer = 52, days_per_week::Real = 365 / 52)
    1 <= start_week <= end_week <= num_weeks ||
        throw(ArgumentError("need 1 <= start_week <= end_week <= num_weeks"))
    n_rainy = end_week - start_week + 1
    n_rainy > 1 || throw(ArgumentError("need at least two rainy weeks"))
    rates = zeros(Float64, num_weeks)
    for (i, w) in enumerate(start_week:end_week)
        rates[w] = sin(π * (i - 1) / (n_rainy - 1))
    end
    return rates .* (annual_mm / (sum(rates) * days_per_week))
end

"""
    uniform_weekly_precip(annual_mm; num_weeks = 52)

Flat forcing at `annual_mm / 365` mm/day — the fallback the Python loader uses when
a year has no weekly record.
"""
uniform_weekly_precip(annual_mm::Real; num_weeks::Integer = 52) =
    fill(annual_mm / 365, num_weeks)

"""
    annual_total(weekly_precip; days_per_week = 7)

Water delivered over a year by a weekly-rate profile (mm): what the model sees.
"""
annual_total(weekly_precip::AbstractVector; days_per_week::Real = 7) =
    sum(weekly_precip) * days_per_week
