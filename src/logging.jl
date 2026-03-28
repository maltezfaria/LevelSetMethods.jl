"""
    struct StepRecord

Record of a single time step during level-set integration.

- `step`: cumulative step number across all `integrate!` calls (1-indexed)
- `t`: simulation time at end of step
- `wall_time`: total wall-clock time for this step (seconds)
- `reinit_time`: time spent in reinitialization (0.0 if not performed)
- `did_reinit`: whether reinitialization was performed this step
- `update_times`: time per `update_term!`, summed over all RK stages (seconds)
- `compute_times`: time per `_compute_term` loop, summed over all RK stages (seconds)
- `ϕ_min`, `ϕ_max`: level-set extrema at end of step
"""
struct StepRecord
    step::Int
    t::Float64
    wall_time::Float64
    reinit_time::Float64
    did_reinit::Bool
    update_times::Vector{Float64}
    compute_times::Vector{Float64}
    ϕ_min::Float64
    ϕ_max::Float64
end

"""
    mutable struct SimulationLog

Accumulates per-step timing and progress data for a [`LevelSetEquation`](@ref).  Persists
and accumulates across multiple calls to [`integrate!`](@ref).  Use [`reset_log!`](@ref) to
clear the history.
"""
mutable struct SimulationLog
    t0::Float64
    term_labels::Vector{String}
    records::Vector{StepRecord}
end

SimulationLog(t0, terms) = SimulationLog(Float64(t0), [sprint(show, t) for t in terms], StepRecord[])

"""
    reset_log!(log::SimulationLog, t0 = log.t0)

Clear all records from `log` and optionally reset the initial time.
"""
function reset_log!(log::SimulationLog, t0 = log.t0)
    log.t0 = Float64(t0)
    empty!(log.records)
    return log
end

function Base.show(io::IO, ::MIME"text/plain", log::SimulationLog)
    records = log.records
    if isempty(records)
        print(io, "SimulationLog (empty)")
        return io
    end

    nsteps = length(records)
    t0 = log.t0
    tf = records[end].t
    total_wall = sum(r.wall_time for r in records)
    avg_wall_ms = total_wall / nsteps * 1.0e3

    # Derive Δt from consecutive t values
    Δts = Vector{Float64}(undef, nsteps)
    Δts[1] = records[1].t - t0
    for i in 2:nsteps
        Δts[i] = records[i].t - records[i - 1].t
    end

    n_reinit = count(r -> r.did_reinit, records)

    println(io, "SimulationLog: $nsteps steps, t ∈ [$(round(t0; sigdigits = 4)), $(round(tf; sigdigits = 4))]")
    println(io, "  ├─ wall time: $(round(total_wall; sigdigits = 4)) s  (avg $(round(avg_wall_ms; sigdigits = 4)) ms/step)")

    if n_reinit > 0
        total_reinit = sum(r.reinit_time for r in records)
        avg_reinit_ms = total_reinit / n_reinit * 1.0e3
        reinit_pct = round(100 * total_reinit / total_wall; sigdigits = 3)
        println(io, "  ├─ reinit:    $n_reinit calls, avg $(round(avg_reinit_ms; sigdigits = 4)) ms  ($reinit_pct% of total)")
    else
        println(io, "  ├─ reinit:    none")
    end

    for (i, label) in enumerate(log.term_labels)
        avg_update_ms = sum(r.update_times[i] for r in records) / nsteps * 1.0e3
        avg_compute_ms = sum(r.compute_times[i] for r in records) / nsteps * 1.0e3
        println(io, "  ├─ $label: avg $(round(avg_update_ms; sigdigits = 4)) ms update, $(round(avg_compute_ms; sigdigits = 4)) ms compute / step")
    end

    last = records[end]
    println(io, "  ├─ ϕ range:   [$(round(last.ϕ_min; sigdigits = 4)), $(round(last.ϕ_max; sigdigits = 4))]")

    Δt_min = minimum(Δts)
    Δt_max = maximum(Δts)
    Δt_avg = sum(Δts) / nsteps
    print(io, "  └─ Δt:        min=$(round(Δt_min; sigdigits = 4))  max=$(round(Δt_max; sigdigits = 4))  avg=$(round(Δt_avg; sigdigits = 4))")
    return io
end

"""
    _timed_update_terms!(terms, ϕ, t, update_times)

Run `update_term!` for each term, accumulating the elapsed time per term into `update_times`.
"""
function _timed_update_terms!(terms, ϕ, t, update_times)
    for i in eachindex(terms)
        t0 = time_ns()
        update_term!(terms[i], ϕ, t)
        update_times[i] += (time_ns() - t0) / 1.0e9
    end
    return
end

"""
    _timed_reinit!(ϕ, reinit, nsteps) -> (elapsed, did_reinit)

Run `reinitialize!` if due at this step, returning the elapsed time and whether it ran.
"""
_timed_reinit!(_, ::Nothing, _) = (0.0, false)
function _timed_reinit!(ϕ, reinit, nsteps)
    mod(nsteps, reinit.reinit_freq) == 0 || return (0.0, false)
    t0 = time_ns()
    reinitialize!(ϕ, reinit)
    return ((time_ns() - t0) / 1.0e9, true)
end

"""
    _level_set_extrema(src) -> (ϕ_min, ϕ_max)

Compute the extrema of the level-set values. Handles both `Array` and `Dict` storage.
"""
function _level_set_extrema(src)
    v = values(src)
    vals_iter = v isa AbstractDict ? Base.values(v) : v
    return extrema(vals_iter)
end

"""
    _push_record!(log, tc, t_step, reinit_time, did_reinit, update_times, compute_times, ϕ_min, ϕ_max)

Add a new `StepRecord` to `log`.
"""
function _push_record!(log, tc, t_step, reinit_time, did_reinit, update_times, compute_times, ϕ_min, ϕ_max)
    wall_time = (time_ns() - t_step) / 1.0e9
    return push!(
        log.records, StepRecord(
            length(log.records) + 1, tc, wall_time, reinit_time, did_reinit,
            copy(update_times), copy(compute_times), ϕ_min, ϕ_max,
        )
    )
end
