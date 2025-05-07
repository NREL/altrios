# %% Copied from ALTRIOS version 'v0.2.3'. Guaranteed compatibility with this version only.
# %%
# Script for running the Wabtech BEL consist for sample data from Barstow to Stockton
# Consist comprises [2X Tier 4](https://www.wabteccorp.com/media/3641/download?inline)
# + [1x BEL](https://www.wabteccorp.com/media/466/download?inline)


import altrios as alt
import numpy as np
import time
import seaborn as sns
import pandas as pd
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

sns.set_theme()

SHOW_PLOTS = alt.utils.show_plots()


SAVE_INTERVAL = 1

# power level in horsepower by notch.  negative is dynamic braking
notch_schedule = {
    -8: -4400,
    -7: -3740,
    -6: -3070,
    -5: -2350,
    -4: -1765,
    -3: -1265,
    -2: -615,
    -1: -310,
    0: 0,
    1: 310,
    2: 615,
    3: 1265,
    4: 1765,
    5: 2350,
    6: 3070,
    7: 3740,
    8: 4400,
}


Trace1 = [
    [1, 120],
    [2, 120],
    [3, 120],
    [4, 3600],
    [1, 600],
    [0, 120],
    [-1, 600],
    [-4, 600],
    [-1, 120],
]

Trace2 = [
    [1, 120],
    [2, 120],
    [3, 120],
    [4, 3600],
    [1, 600],
    [0, 120],
    [-4, 600],
    [-6, 600],
    [-4, 120],
]
traces = [Trace1, Trace2]


def PowerTraceFromNotch(NotchTrace, NotchSchedule, Repeats=1):
    TotalSeconds = np.array(NotchTrace)[:, 1].sum()
    PowerTrace = np.zeros(TotalSeconds)
    CurrentSeconds = 0
    for NotchDuration in NotchTrace:
        print("{}, {}".format(CurrentSeconds, NotchSchedule[NotchDuration[0]]))
        PowerTrace[CurrentSeconds : CurrentSeconds + NotchDuration[1]] = NotchSchedule[
            NotchDuration[0]
        ]
        CurrentSeconds = CurrentSeconds + NotchDuration[1]

    # TODO: put slew rate limiter on power to make ramped notch transitions

    PowerTrace = np.tile(PowerTrace, Repeats)
    PowerTrace = PowerTrace * 745.699872  # convert from hp to watts

    pt_dict = {
        "time_seconds": np.arange(0, PowerTrace.shape[0]).tolist(),
        "pwr_watts": PowerTrace.tolist(),
        "engine_on": [True] * PowerTrace.shape[0],
    }
    pt = alt.PowerTrace.from_json(json.dumps(pt_dict))
    return pt


PowerTraces = []
for trace in traces:
    print("--------------------")
    PowerTraces.append(PowerTraceFromNotch(trace, notch_schedule, Repeats=3))

# pt = alt.PowerTrace.default()
# pt_dict = json.loads(pt.to_json())
# pt_dict['time_seconds'] = np.arange(0,30000,100).tolist()
# pt = pt.from_json(json.dumps(pt_dict))


res = alt.ReversibleEnergyStorage.from_file(
    alt.resources_root()
    / "powertrains/reversible_energy_storages/Kokam_NMC_75Ah_flx_drive.yaml"
)
# instantiate electric drivetrain (motors and any gearboxes)
edrv = alt.ElectricDrivetrain(
    pwr_out_frac_interp=[0.0, 1.0],
    eta_interp=[0.98, 0.98],
    pwr_out_max_watts=5e9,
    save_interval=SAVE_INTERVAL,
)


bel = alt.Locomotive.build_battery_electric_loco(
    reversible_energy_storage=res,
    drivetrain=edrv,
    loco_params=alt.LocoParams.from_dict(
        dict(
            pwr_aux_offset_watts=8.55e3,
            pwr_aux_traction_coeff_ratio=540.0e-6,
            force_max_newtons=667.2e3,
        )
    ),
)

LocoRegenEnergy = []
LocoPositiveEnergy = []
PowerTracePositiveEnergy = []
PowerTraceRegenEnergy = []
FractionOfDemand = []
RegenFraction = []

fig = make_subplots(
    rows=len(PowerTraces), cols=1, specs=[[{"secondary_y": True}]] * len(PowerTraces)
)

i = 0
for trace in PowerTraces:
    print(i)
    # instantiate battery model
    t0 = time.perf_counter()
    sim = alt.LocomotiveSimulation(bel, trace, True, SAVE_INTERVAL)
    t1 = time.perf_counter()
    print(f"Time to load: {t1-t0:.3g}")

    # simulate
    t0 = time.perf_counter()
    sim.walk()
    t1 = time.perf_counter()
    print(f"Time to simulate: {t1-t0:.5g}")

    bel_rslt = sim.loco_unit

    LocoPositiveEnergy.append(
        np.sum(
            np.clip(
                np.array(bel_rslt.history.pwr_out_watts) * 1e-6, a_max=np.inf, a_min=0
            )
        )
        / 3600
    )
    LocoRegenEnergy.append(
        np.sum(
            np.clip(
                np.array(bel_rslt.history.pwr_out_watts) * 1e-6, a_max=0, a_min=-np.inf
            )
        )
        / 3600
    )
    PowerTracePositiveEnergy.append(
        np.sum(
            np.clip(np.array(sim.power_trace.pwr_watts) * 1e-6, a_max=np.inf, a_min=0)
        )
        / 3600
    )
    PowerTraceRegenEnergy.append(
        np.sum(
            np.clip(np.array(sim.power_trace.pwr_watts) * 1e-6, a_max=0, a_min=-np.inf)
        )
        / 3600
    )
    FractionOfDemand.append(LocoPositiveEnergy[-1] / PowerTracePositiveEnergy[-1])
    RegenFraction.append(LocoRegenEnergy[-1] / PowerTraceRegenEnergy[-1])
    t_s = np.array(sim.power_trace.time_seconds)

    fig.add_trace(
        go.Scatter(
            x=t_s,
            y=np.array(bel_rslt.history.pwr_out_watts) * 1e-6,
            name="Locomotive Power",
            marker={"color": px.colors.qualitative.Plotly[0]},
        ),
        row=i + 1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=t_s,
            y=np.array(sim.power_trace.pwr_watts) * 1e-6,
            name="Power Demand",
            marker={"color": px.colors.qualitative.Plotly[1]},
        ),
        row=i + 1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=t_s,
            y=np.array(bel_rslt.res.history.soc) * 100,
            name="SOC",
            marker={"color": px.colors.qualitative.Plotly[2]},
        ),
        secondary_y=True,
        row=i + 1,
        col=1,
    )

    fig.update_xaxes(tickfont=dict(family="Arial", size=20), row=i + 1, col=1)
    fig.update_yaxes(
        tickfont=dict(family="Arial", size=20), range=[-3.1, 3.1], row=i + 1, col=1
    )
    fig.update_yaxes(title_text="Power [MW]", title_font=dict(size=24, family="Arial"))
    fig.update_yaxes(
        title_text="SOC [%]",
        title_font=dict(size=24, family="Arial"),
        secondary_y=True,
        range=[0, 100],
        showgrid=False,
    )
    fig.update_xaxes(
        title_text="Time [s]",
        title_font=dict(size=20, family="Arial"),
        row=i + 1,
        col=1,
    )
    fig.update_layout(legend_title=" ", font=dict(size=20))

    i = i + 1

fig.write_html("test.html", auto_open=True)

# %%
Data = pd.DataFrame()
Data["Actual Regen Energy"] = LocoRegenEnergy
Data["Actual Positive Tractive Effort"] = LocoPositiveEnergy
Data["Potential Positive Tractive Effort"] = PowerTracePositiveEnergy
Data["Potential Regen Energy"] = PowerTraceRegenEnergy
Data["Demand Fraction Achieved"] = FractionOfDemand
Data["Regen Fraction"] = RegenFraction
Data["Power Trace Name"] = ["Trace 1", "Trace 2"]
fig = px.bar(
    Data,
    x="Power Trace Name",
    y=[
        "Actual Positive Tractive Effort",
        "Potential Positive Tractive Effort",
        "Actual Regen Energy",
        "Potential Regen Energy",
    ],
    barmode="group",
)

fig.update_xaxes(tickfont=dict(family="Arial", size=20), row=i + 1, col=1)
fig.update_yaxes(
    tickfont=dict(family="Arial", size=20), range=[-3.1, 3.1], row=i + 1, col=1
)
fig.update_yaxes(title_text="Energy [MWh]", title_font=dict(size=24, family="Arial"))
fig.update_xaxes(title_font=dict(size=20, family="Arial"), row=i + 1, col=1)
fig.update_layout(legend_title=" ", font=dict(size=20))
fig.write_html("bar.html", auto_open=True)


# %%
