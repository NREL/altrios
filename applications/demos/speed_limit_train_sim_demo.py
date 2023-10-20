# %%
import altrios as alt
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
sns.set()

# %%

SAVE_INTERVAL = 1

train_summary = alt.TrainSummary(
    rail_vehicle_type="Manifest",
    cars_empty=50,
    cars_loaded=50,
    train_type=None,
    train_length_meters=None,
    train_mass_kilograms=None,
)

# instantiate battery model
res = alt.ReversibleEnergyStorage.from_file(
    str(alt.resources_root() / 
        "powertrains/reversible_energy_storages/Kokam_NMC_75Ah_flx_drive.yaml"
    )
)
# instantiate electric drivetrain (motors and any gearboxes)
edrv = alt.ElectricDrivetrain(
    pwr_out_frac_interp=[0., 1.],
    eta_interp=[0.98, 0.98],
    pwr_out_max_watts=5e9,
    save_interval=SAVE_INTERVAL,
)

# construct a vector of one BEL and 3 conventional locomotives
loco_vec = [
    alt.Locomotive.build_battery_electric_loco(
        reversible_energy_storage=res,
        drivetrain=edrv,
        loco_params=alt.LocoParams.from_dict(dict(
            pwr_aux_offset_watts=8.55e3,
            pwr_aux_traction_coeff_ratio=540.e-6,
            force_max_newtons=667.2e3,
        ))
    )] + [
    alt.Locomotive.default(),
] * 7
# instantiate consist
loco_con = alt.Consist(
    loco_vec,
    SAVE_INTERVAL,
)
init_train_state = alt.InitTrainState(
    # TODO: fix how `train_length_meters` is set on instantiation of `train_summary`
    # offset_meters=train_summary.train_length_meters
    offset_meters=666,
)

tsb = alt.TrainSimBuilder(
    # TODO: make sure `train_id` is being used meaningfully
    train_id="0",
    # Question: what happens if we use arbitrary nonsense for `origin_id` and `destination_id`?
    origin_id="Minneapolis",
    destination_id="Superior",
    train_summary=train_summary,
    loco_con=loco_con,
    init_train_state=init_train_state,
)

# make sure rail_vehicle_map can be constructed from yaml file and such
rail_vehicle_file = "rolling_stock/rail_vehicles.csv"
rail_vehicle_map = alt.import_rail_vehicles(
    str(alt.resources_root() / rail_vehicle_file)
)

network = alt.import_network(str(alt.resources_root() / "networks/Taconite.yaml"))


location_map = alt.import_locations(
    str(alt.resources_root() / "networks/default_locations.csv")
)

def get_train_sim(
        loco_con: alt.Consist = alt.Consist.default()
    ) -> alt.SpeedLimitTrainSim:
    # function body
    tsb = alt.TrainSimBuilder(
        "demo_train",
        "Hibbing",
        "Allouez",
        train_summary,
        loco_con,
    )
    train_sim = tsb.make_speed_limit_train_sim(
        rail_vehicle_map,
        location_map,
        save_interval=SAVE_INTERVAL,
    )
    network_filename_path = alt.resources_root() / "networks/Taconite.yaml"
    train_sim.extend_path(
        str(network_filename_path),
        train_path
    )

    train_sim.set_save_interval(SAVE_INTERVAL)
    return train_sim


# %%
train_sim = get_train_sim()

t0 = time.perf_counter()
train_sim.walk()
t1 = time.perf_counter()
print(f'Time to simulate: {t1 - t0:.5g}')

# %%

fig, ax = plt.subplots(3, 1, sharex=True)
start_idx = 0
end_idx = -1

ax[0].plot(
    np.array(train_sim.history.time_seconds)[start_idx:end_idx],
    np.array(train_sim.history.pwr_whl_out_watts)[start_idx:end_idx],
    label="whl_out",
)
# ax[0].plot(
#     np.array(train_sim.history.time_seconds)[start_idx:end_idx],
#     (np.array(train_sim.fric_brake.history.force_newtons) *
#         np.array(train_sim.history.velocity_meters_per_second))[start_idx:end_idx],
#     label="fric brake",
# )
ax[0].set_ylabel('Power [W]')
ax[0].legend(loc="lower right")

# ax[1].plot(
#     np.array(train_sim.history.time_seconds)[start_idx:end_idx],
#     np.array(train_sim.fric_brake.history.force_newtons)[
#         start_idx:end_idx],
# )
# ax[1].set_ylabel('Brake Force [N]')

ax[1].plot(
    np.array(train_sim.history.time_seconds)[start_idx:end_idx],
    np.array(train_sim.loco_con.loco_vec.tolist()[1].res.history.soc)[
        start_idx:end_idx],
)
ax[1].set_ylabel('BEL SOC')

ax[-1].plot(
    np.array(train_sim.history.time_seconds)[start_idx:end_idx],
    np.array(train_sim.history.velocity_meters_per_second)[start_idx:end_idx],
    label='achieved'
)
ax[-1].plot(
    np.array(train_sim.history.time_seconds)[start_idx:end_idx],
    np.array(train_sim.history.speed_target_meters_per_second)[
        start_idx:end_idx],
    label='target'
)
ax[-1].plot(
    np.array(train_sim.history.time_seconds)[start_idx:end_idx],
    np.array(train_sim.history.speed_limit_meters_per_second)[
        start_idx:end_idx],
    label='limit'
)
ax[-1].legend()
ax[-1].set_xlabel('Time [s]')
ax[-1].set_ylabel('Speed [m/s]')

# %%

# print composition of consist
for i, loco in enumerate(train_sim.loco_con.loco_vec.tolist()):
    print(f'locomotive at position {i} is {loco.loco_type()}')

# %% Sweep of BELs

bel = alt.Locomotive.default_battery_electic_loco()
diesel = alt.Locomotive.default()

loco_cons = [
    alt.Consist(
        [bel] * n_bels + [diesel] * (5 - n_bels),
        SAVE_INTERVAL,
    ) for n_bels in range(0, 6)
]

train_sims = [get_train_sim(loco_con) for loco_con in loco_cons]

for i, ts in enumerate(train_sims):
    try:
        ts.walk()
        train_sims[i] = ts
    except RuntimeError as e:
        train_sims[i] = e

    ts.to_file('speed_limit_train_sim_results_{}.json'.format(i))
# %%
