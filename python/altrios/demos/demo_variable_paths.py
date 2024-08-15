"""
Script demonstrating how to use variable_path_list() and history_path_list()
demos to find the paths to variables within altrios classes.
"""
import os
import altrios as alt
import polars as pl

SAVE_INTERVAL = 100

# https://docs.rs/altrios-core/latest/altrios_core/train/struct.TrainConfig.html
train_config = alt.TrainConfig(
    cars_empty=50,
    cars_loaded=50,
    rail_vehicle_type="Manifest",
    train_type=alt.TrainType.Freight,
    train_length_meters=None,
    train_mass_kilograms=None,
)

# instantiate battery model
# https://docs.rs/altrios-core/latest/altrios_core/consist/locomotive/powertrain/reversible_energy_storage/struct.ReversibleEnergyStorage.html#
res = alt.ReversibleEnergyStorage.from_file(
    alt.resources_root() / "powertrains/reversible_energy_storages/Kokam_NMC_75Ah_flx_drive.yaml"
)
# instantiate electric drivetrain (motors and any gearboxes)
# https://docs.rs/altrios-core/latest/altrios_core/consist/locomotive/powertrain/electric_drivetrain/struct.ElectricDrivetrain.html
edrv = alt.ElectricDrivetrain(
    pwr_out_frac_interp=[0., 1.],
    eta_interp=[0.98, 0.98],
    pwr_out_max_watts=5e9,
    save_interval=SAVE_INTERVAL,
)

bel = alt.Locomotive.build_battery_electric_loco(
    reversible_energy_storage=res,
    drivetrain=edrv,
    loco_params=alt.LocoParams.from_dict(dict(
        pwr_aux_offset_watts=8.55e3,
        pwr_aux_traction_coeff_ratio=540.e-6,
        force_max_newtons=667.2e3,
)))

# construct a vector of one BEL and several conventional locomotives
loco_vec = [bel.clone()] + [alt.Locomotive.default()] * 7
# instantiate consist
loco_con = alt.Consist(
    loco_vec,
    SAVE_INTERVAL,
)

tsb = alt.TrainSimBuilder(
    train_id="0",
    origin_id="Minneapolis",
    destination_id="Superior",
    train_config=train_config,
    loco_con=loco_con,
)

rail_vehicle_file = "rolling_stock/" + train_config.rail_vehicle_type + ".yaml"
rail_vehicle = alt.RailVehicle.from_file(
    alt.resources_root() / rail_vehicle_file)

network = alt.Network.from_file(
    alt.resources_root() / "networks/Taconite-NoBalloon.yaml")

location_map = alt.import_locations(
    alt.resources_root() / "networks/default_locations.csv")

train_sim: alt.SpeedLimitTrainSim = tsb.make_speed_limit_train_sim(
    rail_vehicle=rail_vehicle,
    location_map=location_map,
    save_interval=1,
)
train_sim.set_save_interval(SAVE_INTERVAL)

est_time_net, _consist = alt.make_est_times(train_sim, network)

timed_link_path = alt.run_dispatch(
    network,
    alt.SpeedLimitTrainSimVec([train_sim]),
    [est_time_net],
    False,
    False,
)[0]

# uncomment this line to see example of logging functionality
# alt.utils.set_log_level("DEBUG")

t0 = time.perf_counter()
train_sim.walk_timed_path(
    network=network,
    timed_path=timed_link_path,
)
t1 = time.perf_counter()
print(f'Time to simulate: {t1 - t0:.5g}')
assert len(train_sim.history) > 1

# whether to run assertions, enabled by default
ENABLE_ASSERTS = os.environ.get("ENABLE_ASSERTS", "true").lower() == "true"
# whether to override reference files used in assertions, disabled by default
ENABLE_REF_OVERRIDE = os.environ.get("ENABLE_REF_OVERRIDE", "false").lower() == "true"
# directory for reference files for checking sim results against expected results
ref_dir = alt.resources_root() / "demos/demo_variable_paths/"

# print out all subpaths for variables in SimDrive
print("List of variable paths for SimDrive:" + "\n".join(train_sim.variable_path_list()))
if ENABLE_REF_OVERRIDE:
    ref_dir.mkdir(exist_ok=True, parents=True)
    with open(ref_dir / "variable_path_list_expected.txt", 'w') as f:
        for line in train_sim.variable_path_list():
            f.write(line + "\n")
if ENABLE_ASSERTS:
    print("Checking output of `variable_path_list()`")
    with open(ref_dir / "variable_path_list_expected.txt", 'r') as f:
        variable_path_list_expected = [line.strip() for line in f.readlines()]
    assert variable_path_list_expected == train_sim.variable_path_list()
print("\n")

# print out all subpaths for history variables in SimDrive
print("List of history variable paths for SimDrive:" +  "\n".join(train_sim.history_path_list()))
print("\n")

# print results as dataframe
print("Results as dataframe:\n", train_sim.to_dataframe(), sep="")
if ENABLE_REF_OVERRIDE:
    df:pl.DataFrame = train_sim.to_dataframe().lazy().collect()
    df.write_csv(ref_dir / "to_dataframe_expected.csv")
    print("Success!")
if ENABLE_ASSERTS:
    print("Checking output of `to_dataframe`")
    to_dataframe_expected = pl.scan_csv(ref_dir / "to_dataframe_expected.csv").collect()
    assert to_dataframe_expected.equals(train_sim.to_dataframe())
    print("Success!")
