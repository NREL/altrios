"""
Script demonstrating how to use variable_path_list() and history_path_list()
demos to find the paths to variables within altrios classes.
"""

import altrios as alt

SAVE_INTERVAL = 1
# load hybrid consist
fc = alt.FuelConverter.default()

gen = alt.Generator.default()

edrv = alt.ElectricDrivetrain.default()

conv = alt.Locomotive.build_conventional_loco(
    fuel_converter=fc,
    generator=gen,
    drivetrain=edrv,
    loco_params=alt.LocoParams(
        pwr_aux_offset_watts=13e3,
        pwr_aux_traction_coeff_ratio=1.1e-3,
        force_max_newtons=667.2e3,
    ),
    save_interval=SAVE_INTERVAL,
)


# %%

pt = alt.PowerTrace.default()

sim = alt.LocomotiveSimulation(conv, pt, SAVE_INTERVAL)

# print relative variable paths within locomotive simulation
print("Locomotive simulation variable paths:\n", "\n".join(sim.variable_path_list()))
# print relative history variable paths within locomotive simulation
print("Locomotive simulation history variable paths:\n", ".\n".join(sim.history_path_list()))
