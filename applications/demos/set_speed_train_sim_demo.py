# %%
import time

import altrios as alt 

save_interval = 1

def run_sim():
    train_sim = alt.SetSpeedTrainSim.default()
    train_sim.set_save_interval(1)
    t0 = time.perf_counter()
    train_sim.walk()
    t1 = time.perf_counter()
    print(f'Time to simulate: {t1 - t0:.5g}')
    return train_sim


# %%
train_sim = run_sim()
# %%
