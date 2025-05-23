# %%
import polars as pl
import altrios as alt
from altrios.train_planner import manual_train_planner


consist_plan = pl.read_csv(alt.resources_root() / "demo_data/Demo Consist Plan.csv")

loco_map = {"x": "y"}

train_consist_plan, loco_pool, refuelers, speed_limit_train_sims, est_time_nets = (
    manual_train_planner(consist_plan, loco_map)
)
