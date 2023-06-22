import matplotlib.pyplot as plt
from altrios.postprocessing import TrainSimResults


def plot_speed(train_sim_results: TrainSimResults, ax: plt.Axes):
    ax.plot(
        train_sim_results.time_seconds,
        train_sim_results.speed_meters_per_second,
        label="speed",
    )
    ax.set_ylabel("Speed [m/s]")
    ax.set_xlabel("Time [s]")
    plt.tight_layout()
    ax.legend()
