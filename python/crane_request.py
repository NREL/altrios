import simpy

def train_process(env, train_id, track_id, crane):
    print(f"Train {train_id} arrives at track {track_id} at time {env.now}")
    with crane['resource'].request() as request:
        yield request
        print(f"Crane {crane['id']} ({crane['type']}) starts working on Train {train_id} at time {env.now}")
        yield env.timeout(1)
        print(f"Crane {crane['id']} ({crane['type']}) finished working on Train {train_id} at time {env.now}")
    print(f"Train {train_id} leaves track {track_id} at time {env.now}")


def main(env):
    cranes = [
        {"id": 1, "type": "hybrid", "resource": simpy.Resource(env, capacity=1)},
        {"id": 1, "type": "electric", "resource": simpy.Resource(env, capacity=1)}
    ]
    print(f"Cranes: {cranes}")

    env.process(train_process(env, train_id=12, track_id=1, crane=cranes[0]))  # Train 12 -> Track 1
    yield env.timeout(1)
    env.process(train_process(env, train_id=70, track_id=2, crane=cranes[1]))  # Train 70 -> Track 2


env = simpy.Environment()
env.process(main(env))
env.run()

