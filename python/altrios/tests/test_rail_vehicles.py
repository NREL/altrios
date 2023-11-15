import pandas as pd
import altrios as alt

def test_rail_vehicles():
    """
    Verifies that csv- and yaml-defined rail vehicles can be loaded and produce same result
    """

    rail_vehicle_path = alt.resources_root() / "rolling_stock"
    df = pd.read_csv(rail_vehicle_path / "rail_vehicles.csv")
    rail_vehicle_map = alt.import_rail_vehicles(str(rail_vehicle_path / "rail_vehicles.csv"))
    for i, row in df.iterrows():
        rv_csv = rail_vehicle_map[row['Car Type']]
        rv_yaml = alt.RailVehicle.from_file(
            str(rail_vehicle_path / f"{row['Car Type']}.yaml")
        )
        assert rv_csv.to_json() == rv_yaml.to_json()
