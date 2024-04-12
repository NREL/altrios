"""
Demonstrates how to convert data between previous and current ALTRIOS release compatibility.

"""
# %%

import json
from typing import Tuple

import altrios as alt

def migrate_network() -> Tuple[alt.Network, alt.Network]:
    old_network_path = alt.resources_root() / "networks/Taconite_v0.1.6.yaml"
    new_network_path = alt.resources_root() / "networks/Taconite.yaml"

    network_from_old = alt.Network.from_file(old_network_path)
    network_from_new = alt.Network.from_file(new_network_path)

    # `network_from_old` could be used to overwrite the file in the new format with 
    # ```
    # network_from_old.to_file(
    #     alt.resources_root() / "networks/Taconite.yaml"
    # )
    # ```

    # TODO: change this to direct comparison after figuring out how to enable that via pyo3
    assert json.loads(network_from_new.to_json()
        ) == json.loads(network_from_old.to_json())

    return (network_from_old, network_from_new)

network_from_old, network_from_new = migrate_network()
