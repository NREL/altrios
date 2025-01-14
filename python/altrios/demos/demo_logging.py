"""
This demo is currently deprecated because all logging functionality has been
removed from altrios-core due to the fact that it was unwieldy and in recent
versions of pyo3, Rust's `println` macro works through pyo3.
"""
from altrios import ConsistSimulation
from altrios.utilities import set_log_level
import logging

log = logging.getLogger(__name__)
set_log_level(logging.INFO)

if __name__ == "__main__":
    c = ConsistSimulation.default()

    log.info("Starting walk method")

    c.walk()
