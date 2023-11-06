from altrios import ConsistSimulation
from altrios.utilities import set_log_level

import logging

log = logging.getLogger(__name__)
set_log_level(logging.INFO)


if __name__ == "__main__":
    c = ConsistSimulation.default()

    log.info("Starting walk method")

    c.walk()
