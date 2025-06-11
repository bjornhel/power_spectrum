from dataclasses import dataclass

import logging
# Set up logger for this module
if __name__ == "__main__":
    logger = logging.getLogger('ct_series')
else:
    logger = logging.getLogger(__name__)


@dataclass
class CTSuperSeries():
    """
    A class to hold series of CT scans with similar characteristics in order
    to make superimages.
    The input is a list of CT series.
    """
    
    