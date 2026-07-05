import sys
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture()
def sample_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "step": [1, 2, 3, 4],
            "type": ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN"],
            "amount": [181.0, 250.0, 9839.64, 500.0],
            "nameOrig": ["C1305486145", "C840083671", "C1231006815", "C222"],
            "oldbalanceOrg": [181.0, 250.0, 170136.0, 1000.0],
            "newbalanceOrig": [0.0, 0.0, 160296.36, 1500.0],
            "nameDest": ["C553264065", "C38997010", "M1979787155", "C333"],
            "oldbalanceDest": [0.0, 21182.0, 0.0, 100.0],
            "newbalanceDest": [0.0, 21432.0, 0.0, 0.0],
            "isFraud": [1, 1, 0, 0],
            "isFlaggedFraud": [0, 0, 0, 0],
        }
    )