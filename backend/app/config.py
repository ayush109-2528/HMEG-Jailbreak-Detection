"""Server configuration. Never serialized to the browser."""
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')


@dataclass(frozen=True)
class Settings:
    mongo_uri: str = os.getenv('MONGODB_URI', '')
    database: str = os.getenv('MONGODB_DATABASE', 'hmeg_research')
    origins: tuple[str, ...] = tuple(x.strip() for x in os.getenv('FRONTEND_URL', 'http://localhost:5176,http://127.0.0.1:5176').split(',') if x.strip())
    model_dir: Path = ROOT / os.getenv('MODEL_DIR', 'models')
    low: float = float(os.getenv('LOW_THRESHOLD', '.2'))
    high: float = float(os.getenv('HIGH_THRESHOLD', '.8'))
    threshold: float = float(os.getenv('FINAL_THRESHOLD', '.5'))
    requests_per_minute: int = int(os.getenv('PREDICTION_REQUESTS_PER_MINUTE', '20'))

    def __post_init__(self):
        if not 0 <= self.low < self.high <= 1 or not 0 <= self.threshold <= 1:
            raise ValueError('Require 0 <= LOW_THRESHOLD < HIGH_THRESHOLD <= 1 and FINAL_THRESHOLD in [0, 1].')


settings = Settings()
