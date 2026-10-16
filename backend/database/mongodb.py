from pymongo import MongoClient
from app.config import Settings


class Database:
    """Lazy connection: absence or outage must not prevent FastAPI startup."""
    def __init__(self, config: Settings):
        self.client = None
        self.db = None
        if config.mongo_uri:
            try:
                self.client = MongoClient(config.mongo_uri, serverSelectionTimeoutMS=2500,
                                          connectTimeoutMS=2500, socketTimeoutMS=15000,
                                          maxPoolSize=8, appname='hmeg-research')
                self.db = self.client[config.database]
            except Exception:
                # Never log URIs or driver exceptions containing credentials.
                self.client = None

    def connected(self) -> bool:
        if self.client is None:
            return False
        try:
            self.client.admin.command('ping')
            return True
        except Exception:
            return False

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
