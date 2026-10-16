"""Verify connectivity without printing credentials or raw driver exceptions."""
import argparse
from app.config import settings
from database.mongodb import Database
from database.indexes import create_indexes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initialize', action='store_true', help='Create HMEG collection indexes without deleting documents')
    args = parser.parse_args()
    if not settings.mongo_uri:
        print('Not configured. Set MONGODB_URI in backend/.env. See docs/MONGODB_ATLAS.md.')
        return 1
    if settings.database in {'admin', 'local', 'config'}:
        print('Choose an application database such as hmeg_research, not a system database.')
        return 1
    database = Database(settings)
    try:
        if not database.connected():
            print('Connection failed. Check the database credentials, IP access list, and cluster availability.')
            return 1
        print('MongoDB connection successful.')
        if args.initialize:
            create_indexes(database.db)
            print('HMEG indexes initialized. Existing documents were retained.')
        return 0
    except Exception:
        print('Initialization failed. Check database permissions and existing index constraints.')
        return 1
    finally:
        database.close()


if __name__ == '__main__':
    raise SystemExit(main())
