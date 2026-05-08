"""Reset all download states and clear cache"""

from core import cache_db

# Initialize and cleanup cache
cache_db.init_db()
cache_db.cleanup_old_entries()

print("Cache database cleaned")
print("Download states reset")
print("\nYou can now restart the app with: python app.py")
