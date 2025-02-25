import os
from pathlib import Path

# API Configuration
API_BASE_URL = os.getenv('API_BASE_URL', 'http://51.20.96.102:8000')
API_KEY = os.getenv('API_KEY')

# Base directories
BASE_DIR = Path(__file__).resolve().parent

# Local paths
LOCAL_UPLOADS_DIR = BASE_DIR / 'uploads'
LOCAL_DB_DIR = BASE_DIR / 'vector_db'

# ChromaDB Configuration
class ChromaConfig:
    def __init__(self, use_api=True):  # Default to using API
        self.use_api = use_api
        self.api_client = None
        
        if use_api:
            from api_client import APIClient
            self.api_client = APIClient(API_BASE_URL)
        else:
            # Ensure base directories exist for local usage
            os.makedirs(LOCAL_UPLOADS_DIR, exist_ok=True)
            os.makedirs(LOCAL_DB_DIR, exist_ok=True)
        
    @property
    def uploads_dir(self):
        return str(LOCAL_UPLOADS_DIR.resolve())
    
    @property
    def db_dir(self):
        return str(LOCAL_DB_DIR.resolve())
    
    def get_db_path(self, db_name, tenant_id=None):
        if not db_name:
            raise ValueError("Database name cannot be empty")
            
        if self.use_api:
            if not tenant_id:
                raise ValueError("Tenant ID is required when using API")
            return f"{API_BASE_URL}/api/v2/tenants/{tenant_id}/databases/{db_name}"
        
        # Local path handling
        db_path = LOCAL_DB_DIR.resolve() / db_name
        os.makedirs(db_path, exist_ok=True)
        return str(db_path)

# Create directories if they don't exist (for local setup)
os.makedirs(LOCAL_UPLOADS_DIR, exist_ok=True)
os.makedirs(LOCAL_DB_DIR, exist_ok=True)