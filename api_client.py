import os
import requests
from typing import List, Dict, Optional

class APIClient:
    def __init__(self, base_url: str = "http://51.20.96.102:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> dict:
        """Make HTTP request to the API"""
        url = f"{self.base_url}{endpoint}"
        response = self.session.request(method, url, **kwargs)
        response.raise_for_status()
        return response.json()
    
    def get_databases(self, tenant_id: str) -> List[str]:
        """Get list of available databases for a tenant"""
        endpoint = f"/api/v2/tenants/{tenant_id}/databases"
        return self._make_request('GET', endpoint)
    
    def create_database(self, tenant_id: str, database_name: str) -> dict:
        """Create a new database"""
        endpoint = f"/api/v2/tenants/{tenant_id}/databases"
        return self._make_request('POST', endpoint, json={'name': database_name})
    
    def get_collections(self, tenant_id: str, database_name: str) -> List[str]:
        """Get list of collections in a database"""
        endpoint = f"/api/v2/tenants/{tenant_id}/databases/{database_name}/collections"
        return self._make_request('GET', endpoint)
    
    def query_collection(self, tenant_id: str, database_name: str, collection_id: str, query: str) -> dict:
        """Query a collection"""
        endpoint = f"/api/v2/tenants/{tenant_id}/databases/{database_name}/collections/{collection_id}/query"
        return self._make_request('POST', endpoint, json={'query': query})
    
    def add_documents(self, tenant_id: str, database_name: str, collection_id: str, documents: List[Dict]) -> dict:
        """Add documents to a collection"""
        endpoint = f"/api/v2/tenants/{tenant_id}/databases/{database_name}/collections/{collection_id}/add"
        return self._make_request('POST', endpoint, json={'documents': documents})