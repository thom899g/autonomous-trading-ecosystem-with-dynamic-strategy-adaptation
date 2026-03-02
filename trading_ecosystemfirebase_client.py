"""
Firebase integration for state management, logging, and real-time data streaming.
Implements robust error handling and reconnection logic.
"""
import logging
import json
import time
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timezone

import firebase_admin
from firebase_admin import credentials, firestore, db
from google.cloud.firestore_v1.base_query import FieldFilter
from google.api_core.exceptions import GoogleAPICallError, RetryError

from config import config

class FirebaseManager:
    """Manages Firebase connections with automatic retry and state recovery"""
    
    def __init__(self):
        self._app = None
        self._firestore = None
        self._realtime_db = None
        self._connected = False
        self._last_heartbeat = None
        self._max_retries = 3
        self._retry_delay = 5
        
    def initialize(self) -> bool:
        """Initialize Firebase connection with exponential backoff"""
        if self._connected:
            logging.warning("Firebase already initialized")
            return True
            
        for attempt in range(self._max_retries):
            try:
                if not firebase_admin._apps:
                    cred = credentials.Certificate(config.firebase.credentials_path)
                    self._app = firebase_admin.initialize_app(cred, {
                        'projectId': config.firebase.project_id,
                        'databaseURL': f'https://{config.firebase.project_id}.firebaseio.com/'
                    })
                
                self._firestore = firestore.client()
                self._realtime_db = db.reference()
                self._connected = True
                self._last_heartbeat = datetime.now(timezone.utc)
                
                # Test connection
                self._test_connection()
                
                logging.info(f"Firebase initialized successfully (attempt {attempt + 1})")
                return True
                
            except (GoogleAPICallError, ValueError, FileNotFoundError) as e:
                logging.error(f"Firebase initialization failed (attempt {attempt + 1}): {str(e)}")
                if attempt < self._max_retries - 1:
                    time.sleep(self._retry_delay * (2 ** attempt))
                continue
                
        logging