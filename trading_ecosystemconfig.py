"""
Configuration management for the Autonomous Trading Ecosystem.
Centralizes all settings with environment variable fallbacks and validation.
"""
import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

@dataclass
class ExchangeConfig:
    """Exchange-specific configuration with validation"""
    name: str
    api_key: str = field(default_factory=lambda: os.getenv("EXCHANGE_API_KEY", ""))
    api_secret: str = field(default_factory=lambda: os.getenv("EXCHANGE_API_SECRET", ""))
    sandbox: bool = field(default_factory=lambda: os.getenv("EXCHANGE_SANDBOX", "True") == "True")
    rate_limit: int = field(default_factory=lambda: int(os.getenv("EXCHANGE_RATE_LIMIT", "1000")))
    
    def validate(self) -> bool:
        """Validate exchange configuration"""
        if not self.api_key or not self.api_secret:
            logging.warning(f"Exchange {self.name}: Missing API credentials")
            return False
        return True

@dataclass
class FirebaseConfig:
    """Firebase configuration with credential validation"""
    project_id: str = field(default_factory=lambda: os.getenv("FIREBASE_PROJECT_ID", ""))
    credentials_path: str = field(default_factory=lambda: os.getenv("FIREBASE_CREDENTIALS_PATH", "./firebase-credentials.json"))
    
    def validate(self) -> bool:
        """Validate Firebase configuration"""
        if not os.path.exists(self.credentials_path):
            logging.error(f"Firebase credentials not found at {self.credentials_path}")
            return False
        return True

@dataclass
class TradingConfig:
    """Global trading configuration with risk limits"""
    max_position_size_usd: float = field(default_factory=lambda: float(os.getenv("MAX_POSITION_SIZE_USD", "10000")))
    max_daily_loss_pct: float = field(default_factory=lambda: float(os.getenv("MAX_DAILY_LOSS_PCT", "5.0")))
    max_open_positions: int = field(default_factory=lambda: int(os.getenv("MAX_OPEN_POSITIONS", "10")))
    cooloff_period_seconds: int = field(default_factory=lambda: int(os.getenv("COOLOFF_SECONDS", "300")))
    default_timeframe: str = field(default_factory=lambda: os.getenv("DEFAULT_TIMEFRAME", "1h"))
    
    def validate(self) -> bool:
        """Validate trading parameters"""
        if self.max_daily_loss_pct <= 0 or self.max_daily_loss_pct > 100:
            logging.error(f"Invalid max_daily_loss_pct: {self.max_daily_loss_pct}")
            return False
        return True

@dataclass
class ModelConfig:
    """Machine learning model configuration"""
    rl_learning_rate: float = field(default_factory=lambda: float(os.getenv("RL_LEARNING_RATE", "0.001")))
    rl_discount_factor: float = field(default_factory=lambda: float(os.getenv("RL_DISCOUNT_FACTOR", "0.99")))
    model_update_frequency: int = field(default_factory=lambda: int(os.getenv("MODEL_UPDATE_FREQ", "100")))
    min_samples_for_training: int = field(default_factory=lambda: int(os.getenv("MIN_TRAINING_SAMPLES", "1000")))
    
    def validate(self) -> bool:
        """Validate model parameters"""
        if not 0 < self.rl_learning_rate < 1:
            logging.error(f"Invalid learning rate: {self.rl_learning_rate}")
            return False
        return True

class ConfigManager:
    """Singleton configuration manager with validation"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize all configuration components"""
        self.exchange = ExchangeConfig(name="binance")
        self.firebase = FirebaseConfig()
        self.trading = TradingConfig()
        self.model = ModelConfig()
        self._validate_all()
        
    def _validate_all(self) -> bool:
        """Validate all configuration sections"""
        validations = [
            self.exchange.validate(),
            self.firebase.validate(),
            self.trading.validate(),
            self.model.validate()
        ]
        
        if not all(validations):
            logging.critical("Configuration validation failed")
            return False
        
        logging.info("All configurations validated successfully")
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary for logging"""
        return {
            "exchange": self.exchange.__dict__,
            "trading": self.trading.__dict__,
            "model": self.model.__dict__
        }

# Global configuration instance
config = ConfigManager()