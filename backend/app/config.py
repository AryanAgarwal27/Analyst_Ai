from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DEBUG: bool = True
    PROJECT_NAME: str = "Analyst AI"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

settings = Settings() 