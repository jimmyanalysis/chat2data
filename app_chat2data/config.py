import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-change-this-in-production'

    # File upload configuration
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

    # OpenAI configuration
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    OPENAI_MODEL = 'gpt-4o-mini'

    # MySQL configuration for storing imported CSV data
    MYSQL_HOST = os.environ.get('MYSQL_HOST', 'localhost')
    MYSQL_USER = os.environ.get('MYSQL_USER', 'root')
    MYSQL_PASSWORD = os.environ.get('MYSQL_PASSWORD', '')
    MYSQL_DATABASE = os.environ.get('MYSQL_DATABASE', 'data_analysis')

    # MySQL connection pool settings
    MYSQL_POOL_SIZE = int(os.environ.get('MYSQL_POOL_SIZE', '10'))
    MYSQL_MAX_OVERFLOW = int(os.environ.get('MYSQL_MAX_OVERFLOW', '20'))

    # LangChain settings
    LANGCHAIN_VERBOSE = os.environ.get('LANGCHAIN_VERBOSE', 'True').lower() == 'true'
    LANGCHAIN_TEMPERATURE = float(os.environ.get('LANGCHAIN_TEMPERATURE', '0.0'))