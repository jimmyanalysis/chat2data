# Flask Data Analysis Chat Tool with LangChain

An intelligent data analysis tool built with Flask, LangChain, and OpenAI that supports CSV file uploads and database connections for conversational data analysis through a natural language chat interface.

## Key Features

- **Smart CSV Processing**: Automatic encoding detection and intelligent handling of various date formats
- **LangChain SQL Agent**: Powered by LangChain for intelligent SQL query generation and execution
- **Multi-Source Support**: CSV file uploads and MySQL database connections
- **Natural Language Analysis**: Conversational data analysis using OpenAI GPT models
- **Secure Architecture**: Session-based data management, no server-side password storage
- **Modern Interface**: Responsive design with elegant chat interaction

## System Requirements

- Python 3.8+
- MySQL 5.7+ (for database functionality)
- OpenAI API key

## Quick Start

### 1. Install Dependencies

```bash
# Clone the project or create a new directory
mkdir flask-data-analyzer
cd flask-data-analyzer

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Environment Setup

Create a `.env` file:

```env
SECRET_KEY=your-super-secret-key-here
OPENAI_API_KEY=sk-your-openai-api-key-here

# MySQL configuration (for storing imported CSV data)
MYSQL_HOST=localhost
MYSQL_USER=root
MYSQL_PASSWORD=your-mysql-password
MYSQL_DATABASE=data_analysis

# LangChain configuration (optional)
LANGCHAIN_VERBOSE=True
LANGCHAIN_TEMPERATURE=0.0
```

### 3. Initialize Database

```bash
# Run database initialization script
python database_init.py
```

### 4. Start Application

```bash
python simplified_app.py
```

Visit `http://localhost:5000` to begin using the application.

## Project Structure

```
flask-data-analyzer/
├── simplified_app.py        # Main application file
├── config.py               # Configuration settings
├── database_init.py        # Database initialization script
├── requirements.txt        # Python dependencies
├── .env                   # Environment variables
├── uploads/               # File upload directory
├── templates/             # HTML templates
│   ├── index.html         # Homepage template
│   └── chat.html          # Chat interface template

```

## Dependencies

### Core Requirements

```txt
Flask==2.3.3
pandas==2.0.3
mysql-connector-python==8.1.0
python-dotenv==1.0.0
Werkzeug==2.3.7
openai==0.28.1

# LangChain dependencies
langchain==0.0.340
langchain-experimental==0.0.45
SQLAlchemy==2.0.23
PyMySQL==1.1.0

# Date parsing and encoding detection
python-dateutil==2.8.2
chardet==5.2.0

# Additional utilities
openpyxl==3.1.2
xlrd==2.0.1
```

## Usage Guide

### Data Upload Methods

#### Method 1: CSV File Upload
1. Click "Choose CSV File" on the homepage
2. Select your CSV file (supports multiple encodings)
3. Click "Upload File"
4. System automatically processes and imports data to MySQL
5. Redirects to chat interface upon successful upload

#### Method 2: Database Connection
1. Fill in database connection details:
   - Host address (e.g., localhost)
   - Username and password
   - Database and table name
2. Click "Connect Database"
3. Redirects to chat interface upon successful connection

### Chat Analysis Features

The chat interface supports natural language queries:

**Basic Analysis:**
- "How many rows and columns are in this dataset?"
- "Show me the first 10 rows"
- "What are the data types of each column?"
- "Check for missing values"

**Statistical Analysis:**
- "Calculate the average sales amount"
- "Find the top 10 customers by revenue"
- "Show sales trends by month"
- "What's the distribution of ages?"

**Data Queries:**
- "Find all orders from New York"
- "Show records where sales > 1000"
- "Count orders by region"
- "Who are the most active customers?"

**Data Quality:**
- "Find duplicate records"
- "Check for outliers in the price column"
- "Show null values by column"

## Advanced Features

### Smart Date Processing

The system automatically handles various date formats:
- US format: `2/24/2003 0:00`, `12/31/2023 23:59`
- ISO format: `2003-02-24 00:00:00`
- European format: `24/02/2003`
- Text format: `Feb 24, 2003`

All dates are automatically converted to MySQL-compatible format (`YYYY-MM-DD HH:MM:SS`).

### Encoding Detection

Supports multiple file encodings:
- UTF-8, UTF-8 with BOM
- Chinese: GBK, GB2312, GB18030, Big5
- Western European: Latin1, CP1252, ISO-8859-1
- Automatic fallback mechanisms

### LangChain Integration

- Intelligent SQL query generation
- Context-aware conversations
- Error handling and query optimization
- Natural language to SQL translation

## API Endpoints

### File Upload
```http
POST /upload
Content-Type: multipart/form-data

Response:
{
  "success": true,
  "message": "CSV imported successfully! Table ana_12345678 created with 1000 rows",
  "data_info": {
    "table_name": "ana_12345678",
    "shape": [1000, 10],
    "columns": ["name", "age", "city"],
    "encoding": "utf-8",
    "date_columns": ["order_date", "created_at"]
  }
}
```

### Database Connection
```http
POST /connect_db
Content-Type: application/json

{
  "host": "localhost",
  "user": "root",
  "password": "password",
  "database": "test_db",
  "table": "users"
}
```

### Chat Message
```http
POST /send_message
Content-Type: application/json

{
  "message": "Show me sales trends over time"
}

Response:
{
  "success": true,
  "response": "Based on the data analysis...",
  "timestamp": "14:30:25",
  "table_name": "ana_12345678"
}
```

### Data Summary
```http
GET /get_data_summary

Response:
{
  "success": true,
  "data_info": {
    "type": "csv_import",
    "shape": [1000, 10],
    "columns": ["id", "name", "age"]
  },
  "sample_data": [...]
}
```

## Troubleshooting

### Common Issues

**1. CSV Import Failures**
```bash
# Debug CSV encoding and date issues
python debug_csv_import.py your_file.csv

# Preprocess problematic CSV files
python csv_preprocessor.py your_file.csv fixed_file.csv
```

**2. Date Format Errors**
The system handles most date formats automatically. For persistent issues:
- Use the CSV preprocessor tool
- Convert dates to ISO format (YYYY-MM-DD) before upload
- Check for non-standard date separators

**3. Encoding Issues**
```bash
# Install additional encoding support
pip install chardet

# Use encoding detection tool
python tools/encoding_detector.py your_file.csv
```

**4. Database Connection Problems**
- Verify MySQL service is running
- Check credentials and network connectivity
- Ensure the target database exists
- Confirm user has necessary permissions

**5. OpenAI API Issues**
- Verify API key is correct and active
- Check account balance and rate limits
- Ensure network connectivity

### Debug Mode

```bash
# Enable detailed logging
export FLASK_DEBUG=1
python simplified_app.py

# View application logs in console
```

## Deployment

### Development
```bash
python simplified_app.py
```

### Production with Gunicorn
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 simplified_app:app
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "simplified_app:app"]
```

### Nginx Configuration
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
    
    client_max_body_size 16M;
}
```

## Security Considerations

### For Production Use

1. **Input Validation**
   - Implement strict file type validation
   - Add file size limits
   - Sanitize user inputs

2. **Database Security**
   - Use connection pooling
   - Implement proper SQL injection protection
   - Regular backup procedures

3. **API Security**
   - Add rate limiting
   - Implement user authentication
   - Use HTTPS in production

4. **Data Privacy**
   - Implement data retention policies
   - Add data encryption for sensitive information
   - Regular cleanup of temporary tables

## Extension Ideas

### Adding New Data Sources
```python
# Extend DatabaseManager class
@staticmethod
def connect_postgresql(connection_params):
    # PostgreSQL connection logic
    pass

@staticmethod  
def connect_mongodb(connection_params):
    # MongoDB connection logic
    pass
```

### Custom Analysis Functions
```python
# Add specialized analysis methods
def statistical_analysis(df, column):
    # Custom statistical calculations
    pass

def time_series_analysis(df, date_col, value_col):
    # Time series specific analysis
    pass
```

### Data Visualization
```bash
pip install plotly matplotlib seaborn
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

MIT License

## Support

For issues, questions, or feature requests, please:
1. Check the troubleshooting section
2. Search existing issues
3. Create a new issue with detailed information

---

**Note**: This tool is designed for data analysis and exploration. For production environments with sensitive data, implement additional security measures including user authentication, data encryption, and comprehensive audit logging.