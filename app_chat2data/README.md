# Flask Data Analysis Chat Tool

An intelligent data analysis tool based on Flask and OpenAI that supports CSV file uploads and database connections for data analysis through a chat interface.

## Features

- 📊 **Multi-Data Source Support**: Supports CSV file uploads and MySQL database connections
- 💬 **Intelligent Chat Analysis**: Natural language data analysis based on OpenAI GPT models
- 🔒 **Security**: Database passwords not stored on server, session-based user data management
- 📈 **SQL Query Execution**: AI can generate and execute SQL queries to return results
- 🎨 **Modern UI**: Responsive design with elegant chat interface

## System Requirements

- Python 3.8+
- MySQL 5.7+ (if using database functionality)
- OpenAI API key

## Quick Start

### 1. Install Dependencies

```bash
# Clone project or create new directory
mkdir flask-data-analyzer
cd flask-data-analyzer

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

Create `.env` file:

```env
SECRET_KEY=your-super-secret-key-here
OPENAI_API_KEY=sk-your-openai-api-key-here
MYSQL_HOST=localhost
MYSQL_USER=root
MYSQL_PASSWORD=your-mysql-password
MYSQL_DATABASE=data_analysis
```

### 3. Create Directory Structure

```
flask-data-analyzer/
├── app.py                 # Main application file
├── config.py             # Configuration file
├── requirements.txt      # Dependencies list
├── .env                 # Environment variables
├── uploads/             # File upload directory
└── templates/           # HTML templates
    ├── index.html       # Homepage
    └── chat.html        # Chat page
```

### 4. Run Application

```bash
python app.py
```

Visit `http://localhost:5000` to start using.

## Usage Guide

### Data Upload Methods

#### Method 1: Upload CSV File
1. Click "Choose CSV File" on homepage
2. Select your CSV file (supports UTF-8 encoding)
3. Click "Upload File"
4. Automatically redirects to chat page after successful upload

#### Method 2: Connect to Database
1. Fill in database connection information:
   - Host address (e.g., localhost)
   - Username
   - Password
   - Database name
   - Table name
2. Click "Connect Database"
3. Automatically redirects to chat page after successful connection

### Chat Analysis Features

In the chat interface, you can ask in natural language:

**Basic Analysis:**
- "How many rows and columns does this dataset have?"
- "Show the first 5 rows of data"
- "What are the data types of each column?"

**Statistical Analysis:**
- "Calculate the average sales amount"
- "Find the 10 oldest people"
- "Count users by region"

**Data Queries:**
- "Find all users from New York"
- "Show records with sales amount greater than 1000"
- "Count orders by month"

**Visualization Suggestions:**
- "How to visualize this data?"
- "Recommend a chart type"
- "Which fields are suitable for scatter plots?"

## Technical Architecture

### Backend Architecture
- **Flask**: Web framework
- **Pandas**: Data processing
- **MySQL Connector**: Database connections
- **OpenAI API**: AI analysis
- **Session Management**: User data storage

### Frontend Design
- **Native JavaScript**: Interactive logic
- **CSS3**: Modern styling
- **Responsive Layout**: Adapts to various devices

### Security Measures
- File type validation
- SQL injection protection (basic version)
- Session data isolation
- Password non-storage principle

## API Endpoints

### File Upload
```http
POST /upload
Content-Type: multipart/form-data

Response:
{
  "success": true,
  "message": "File uploaded successfully!",
  "data_info": {
    "shape": [1000, 10],
    "columns": ["name", "age", "city"]
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

### Send Chat Message
```http
POST /send_message
Content-Type: application/json

{
  "message": "Show first 5 rows of data"
}

Response:
{
  "success": true,
  "response": "AI analysis result...",
  "timestamp": "14:30:25"
}
```

## Deployment Recommendations

### Development Environment
```bash
# Run directly
python app.py
```

### Production Environment

#### Using Gunicorn
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

#### Using Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

#### Nginx Configuration
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Troubleshooting

### Common Issues

**1. OpenAI API Error**
- Check if API key is correct
- Confirm account has sufficient balance
- Check network connection

**2. Database Connection Failed**
- Verify database service is running
- Check username and password
- Confirm database and table names exist

**3. File Upload Failed**
- Check file format (only supports CSV)
- Confirm file size doesn't exceed 16MB
- Check file encoding (UTF-8 recommended)

**4. Character Encoding Issues**
- The system now automatically detects file encoding
- Supports multiple encodings: UTF-8, GBK, GB2312, Big5, Latin1, CP1252, ISO-8859-1
- If automatic detection fails, saves the file with UTF-8 encoding
- For best results, use UTF-8 encoded CSV files

**5. File Reading Errors**
- Install chardet for better encoding detection: `pip install chardet`
- If problems persist, try converting your CSV to UTF-8 encoding first
- Check for special characters or binary data in the CSV file

### View Logs
```bash
# View application logs
tail -f app.log

# Enable debug mode
export FLASK_DEBUG=1
python app.py
```

## Extended Features

### Adding New Data Sources
1. Add new connection methods in `DatabaseManager` class
2. Modify frontend forms to support new parameters
3. Update data information storage format

### Integrating More AI Models
```python
# Add in config.py
CLAUDE_API_KEY = "your-claude-key"
GEMINI_API_KEY = "your-gemini-key"

# Add new methods in AIAnalyzer class
@staticmethod
def analyze_with_claude(message, data_info):
    # Claude API call logic
    pass
```

### Adding Data Visualization
```bash
pip install plotly matplotlib seaborn
```

Generate charts in chat responses:
```python
def generate_chart(data, chart_type):
    # Chart generation logic
    # Return chart HTML or base64 encoding
    pass
```

## Contributing Guidelines

1. Fork the project
2. Create feature branch
3. Submit changes
4. Push to branch
5. Create Pull Request

## License

MIT License

## Contact

For questions or suggestions, please submit an Issue or contact the developer.

---

**Note**: This is a demonstration project. For production use, please strengthen security measures including but not limited to:
- Strict SQL injection protection
- User authentication and authorization
- Encrypted data storage
- Rate limiting
- Complete error handling