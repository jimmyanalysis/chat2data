from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.utils import secure_filename
import pandas as pd
import mysql.connector
import os
import json
import uuid
import logging
import re
from datetime import datetime
from dateutil import parser as dateutil_parser
from config import Config

# LangChain imports
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'xlsx', 'xls'}


def detect_csv_encoding(file_path):
    """Detect CSV file encoding with multiple fallbacks"""
    encodings_to_try = [
        'utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'gb18030',
        'big5', 'latin1', 'cp1252', 'iso-8859-1', 'windows-1252'
    ]

    # Try chardet first
    try:
        import chardet
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)
            detected = chardet.detect(raw_data)
            if detected['confidence'] > 0.7:
                return detected['encoding']
    except ImportError:
        logger.warning("chardet not available, using fallback detection")
    except Exception as e:
        logger.warning(f"chardet detection failed: {e}")

    # Try encodings one by one
    for encoding in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                f.read(1000)  # Try to read first 1000 chars
            return encoding
        except UnicodeDecodeError:
            continue
        except Exception:
            continue

    return 'latin1'  # Last resort


def read_csv_safe(file_path):
    """Safely read CSV with encoding detection"""
    encoding = detect_csv_encoding(file_path)
    logger.info(f"Using encoding: {encoding}")

    try:
        df = pd.read_csv(file_path, encoding=encoding)
        return df, encoding
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        # Last resort: read with errors ignored
        df = pd.read_csv(file_path, encoding='latin1', errors='ignore')
        return df, 'latin1 (with errors ignored)'


def clean_column_names(columns):
    """Clean column names for MySQL compatibility"""
    clean_cols = []
    for i, col in enumerate(columns):
        # Remove/replace problematic characters
        clean_col = str(col).strip()
        clean_col = clean_col.replace(' ', '_').replace('-', '_')
        clean_col = clean_col.replace('(', '').replace(')', '')
        clean_col = clean_col.replace('[', '').replace(']', '')
        clean_col = clean_col.replace('.', '_').replace(',', '_')
        clean_col = clean_col.replace('/', '_').replace('\\', '_')
        clean_col = clean_col.replace('&', 'and').replace('%', 'percent')
        clean_col = clean_col.replace('#', 'num').replace('@', 'at')
        clean_col = clean_col.replace('$', 'dollar').replace('!', '')
        clean_col = clean_col.replace('?', '').replace('*', '')
        clean_col = clean_col.replace('+', 'plus').replace('=', 'eq')
        clean_col = clean_col.replace('<', 'lt').replace('>', 'gt')
        clean_col = clean_col.replace('|', '_').replace(';', '_')
        clean_col = clean_col.replace(':', '_').replace('"', '')
        clean_col = clean_col.replace("'", '').replace('`', '')

        # Remove non-ASCII characters
        clean_col = clean_col.encode('ascii', 'ignore').decode('ascii')

        # Ensure not empty
        if not clean_col or clean_col == '_':
            clean_col = f"column_{i + 1}"

        # Handle duplicates
        original_clean = clean_col
        counter = 1
        while clean_col in clean_cols:
            clean_col = f"{original_clean}_{counter}"
            counter += 1

        clean_cols.append(clean_col)

    return clean_cols


def smart_date_parser(date_string):
    """
    Smart date parser using dateutil.parser with fallbacks
    Returns standardized datetime string or None if parsing fails
    """
    if pd.isna(date_string) or str(date_string).strip() == '':
        return None

    date_str = str(date_string).strip()

    try:
        # Use dateutil parser - it handles most date formats automatically
        parsed_date = dateutil_parser.parse(date_str)
        # Return in MySQL-compatible format
        return parsed_date.strftime('%Y-%m-%d %H:%M:%S')
    except (ValueError, TypeError, dateutil_parser.ParserError):
        # If dateutil fails, try some common problematic formats manually

        # Handle dates with 0:00 time (your specific issue)
        if re.match(r'^\d{1,2}/\d{1,2}/\d{4}\s+0:00$', date_str):
            try:
                # Parse as MM/DD/YYYY format and add default time
                date_part = date_str.split()[0]
                parsed_date = datetime.strptime(date_part, '%m/%d/%Y')
                return parsed_date.strftime('%Y-%m-%d %H:%M:%S')
            except:
                pass

        # Handle dates without time
        if re.match(r'^\d{1,2}/\d{1,2}/\d{4}$', date_str):
            try:
                parsed_date = datetime.strptime(date_str, '%m/%d/%Y')
                return parsed_date.strftime('%Y-%m-%d %H:%M:%S')
            except:
                pass

        # Handle Excel-style dates with .0 decimals
        if re.match(r'^\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}\.0$', date_str):
            try:
                # Remove the .0 and parse
                clean_date = date_str.replace('.0', '')
                parsed_date = dateutil_parser.parse(clean_date)
                return parsed_date.strftime('%Y-%m-%d %H:%M:%S')
            except:
                pass

        # Last resort: try pandas to_datetime
        try:
            parsed_date = pd.to_datetime(date_str, errors='raise')
            return parsed_date.strftime('%Y-%m-%d %H:%M:%S')
        except:
            logger.warning(f"Failed to parse date: '{date_string}'")
            return None


def process_date_values(df):
    """Process and standardize date columns using smart date parser"""
    date_columns = []
    conversion_stats = {}

    for col in df.columns:
        # Check if column looks like a date column
        col_lower = col.lower()
        date_keywords = ['date', 'time', 'timestamp', 'created', 'updated', 'modified', 'birth', 'expire']

        looks_like_date = any(keyword in col_lower for keyword in date_keywords)

        # Also check if values look like dates
        if not looks_like_date:
            sample_values = df[col].dropna().head(10)
            if len(sample_values) > 0:
                date_like_count = 0
                for val in sample_values:
                    val_str = str(val).strip()
                    # Check for date-like patterns
                    if (re.search(r'\d{1,4}[/-]\d{1,2}[/-]\d{2,4}', val_str) or
                            re.search(r'\d{4}-\d{2}-\d{2}', val_str) or
                            re.search(r'\d{1,2}:\d{2}', val_str)):
                        date_like_count += 1

                if date_like_count / len(sample_values) > 0.6:
                    looks_like_date = True

        if looks_like_date:
            logger.info(f"Processing potential date column: {col}")
            original_values = df[col].copy()

            # Use smart date parser for all values
            converted_values = []
            successful_conversions = 0
            total_non_null = original_values.notna().sum()

            for value in original_values:
                parsed_date = smart_date_parser(value)
                if parsed_date:
                    converted_values.append(parsed_date)
                    successful_conversions += 1
                else:
                    # Keep original value as string if parsing fails
                    converted_values.append(str(value) if pd.notna(value) else '')

            # Calculate success rate
            success_rate = successful_conversions / total_non_null if total_non_null > 0 else 0

            if success_rate > 0.7:  # If more than 70% successful
                df[col] = converted_values
                date_columns.append(col)
                conversion_stats[col] = {
                    'method': 'dateutil_parser',
                    'success_rate': success_rate,
                    'converted': successful_conversions,
                    'total': total_non_null
                }
                logger.info(f"Successfully converted '{col}': {successful_conversions}/{total_non_null} values")
            else:
                # Keep original values if conversion rate is too low
                logger.warning(f"Low conversion rate for '{col}' ({success_rate:.2f}), keeping as text")
                df[col] = original_values.astype(str).fillna('')
                conversion_stats[col] = {
                    'method': 'failed',
                    'success_rate': success_rate,
                    'converted': successful_conversions,
                    'total': total_non_null
                }

    return df, date_columns, conversion_stats


def infer_mysql_type(series, column_name):
    """Infer MySQL column type from pandas series with improved date handling"""
    if series.empty:
        return "TEXT"

    # Remove nulls for type checking
    non_null_series = series.dropna()
    if non_null_series.empty:
        return "TEXT"

    # Check data types
    if non_null_series.dtype in ['int64', 'int32', 'int16', 'int8']:
        return "INT"
    elif non_null_series.dtype in ['float64', 'float32']:
        return "DECIMAL(15,4)"
    else:
        # For all other types including dates, use TEXT to avoid format issues
        # This is safer than trying to use DATETIME columns
        try:
            max_length = non_null_series.astype(str).str.len().max()
            if max_length <= 50:
                return "VARCHAR(100)"
            elif max_length <= 255:
                return "VARCHAR(500)"
            elif max_length <= 65535:
                return "TEXT"
            else:
                return "LONGTEXT"
        except:
            return "TEXT"


class DatabaseManager:
    @staticmethod
    def get_connection():
        """Create database connection to default MySQL database"""
        try:
            connection = mysql.connector.connect(
                host=app.config['MYSQL_HOST'],
                user=app.config['MYSQL_USER'],
                password=app.config['MYSQL_PASSWORD'],
                database=app.config['MYSQL_DATABASE'],
                charset='utf8mb4'
            )
            return connection
        except mysql.connector.Error as e:
            logger.error(f"Database connection error: {e}")
            return None

    @staticmethod
    def csv_to_mysql(csv_file, connection):
        """Import CSV data to MySQL with improved error handling"""
        try:
            # Read CSV with encoding detection
            df, encoding_used = read_csv_safe(csv_file)
            logger.info(f"Read CSV with {len(df)} rows and {len(df.columns)} columns using {encoding_used}")

            if df.empty:
                return {"success": False, "error": "CSV file is empty"}

            # Clean column names
            original_columns = df.columns.tolist()
            clean_columns = clean_column_names(df.columns)
            df.columns = clean_columns

            logger.info(f"Column mapping: {dict(zip(original_columns, clean_columns))}")

            # IMPORTANT: Process date columns BEFORE creating the table
            df_processed, date_columns, conversion_stats = process_date_values(df)

            if date_columns:
                logger.info(f"Successfully processed date columns: {date_columns}")
                for col, stats in conversion_stats.items():
                    if stats['success_rate'] > 0:
                        logger.info(
                            f"  {col}: {stats['method']} - {stats['converted']}/{stats['total']} values converted")
            else:
                logger.info("No date columns found or processed")

            # Generate table name
            table_name = f"ana_{uuid.uuid4().hex[:8]}"

            # Create table
            cursor = connection.cursor()

            # Build CREATE TABLE statement using processed DataFrame
            column_defs = []
            for col in df_processed.columns:
                col_type = infer_mysql_type(df_processed[col], col)
                column_defs.append(f"`{col}` {col_type}")

            create_sql = f"""
            CREATE TABLE {table_name} (
                id INT AUTO_INCREMENT PRIMARY KEY,
                {', '.join(column_defs)}
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """

            cursor.execute(create_sql)
            logger.info(f"Created table: {table_name}")

            # Insert data in chunks using processed DataFrame
            chunk_size = 1000
            total_inserted = 0

            insert_sql = f"""
            INSERT INTO {table_name} ({', '.join([f'`{col}`' for col in df_processed.columns])}) 
            VALUES ({', '.join(['%s'] * len(df_processed.columns))})
            """

            for start_idx in range(0, len(df_processed), chunk_size):
                end_idx = min(start_idx + chunk_size, len(df_processed))
                chunk = df_processed.iloc[start_idx:end_idx]

                # Prepare data with careful handling of different types
                data_rows = []
                for _, row in chunk.iterrows():
                    processed_row = []
                    for col_name, val in zip(df_processed.columns, row):
                        if pd.isna(val) or val == '' or str(val).strip() == '':
                            processed_row.append(None)
                        elif isinstance(val, (int, float)) and not pd.isna(val):
                            # Keep numeric values as they are
                            processed_row.append(val)
                        else:
                            # For all other values including processed dates, convert to string
                            str_val = str(val).strip()
                            # Truncate very long strings to avoid MySQL errors
                            if len(str_val) > 65535:
                                str_val = str_val[:65535]
                            processed_row.append(str_val)
                    data_rows.append(tuple(processed_row))

                try:
                    cursor.executemany(insert_sql, data_rows)
                    total_inserted += len(data_rows)
                    logger.info(f"Inserted chunk {start_idx}-{end_idx} ({total_inserted}/{len(df_processed)})")
                except mysql.connector.Error as e:
                    logger.error(f"Error inserting chunk {start_idx}-{end_idx}: {e}")
                    # Try inserting row by row to identify problematic rows
                    successful_rows = 0
                    for i, row_data in enumerate(data_rows):
                        try:
                            cursor.execute(insert_sql, row_data)
                            successful_rows += 1
                        except mysql.connector.Error as row_error:
                            logger.error(f"Error in row {start_idx + i}: {row_error}")
                            logger.error(f"Problematic data: {dict(zip(df_processed.columns, row_data))}")
                            continue
                    total_inserted += successful_rows
                    logger.info(f"Inserted {successful_rows}/{len(data_rows)} rows from chunk {start_idx}-{end_idx}")

            connection.commit()
            cursor.close()

            logger.info(f"Successfully imported {len(df_processed)} rows to table {table_name}")
            return {
                "success": True,
                "table_name": table_name,
                "row_count": len(df_processed),
                "encoding": encoding_used,
                "columns": clean_columns,
                "date_columns": date_columns,
                "conversion_stats": conversion_stats
            }

        except Exception as e:
            logger.error(f"CSV import error: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    @staticmethod
    def execute_query(connection, query, params=None):
        """Safely execute database query"""
        try:
            cursor = connection.cursor()
            cursor.execute(query, params or ())

            if query.strip().upper().startswith('SELECT'):
                results = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                cursor.close()
                return results, columns
            else:
                connection.commit()
                affected_rows = cursor.rowcount
                cursor.close()
                return affected_rows, None
        except Exception as e:
            logger.error(f"Query execution error: {e}")
            return None, str(e)

    @staticmethod
    def get_table_schema(connection, table_name):
        """Get table schema information"""
        try:
            cursor = connection.cursor()
            cursor.execute(f"DESCRIBE {table_name}")
            schema = cursor.fetchall()
            cursor.close()
            return schema
        except Exception as e:
            logger.error(f"Schema fetch error: {e}")
            return None


class LangChainAnalyzer:
    def __init__(self):
        self.llm = OpenAI(
            openai_api_key=app.config['OPENAI_API_KEY'],
            temperature=0,
            model_name="gpt-3.5-turbo-instruct"
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

    def create_sql_agent(self, connection_string, table_name):
        """Create SQL agent with LangChain"""
        try:
            db = SQLDatabase.from_uri(connection_string)
            toolkit = SQLDatabaseToolkit(db=db, llm=self.llm)

            agent_executor = create_sql_agent(
                llm=self.llm,
                toolkit=toolkit,
                verbose=True,
                agent_type="zero-shot-react-description",
                memory=self.memory,
                handle_parsing_errors=True
            )

            return agent_executor
        except Exception as e:
            logger.error(f"SQL Agent creation error: {e}")
            return None

    def analyze_with_langchain(self, user_message, table_name, connection_info):
        """Analyze data using LangChain SQL agent"""
        try:
            # Build connection string
            if connection_info['type'] == 'default':
                connection_string = f"mysql+pymysql://{app.config['MYSQL_USER']}:{app.config['MYSQL_PASSWORD']}@{app.config['MYSQL_HOST']}/{app.config['MYSQL_DATABASE']}"
            else:
                connection_string = f"mysql+pymysql://{connection_info['user']}:{connection_info['password']}@{connection_info['host']}/{connection_info['database']}"

            # Create SQL agent
            agent = self.create_sql_agent(connection_string, table_name)
            if not agent:
                return "Failed to create SQL analysis agent."

            # Enhanced prompt
            enhanced_prompt = f"""
            You are analyzing data from table '{table_name}'. 
            User question: {user_message}

            Please:
            1. Write and execute appropriate SQL queries to answer the question
            2. Provide clear insights based on the results
            3. If creating visualizations, suggest appropriate chart types
            4. Explain your analysis process

            Focus on being helpful and providing actionable insights.
            """

            # Execute analysis
            result = agent.run(enhanced_prompt)
            return result

        except Exception as e:
            logger.error(f"LangChain analysis error: {e}")
            return f"Analysis failed: {str(e)}"


# Initialize LangChain analyzer
langchain_analyzer = LangChainAnalyzer()


# Routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and import to MySQL"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'No file provided'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No file selected'})

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Get database connection
            connection = DatabaseManager.get_connection()
            if not connection:
                return jsonify({'success': False, 'message': 'Database connection failed'})

            try:
                # Import CSV to MySQL
                result = DatabaseManager.csv_to_mysql(filepath, connection)

                if not result["success"]:
                    return jsonify({'success': False, 'message': f'CSV import failed: {result["error"]}'})

                # Get table schema for verification
                schema = DatabaseManager.get_table_schema(connection, result["table_name"])
                schema_columns = [col[0] for col in schema] if schema else []

                # Store data information in session
                session['data_info'] = {
                    'type': 'csv_import',
                    'filename': filename,
                    'table_name': result["table_name"],
                    'shape': [result["row_count"], len(result["columns"])],
                    'columns': result["columns"],
                    'encoding': result["encoding"],
                    'date_columns': result.get("date_columns", [])
                }
                session['connection_info'] = {
                    'type': 'default',
                    'host': app.config['MYSQL_HOST'],
                    'user': app.config['MYSQL_USER'],
                    'password': app.config['MYSQL_PASSWORD'],
                    'database': app.config['MYSQL_DATABASE']
                }

                # Get sample data
                sample_results, sample_columns = DatabaseManager.execute_query(
                    connection, f"SELECT * FROM {result['table_name']} LIMIT 5"
                )
                session['sample_data'] = [dict(zip(sample_columns, row)) for row in
                                          sample_results] if sample_results else []

                connection.close()

                # Clean up uploaded file
                os.remove(filepath)

                # Build success message
                success_msg = f'CSV imported successfully! Table {result["table_name"]} created with {result["row_count"]} rows (Encoding: {result["encoding"]})'
                if result.get("date_columns"):
                    success_msg += f'. Processed date columns: {", ".join(result["date_columns"])}'

                return jsonify({
                    'success': True,
                    'message': success_msg,
                    'data_info': session['data_info']
                })

            except Exception as e:
                connection.close()
                return jsonify({'success': False, 'message': f'Import failed: {str(e)}'})

    except Exception as e:
        logger.error(f"File upload error: {e}")
        return jsonify({'success': False, 'message': f'File processing failed: {str(e)}'})

    return jsonify({'success': False, 'message': 'File format not supported'})


@app.route('/chat')
def chat():
    """Chat page"""
    if 'data_info' not in session:
        return redirect(url_for('index'))
    return render_template('chat.html', data_info=session['data_info'])


@app.route('/send_message', methods=['POST'])
def send_message():
    """Handle chat messages with LangChain analysis"""
    try:
        if 'data_info' not in session:
            return jsonify({'success': False, 'message': 'Please upload data or connect to database first'})

        user_message = request.json.get('message', '').strip()
        if not user_message:
            return jsonify({'success': False, 'message': 'Message cannot be empty'})

        data_info = session['data_info']
        connection_info = session.get('connection_info', {})
        table_name = data_info['table_name']

        # Use LangChain for analysis
        ai_response = langchain_analyzer.analyze_with_langchain(
            user_message,
            table_name,
            connection_info
        )

        return jsonify({
            'success': True,
            'response': ai_response,
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'table_name': table_name
        })

    except Exception as e:
        logger.error(f"Message processing error: {e}")
        return jsonify({'success': False, 'message': f'Processing failed: {str(e)}'})


@app.route('/get_data_summary')
def get_data_summary():
    """Get data summary"""
    if 'data_info' not in session:
        return jsonify({'success': False, 'message': 'No data available'})

    return jsonify({
        'success': True,
        'data_info': session['data_info'],
        'sample_data': session.get('sample_data', [])[:5]
    })


@app.route('/cleanup_table', methods=['POST'])
def cleanup_table():
    """Clean up temporary table"""
    try:
        if 'data_info' not in session:
            return jsonify({'success': False, 'message': 'No data to clean up'})

        data_info = session['data_info']
        if data_info['type'] == 'csv_import':
            connection = DatabaseManager.get_connection()
            if connection:
                try:
                    cursor = connection.cursor()
                    cursor.execute(f"DROP TABLE IF EXISTS {data_info['table_name']}")
                    connection.commit()
                    cursor.close()
                    connection.close()

                    # Clear session
                    session.pop('data_info', None)
                    session.pop('connection_info', None)
                    session.pop('sample_data', None)

                    return jsonify({'success': True, 'message': 'Table cleaned up successfully'})
                except Exception as e:
                    connection.close()
                    return jsonify({'success': False, 'message': f'Cleanup failed: {str(e)}'})

        return jsonify({'success': True, 'message': 'No cleanup needed'})

    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        return jsonify({'success': False, 'message': f'Cleanup failed: {str(e)}'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)