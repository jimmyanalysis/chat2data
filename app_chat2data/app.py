from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.utils import secure_filename
import pandas as pd
import mysql.connector
import os
import json
import openai
from datetime import datetime
import uuid
import logging
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Set OpenAI API key
openai.api_key = app.config['OPENAI_API_KEY']


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'xlsx', 'xls'}


class DatabaseManager:
    @staticmethod
    def get_connection(host, user, password, database):
        """Create database connection"""
        try:
            connection = mysql.connector.connect(
                host=host,
                user=user,
                password=password,
                database=database,
                charset='utf8mb4'
            )
            return connection
        except mysql.connector.Error as e:
            logger.error(f"Database connection error: {e}")
            return None

    @staticmethod
    def csv_to_mysql(csv_file, connection):
        """Import CSV data to MySQL"""
        try:
            # Read CSV file
            df = pd.read_csv(csv_file, encoding='utf-8')

            # Clean column names
            df.columns = [col.strip().replace(' ', '_').replace('-', '_') for col in df.columns]

            # Generate unique table name
            table_name = f"user_data_{uuid.uuid4().hex[:8]}"

            cursor = connection.cursor()

            # Generate CREATE TABLE statement
            columns = []
            for col in df.columns:
                columns.append(f"`{col}` TEXT")

            create_table_sql = f"CREATE TABLE {table_name} ({', '.join(columns)})"
            cursor.execute(create_table_sql)

            # Batch insert data
            insert_sql = f"INSERT INTO {table_name} ({', '.join([f'`{col}`' for col in df.columns])}) VALUES ({', '.join(['%s'] * len(df.columns))})"

            data_to_insert = []
            for index, row in df.iterrows():
                data_to_insert.append(tuple(str(val) if pd.notna(val) else None for val in row))

            cursor.executemany(insert_sql, data_to_insert)
            connection.commit()
            cursor.close()

            return table_name, len(df)
        except Exception as e:
            logger.error(f"CSV import error: {e}")
            return None, str(e)

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


class AIAnalyzer:
    @staticmethod
    def analyze_with_openai(user_message, data_info, sample_data):
        """Analyze data using OpenAI"""
        try:
            # Build system prompt
            system_prompt = f"""
            You are a professional data analysis assistant. The user has a dataset with the following information:

            Basic data information:
            - Number of rows: {data_info.get('shape', ['Unknown', 'Unknown'])[0]}
            - Number of columns: {data_info.get('shape', ['Unknown', 'Unknown'])[1]}
            - Column names: {', '.join(data_info.get('columns', []))}

            Sample data (first 3 rows):
            {json.dumps(sample_data[:3], ensure_ascii=False, indent=2)}

            Please provide professional data analysis advice based on the user's question. If database queries are needed, provide corresponding SQL statements.
            Use {data_info.get('table_name', 'TABLE_NAME')} as the table name in SQL statements.

            Requirements:
            1. Provide clear, professional analysis advice
            2. If SQL queries are involved, give complete SQL statements
            3. Explain analysis approach and methods
            4. If possible, provide data visualization suggestions
            """

            response = openai.ChatCompletion.create(
                model=app.config['OPENAI_MODEL'],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=1500,
                temperature=0.7
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"OpenAI analysis error: {e}")
            return f"AI analysis service temporarily unavailable: {str(e)}"


# Route definitions
@app.route('/')
def index():
    """Homepage"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
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

            # Read file to get basic information
            if filename.endswith('.csv'):
                # Try multiple encodings for CSV
                encodings_to_try = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'big5', 'latin1', 'cp1252', 'iso-8859-1']
                df = None
                encoding_used = None

                for encoding in encodings_to_try:
                    try:
                        df = pd.read_csv(filepath, encoding=encoding)
                        encoding_used = encoding
                        break
                    except UnicodeDecodeError:
                        continue
                    except Exception:
                        continue

                if df is None:
                    # Last resort: read with error handling
                    df = pd.read_csv(filepath, encoding='utf-8', errors='ignore')
                    encoding_used = 'utf-8 (with errors ignored)'

            else:  # Excel files
                df = pd.read_excel(filepath)
                encoding_used = 'Excel format'

            # Store data information in session
            session['data_info'] = {
                'type': 'file',
                'filename': filename,
                'filepath': filepath,
                'shape': list(df.shape),
                'columns': df.columns.tolist(),
                'encoding': encoding_used
            }
            session['sample_data'] = df.head(5).to_dict('records')

            return jsonify({
                'success': True,
                'message': f'File uploaded successfully! Data shape: {df.shape[0]} rows × {df.shape[1]} columns (Encoding: {encoding_used})',
                'data_info': session['data_info']
            })

    except Exception as e:
        logger.error(f"File upload error: {e}")
        return jsonify({'success': False, 'message': f'File processing failed: {str(e)}'})

    return jsonify({'success': False, 'message': 'File format not supported'})


@app.route('/connect_db', methods=['POST'])
def connect_database():
    """Connect to user database"""
    try:
        data = request.json
        host = data.get('host')
        user = data.get('user')
        password = data.get('password')
        database = data.get('database')
        table = data.get('table')

        # Validate input
        if not all([host, user, database, table]):
            return jsonify({'success': False, 'message': 'Please fill in complete database information'})

        connection = DatabaseManager.get_connection(host, user, password, database)
        if not connection:
            return jsonify({'success': False, 'message': 'Database connection failed'})

        try:
            # Get table information
            results, columns = DatabaseManager.execute_query(
                connection, f"SELECT * FROM {table} LIMIT 5"
            )

            row_count_result, _ = DatabaseManager.execute_query(
                connection, f"SELECT COUNT(*) FROM {table}"
            )
            row_count = row_count_result[0][0] if row_count_result else 0

            # Store data information in session
            session['data_info'] = {
                'type': 'database',
                'table_name': table,
                'shape': [row_count, len(columns)],
                'columns': columns
            }
            session['sample_data'] = [dict(zip(columns, row)) for row in results]
            session['db_config'] = {
                'host': host,
                'user': user,
                'password': password,
                'database': database
            }

            connection.close()

            return jsonify({
                'success': True,
                'message': f'Database connected successfully! Table {table} contains {row_count} rows of data',
                'data_info': session['data_info']
            })

        except Exception as e:
            connection.close()
            return jsonify({'success': False, 'message': f'Table query failed: {str(e)}'})

    except Exception as e:
        logger.error(f"Database connection error: {e}")
        return jsonify({'success': False, 'message': f'Connection failed: {str(e)}'})


@app.route('/chat')
def chat():
    """Chat page"""
    if 'data_info' not in session:
        return redirect(url_for('index'))
    return render_template('chat.html', data_info=session['data_info'])


@app.route('/send_message', methods=['POST'])
def send_message():
    """Handle chat messages"""
    try:
        if 'data_info' not in session:
            return jsonify({'success': False, 'message': 'Please upload data or connect to database first'})

        user_message = request.json.get('message', '').strip()
        if not user_message:
            return jsonify({'success': False, 'message': 'Message cannot be empty'})

        data_info = session['data_info']
        sample_data = session.get('sample_data', [])

        # Use OpenAI for analysis
        ai_response = AIAnalyzer.analyze_with_openai(user_message, data_info, sample_data)

        # If database type and response contains SQL, try to execute query
        if (data_info['type'] == 'database' and
                any(keyword in ai_response.upper() for keyword in ['SELECT', 'FROM', 'WHERE'])):

            try:
                db_config = session.get('db_config')
                connection = DatabaseManager.get_connection(**db_config)

                if connection:
                    # Simple SQL extraction (real projects need stricter parsing)
                    import re
                    sql_pattern = r'(SELECT.*?;|SELECT.*?(?=\n\n|$))'
                    sql_matches = re.findall(sql_pattern, ai_response, re.IGNORECASE | re.DOTALL)

                    if sql_matches:
                        sql_query = sql_matches[0].strip()
                        if sql_query.endswith(';'):
                            sql_query = sql_query[:-1]

                        results, columns = DatabaseManager.execute_query(connection, sql_query)

                        if results is not None:
                            ai_response += f"\n\n📊 Query Results:\n"
                            ai_response += f"Returned {len(results)} rows of data\n\n"

                            # Format display results (limit display rows)
                            display_rows = min(10, len(results))
                            for i, row in enumerate(results[:display_rows]):
                                row_data = dict(zip(columns, row))
                                ai_response += f"Row {i + 1}: {row_data}\n"

                            if len(results) > display_rows:
                                ai_response += f"... {len(results) - display_rows} more rows"

                    connection.close()

            except Exception as e:
                ai_response += f"\n\n⚠️ SQL execution error: {str(e)}"

        return jsonify({
            'success': True,
            'response': ai_response,
            'timestamp': datetime.now().strftime('%H:%M:%S')
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


@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'message': 'Page not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'message': 'Internal server error'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)