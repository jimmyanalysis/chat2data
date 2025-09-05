from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_from_directory
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
from chart_generator import ChartGeneratorTool, create_chart_tool


# 修复后的LangChain imports - 使用最新版本
try:
    from langchain.agents import create_sql_agent
    from langchain.agents.agent_toolkits import SQLDatabaseToolkit
    from langchain_community.utilities import SQLDatabase
    from langchain_openai import OpenAI  # 新版本的导入方式
    from langchain.memory import ConversationBufferMemory
except ImportError:
    # 如果新版本不可用，尝试旧版本
    from langchain.agents import create_sql_agent
    from langchain.agents.agent_toolkits import SQLDatabaseToolkit
    from langchain.sql_database import SQLDatabase
    from langchain.llms import OpenAI
    from langchain.memory import ConversationBufferMemory

# 添加图表生成工具导入
from chart_generator import ChartGeneratorTool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def get_base_url():
    """根据请求来源自动确定base URL"""
    if request:
        # 检查是否通过代理访问（生产环境）
        if request.headers.get('X-Forwarded-Host') or '/chat2data/' in request.path:
            return '/chat2data'
        # 本地开发环境
        return ''
    return ''


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
        try:
            # 检查OpenAI API Key
            if not app.config.get('OPENAI_API_KEY'):
                logger.error("OPENAI_API_KEY not found in config")
                raise ValueError("OPENAI_API_KEY is required")

            logger.info("Initializing OpenAI LLM...")

            # 尝试使用新版本的OpenAI
            try:
                from langchain_openai import OpenAI
                self.llm = OpenAI(
                    api_key=app.config['OPENAI_API_KEY'],
                    temperature=0,
                    model="gpt-3.5-turbo-instruct"
                )
                logger.info("Successfully initialized OpenAI with langchain_openai")
            except ImportError:
                logger.info("langchain_openai not available, trying legacy import...")
                from langchain.llms import OpenAI
                self.llm = OpenAI(
                    openai_api_key=app.config['OPENAI_API_KEY'],
                    temperature=0,
                    model_name="gpt-3.5-turbo-instruct"
                )
                logger.info("Successfully initialized OpenAI with legacy import")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI: {e}")
                # 尝试最基本的方式
                from langchain.llms import OpenAI
                self.llm = OpenAI(
                    openai_api_key=app.config['OPENAI_API_KEY'],
                    temperature=0
                )
                logger.info("Initialized OpenAI with basic configuration")

        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise

        try:
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            logger.info("Successfully initialized ConversationBufferMemory")
        except Exception as e:
            logger.error(f"Failed to initialize memory: {e}")
            self.memory = None

    def create_sql_agent(self, connection_string, table_name):
        """Create SQL agent with properly integrated chart tool"""
        try:
            logger.info(f"Creating SQL agent for table: {table_name}")

            # Create SQLDatabase
            try:
                from langchain_community.utilities import SQLDatabase
            except ImportError:
                from langchain.sql_database import SQLDatabase

            db = SQLDatabase.from_uri(connection_string)

            # Create toolkit and get SQL tools
            try:
                from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
            except ImportError:
                from langchain.agents.agent_toolkits import SQLDatabaseToolkit

            toolkit = SQLDatabaseToolkit(db=db, llm=self.llm)
            sql_tools = toolkit.get_tools()

            # Create chart tool
            base_url = get_base_url()
            from chart_generator import create_chart_tool
            chart_tool = create_chart_tool(DatabaseManager, base_url)

            # Combine all tools
            all_tools = sql_tools + [chart_tool]

            logger.info(f"All available tools: {[tool.name for tool in all_tools]}")

            # Create agent with all tools
            try:
                from langchain.agents import create_react_agent, AgentExecutor
                from langchain import hub

                # Get a React prompt
                try:
                    prompt = hub.pull("hwchase17/react")
                except:
                    # Fallback prompt if hub doesn't work
                    from langchain.prompts import PromptTemplate
                    prompt = PromptTemplate.from_template("""
                    Answer the following questions as best you can. You have access to the following tools:

                    {tools}

                    Use the following format:

                    Question: the input question you must answer
                    Thought: you should always think about what to do
                    Action: the action to take, should be one of [{tool_names}]
                    Action Input: the input to the action
                    Observation: the result of the action
                    ... (this Thought/Action/Action Input/Observation can repeat N times)
                    Thought: I now know the final answer
                    Final Answer: the final answer to the original input question

                    Question: {input}
                    Thought: {agent_scratchpad}
                    """)

                # Create the agent
                agent = create_react_agent(self.llm, all_tools, prompt)

                # Create the executor
                agent_executor = AgentExecutor(
                    agent=agent,
                    tools=all_tools,
                    verbose=True,
                    handle_parsing_errors=True,
                    max_iterations=15,
                    max_execution_time=180
                )

                logger.info("Successfully created React agent with chart tools")
                return agent_executor

            except Exception as react_error:
                logger.warning(f"React agent creation failed: {react_error}")

                # Fallback: Try the old create_sql_agent but with explicit extra_tools
                try:
                    from langchain.agents import create_sql_agent, AgentType

                    agent_executor = create_sql_agent(
                        llm=self.llm,
                        toolkit=toolkit,
                        verbose=True,
                        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                        handle_parsing_errors=True,
                        max_iterations=10,
                        extra_tools=[chart_tool]  # This is the key fix
                    )

                    logger.info("Created SQL agent with extra_tools parameter")
                    return agent_executor

                except Exception as sql_error:
                    logger.error(f"SQL agent creation also failed: {sql_error}")

                    # Last resort: Create a custom agent executor
                    try:
                        from langchain.agents import AgentExecutor, ZeroShotAgent
                        from langchain.prompts import PromptTemplate

                        # Create custom prompt that includes chart tool
                        tool_names = [tool.name for tool in all_tools]
                        tool_descriptions = "\n".join([f"{tool.name}: {tool.description}" for tool in all_tools])

                        prompt_template = f"""
                        Answer the following questions as best you can. You have access to these tools:

                        {tool_descriptions}

                        Use this format:
                        Question: the input question
                        Thought: think about what to do
                        Action: choose from [{', '.join(tool_names)}]
                        Action Input: the input to the action
                        Observation: the result of the action
                        ... (repeat Thought/Action/Action Input/Observation as needed)
                        Thought: I now know the final answer
                        Final Answer: the final answer

                        Question: {{input}}
                        Thought: {{agent_scratchpad}}
                        """

                        prompt = PromptTemplate.from_template(prompt_template)
                        llm_chain = prompt | self.llm

                        agent = ZeroShotAgent(llm_chain=llm_chain, tools=all_tools)
                        agent_executor = AgentExecutor.from_agent_and_tools(
                            agent=agent,
                            tools=all_tools,
                            verbose=True,
                            handle_parsing_errors=True,
                            max_iterations=10
                        )

                        logger.info("Created custom agent executor")
                        return agent_executor

                    except Exception as custom_error:
                        logger.error(f"Custom agent creation failed: {custom_error}")
                        return None

        except Exception as e:
            logger.error(f"SQL Agent creation error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def analyze_with_langchain(self, user_message, table_name, connection_info):
        """Analyze data with timeout fixes and better result handling"""
        try:
            logger.info(f"Starting analysis for message: {user_message[:100]}...")

            # Check if user wants a chart first
            chart_keywords = ['chart', 'graph', 'plot', 'visualiz', 'pie', 'bar', 'line']
            wants_chart = any(keyword in user_message.lower() for keyword in chart_keywords)

            if wants_chart:
                return self.handle_chart_request(user_message, table_name, connection_info)

            # For simple data requests, use direct SQL execution to avoid agent timeout
            simple_requests = ['sample', 'preview', 'show', 'first', 'top', 'example']
            is_simple_request = any(keyword in user_message.lower() for keyword in simple_requests)

            if is_simple_request:
                logger.info("Handling as simple request with direct SQL")
                return self.handle_simple_request(user_message, table_name, connection_info)

            # Build connection string for complex queries
            if connection_info['type'] == 'default':
                connection_string = f"mysql+pymysql://{app.config['MYSQL_USER']}:{app.config['MYSQL_PASSWORD']}@{app.config['MYSQL_HOST']}/{app.config['MYSQL_DATABASE']}"
            else:
                connection_string = f"mysql+pymysql://{connection_info['user']}:{connection_info['password']}@{connection_info['host']}/{connection_info['database']}"

            # Create SQL agent with shorter timeouts
            agent = self.create_basic_sql_agent(connection_string, table_name)
            if not agent:
                return "Failed to create SQL analysis agent."

            # Enhanced prompt for complex analysis
            enhanced_prompt = f"""
            You are analyzing data from table '{table_name}'. 
            User question: {user_message}

            Please write and execute ONE SQL query to answer the question, then provide a clear, concise answer.
            Do NOT execute multiple queries unless absolutely necessary.
            Focus on being direct and helpful.
            """

            # Execute with shorter timeout
            try:
                if hasattr(agent, 'invoke'):
                    logger.info("Using invoke method with timeout controls")
                    result = agent.invoke(
                        {"input": enhanced_prompt},
                        config={"configurable": {"max_execution_time": 20, "max_iterations": 3}}
                    )

                    if isinstance(result, dict):
                        if "output" in result:
                            return result["output"]
                        elif "result" in result:
                            return result["result"]
                        else:
                            # Extract the meaningful content
                            for key, value in result.items():
                                if isinstance(value, str) and len(value) > 10:
                                    return value
                            return str(result)
                    else:
                        return str(result)
                else:
                    return "Agent execution method not available"

            except Exception as exec_error:
                logger.error(f"Agent execution failed: {exec_error}")
                # Fall back to direct SQL execution
                return self.handle_simple_request(user_message, table_name, connection_info)

        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return f"Analysis failed: {str(e)}"

    def handle_simple_request(self, user_message, table_name, connection_info):
        """Handle simple requests directly with SQL to avoid agent timeouts"""
        try:
            logger.info(f"Handling simple request: {user_message}")

            # Get database connection
            connection = DatabaseManager.get_connection()
            if not connection:
                return "Failed to connect to database"

            try:
                # Generate appropriate query based on user request
                query = self.generate_simple_query(user_message, table_name)
                logger.info(f"Executing query: {query}")

                # Execute query
                results, columns = DatabaseManager.execute_query(connection, query)
                connection.close()

                if not results:
                    return f"No results found for query: {query}"

                # Format results in a user-friendly way
                return self.format_query_results(results, columns, query, user_message)

            except Exception as exec_error:
                if 'connection' in locals():
                    connection.close()
                logger.error(f"Simple request execution failed: {exec_error}")
                return f"Query execution failed: {str(exec_error)}"

        except Exception as e:
            logger.error(f"Simple request handling error: {e}")
            return f"Failed to handle request: {str(e)}"

    def format_query_results(self, results, columns, query, user_message):
        """Format query results in a readable format"""
        try:
            result_text = f"**Query executed:** `{query}`\n\n"

            if 'count' in query.lower() and len(results) == 1 and len(results[0]) == 1:
                # Handle COUNT queries
                count = results[0][0]
                result_text += f"**Result:** {count:,} records found in the table.\n"
                return result_text

            if 'describe' in query.lower():
                # Handle DESCRIBE queries
                result_text += "**Table Structure:**\n"
                for row in results:
                    field, field_type = row[0], row[1]
                    result_text += f"- **{field}**: {field_type}\n"
                return result_text

            # Handle regular SELECT queries
            result_text += f"**Results ({len(results)} records):**\n\n"

            # Create formatted table
            if len(results) <= 10:
                # Show all results for small datasets
                for i, row in enumerate(results, 1):
                    result_text += f"**Record {i}:**\n"
                    for col, val in zip(columns, row):
                        # Format different data types appropriately
                        if val is None:
                            formatted_val = "NULL"
                        elif isinstance(val, (int, float)):
                            formatted_val = f"{val:,}" if isinstance(val, int) else f"{val:.2f}"
                        elif len(str(val)) > 50:
                            formatted_val = str(val)[:47] + "..."
                        else:
                            formatted_val = str(val)

                        result_text += f"  - {col}: {formatted_val}\n"
                    result_text += "\n"
            else:
                # Show summary for large datasets
                result_text += f"Showing first 5 of {len(results)} records:\n\n"
                for i, row in enumerate(results[:5], 1):
                    result_text += f"**Record {i}:**\n"
                    for col, val in zip(columns, row):
                        if val is None:
                            formatted_val = "NULL"
                        elif isinstance(val, (int, float)):
                            formatted_val = f"{val:,}" if isinstance(val, int) else f"{val:.2f}"
                        elif len(str(val)) > 50:
                            formatted_val = str(val)[:47] + "..."
                        else:
                            formatted_val = str(val)

                        result_text += f"  - {col}: {formatted_val}\n"
                    result_text += "\n"

                result_text += f"... and {len(results) - 5} more records.\n"

            # Add column summary
            result_text += f"\n**Columns:** {', '.join(columns)}"

            return result_text

        except Exception as e:
            logger.error(f"Error formatting results: {e}")
            # Fallback to simple format
            return f"Query: {query}\n\nResults: {len(results)} records returned\nColumns: {', '.join(columns)}"

    def create_basic_sql_agent(self, connection_string, table_name):
        """Create SQL agent with strict timeout controls"""
        try:
            logger.info(f"Creating SQL agent with timeout controls")

            # Create SQLDatabase
            try:
                from langchain_community.utilities import SQLDatabase
            except ImportError:
                from langchain.sql_database import SQLDatabase

            db = SQLDatabase.from_uri(connection_string)

            # Create toolkit
            try:
                from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
            except ImportError:
                from langchain.agents.agent_toolkits import SQLDatabaseToolkit

            toolkit = SQLDatabaseToolkit(db=db, llm=self.llm)

            # Create agent with very strict limits
            try:
                from langchain.agents import create_sql_agent

                agent = create_sql_agent(
                    llm=self.llm,
                    toolkit=toolkit,
                    verbose=True,
                    handle_parsing_errors=True,
                    max_iterations=2,  # Very limited iterations
                    max_execution_time=15,  # Short timeout
                    early_stopping_method="generate"  # Stop early when possible
                )

                logger.info("SQL agent created with timeout controls")
                return agent

            except Exception as agent_error:
                logger.error(f"Agent creation failed: {agent_error}")
                return None

        except Exception as e:
            logger.error(f"SQL agent creation error: {e}")
            return None

    def fallback_sql_analysis(self, user_message, table_name, connection_info):
        """Fallback method for direct SQL analysis when agent fails"""
        try:
            logger.info("Using fallback SQL analysis")

            # Get database connection
            connection = DatabaseManager.get_connection()
            if not connection:
                return "Failed to connect to database"

            # Get table schema for context
            schema = DatabaseManager.get_table_schema(connection, table_name)
            schema_info = [f"{col[0]} ({col[1]})" for col in schema] if schema else ["No schema available"]

            # Generate simple queries based on user request
            if any(word in user_message.lower() for word in ['first', 'preview', 'sample', 'show me']):
                query = f"SELECT * FROM {table_name} LIMIT 10"
            elif any(word in user_message.lower() for word in ['count', 'total', 'how many']):
                query = f"SELECT COUNT(*) as total_records FROM {table_name}"
            elif any(word in user_message.lower() for word in ['statistics', 'stats', 'summary']):
                # Try to get basic stats
                numeric_columns = [col[0] for col in schema if
                                   'int' in col[1].lower() or 'decimal' in col[1].lower() or 'float' in col[1].lower()]
                if numeric_columns:
                    query = f"SELECT COUNT(*) as count, MIN({numeric_columns[0]}) as min_val, MAX({numeric_columns[0]}) as max_val FROM {table_name}"
                else:
                    query = f"SELECT COUNT(*) as total_records FROM {table_name}"
            else:
                # Default query
                query = f"SELECT * FROM {table_name} LIMIT 5"

            # Execute query
            results, columns = DatabaseManager.execute_query(connection, query)
            connection.close()

            if not results:
                return f"No results found. Table schema: {', '.join(schema_info)}"

            # Format results
            result_text = f"**Query executed:** `{query}`\n\n"
            result_text += f"**Results:**\n"

            # Create a simple table format
            if len(results) <= 10:  # Only show detailed results for small datasets
                for i, row in enumerate(results):
                    result_text += f"Row {i + 1}: "
                    result_text += ", ".join([f"{col}={val}" for col, val in zip(columns, row)])
                    result_text += "\n"
            else:
                result_text += f"Found {len(results)} records. First few rows:\n"
                for i in range(min(3, len(results))):
                    row = results[i]
                    result_text += f"Row {i + 1}: "
                    result_text += ", ".join([f"{col}={val}" for col, val in zip(columns, row)])
                    result_text += "\n"

            result_text += f"\n**Table Info:** {len(schema_info)} columns: {', '.join(schema_info)}"

            return result_text

        except Exception as e:
            logger.error(f"Fallback analysis failed: {e}")
            return f"Fallback analysis failed: {str(e)}"

    def generate_simple_query(self, user_message, table_name):
        """Generate SQL query for simple requests"""
        message_lower = user_message.lower()

        # Extract number if mentioned
        import re
        numbers = re.findall(r'\d+', user_message)
        limit = int(numbers[0]) if numbers else 5

        # Keep reasonable limits
        limit = min(limit, 50)

        if any(word in message_lower for word in ['sample', 'example', 'show', 'preview']):
            return f"SELECT * FROM {table_name} LIMIT {limit}"
        elif any(word in message_lower for word in ['first', 'top']):
            return f"SELECT * FROM {table_name} LIMIT {limit}"
        elif any(word in message_lower for word in ['count', 'total', 'how many']):
            return f"SELECT COUNT(*) as total_records FROM {table_name}"
        elif any(word in message_lower for word in ['columns', 'structure', 'schema']):
            return f"DESCRIBE {table_name}"
        else:
            return f"SELECT * FROM {table_name} LIMIT {limit}"

    def create_basic_sql_agent(self, connection_string, table_name):
        """Create basic SQL agent with improved error handling"""
        try:
            logger.info(f"Creating basic SQL agent for table: {table_name}")

            # Create SQLDatabase
            try:
                from langchain_community.utilities import SQLDatabase
            except ImportError:
                from langchain.sql_database import SQLDatabase

            db = SQLDatabase.from_uri(connection_string)
            logger.info("Database connection successful")

            # Create toolkit
            try:
                from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
            except ImportError:
                from langchain.agents.agent_toolkits import SQLDatabaseToolkit

            toolkit = SQLDatabaseToolkit(db=db, llm=self.llm)

            # Create agent with minimal configuration
            try:
                from langchain.agents import create_sql_agent

                agent = create_sql_agent(
                    llm=self.llm,
                    toolkit=toolkit,
                    verbose=True,
                    handle_parsing_errors=True,
                    max_iterations=10,  # Reduced to avoid timeouts
                    max_execution_time=120
                )

                logger.info("Basic SQL agent created successfully")
                return agent

            except Exception as agent_error:
                logger.error(f"SQL agent creation failed: {agent_error}")

                # Try alternative agent creation
                try:
                    from langchain.agents import AgentType
                    agent = create_sql_agent(
                        llm=self.llm,
                        toolkit=toolkit,
                        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                        verbose=True,
                        handle_parsing_errors=True
                    )
                    logger.info("Alternative SQL agent created")
                    return agent
                except Exception as alt_error:
                    logger.error(f"Alternative agent creation failed: {alt_error}")
                    return None

        except Exception as e:
            logger.error(f"Basic SQL agent creation error: {e}")
            return None


    def handle_chart_request(self, user_message, table_name, connection_info):
        """Handle chart requests by analyzing user intent and generating appropriate charts"""
        try:
            logger.info(f"Handling chart request: {user_message}")

            # Get database connection
            connection = DatabaseManager.get_connection()
            if not connection:
                return "Failed to connect to database for chart generation"

            try:
                # Get table schema to understand available columns
                schema = DatabaseManager.get_table_schema(connection, table_name)
                if not schema:
                    connection.close()
                    return f"Could not get schema for table {table_name}"

                columns = [col[0] for col in schema]
                logger.info(f"Available columns: {columns}")

                # Determine chart type from user message
                chart_type = "bar"  # default
                if 'pie' in user_message.lower():
                    chart_type = "pie"
                elif 'line' in user_message.lower():
                    chart_type = "line"
                elif 'horizontal' in user_message.lower() or 'ranking' in user_message.lower():
                    chart_type = "horizontal_bar"

                # Generate appropriate SQL query based on chart type and available columns
                sql_query = self.generate_chart_query(table_name, columns, chart_type, user_message)

                logger.info(f"Generated query: {sql_query}")

                # Execute the query
                results, result_columns = DatabaseManager.execute_query(connection, sql_query)
                connection.close()

                if not results or len(result_columns) < 2:
                    return f"Query returned insufficient data for chart. SQL: {sql_query}"

                # Create chart
                from chart_generator import ChartGenerator
                chart_generator = ChartGenerator()
                base_url = get_base_url()

                # Convert data for chart
                import pandas as pd
                df = pd.DataFrame(results, columns=result_columns)

                # Generate chart title
                title = f"{chart_type.title()} Chart of {result_columns[1]} by {result_columns[0]}"
                if "count" in result_columns[1].lower():
                    title = f"Distribution of {result_columns[0]}"

                filename = None

                if chart_type == "pie":
                    labels = df.iloc[:, 0].astype(str).tolist()
                    values = pd.to_numeric(df.iloc[:, 1], errors='coerce').tolist()

                    # Remove any NaN values
                    clean_data = [(l, v) for l, v in zip(labels, values) if not pd.isna(v)]
                    if clean_data:
                        labels, values = zip(*clean_data)
                        # Limit to top 10 for readability
                        if len(labels) > 10:
                            sorted_data = sorted(zip(labels, values), key=lambda x: x[1], reverse=True)[:10]
                            labels, values = zip(*sorted_data)

                        filename = chart_generator.create_pie_chart(list(values), list(labels), title)

                elif chart_type in ["bar", "horizontal_bar"]:
                    x_data = df.iloc[:, 0].astype(str).tolist()
                    y_data = pd.to_numeric(df.iloc[:, 1], errors='coerce').tolist()

                    # Remove any NaN values
                    clean_data = [(x, y) for x, y in zip(x_data, y_data) if not pd.isna(y)]
                    if clean_data:
                        x_data, y_data = zip(*clean_data)
                        horizontal = chart_type == "horizontal_bar"
                        filename = chart_generator.create_bar_chart(
                            list(x_data), list(y_data), title,
                            result_columns[0], result_columns[1], horizontal
                        )

                elif chart_type == "line":
                    x_data = df.iloc[:, 0].tolist()
                    y_data = pd.to_numeric(df.iloc[:, 1], errors='coerce').tolist()

                    # Remove any NaN values
                    clean_data = [(x, y) for x, y in zip(x_data, y_data) if not pd.isna(y)]
                    if clean_data:
                        x_data, y_data = zip(*clean_data)
                        filename = chart_generator.create_line_chart(
                            list(x_data), list(y_data), title,
                            result_columns[0], result_columns[1]
                        )

                if filename:
                    chart_url = f"{base_url}/uploads/{filename}"
                    return f"""✅ **Chart Generated Successfully!**
    
    📊 **Chart Type:** {chart_type.replace('_', ' ').title()}
    📈 **Title:** {title}
    
    🔗 **View Chart:** {chart_url}
    
    📋 **Data Summary:**
    - Records: {len(df)}
    - Columns: {', '.join(result_columns)}
    
    💻 **SQL Query Used:**
    ```sql
    {sql_query}
    ```
    
    The chart visualizes your data based on the analysis of {len(df)} records from the table."""
                else:
                    return f"❌ Failed to generate {chart_type} chart. The data might not be suitable for this chart type.\n\nSQL used: {sql_query}\nData preview: {df.head().to_string()}"

            except Exception as exec_error:
                if 'connection' in locals():
                    connection.close()
                logger.error(f"Error in chart generation: {exec_error}")
                return f"Error generating chart: {str(exec_error)}"

        except Exception as e:
            logger.error(f"Chart request handling error: {e}")
            import traceback
            traceback.print_exc()
            return f"Failed to handle chart request: {str(e)}"


    def generate_chart_query(self, table_name, columns, chart_type, user_message):
        """Generate appropriate SQL query for chart based on available columns and user intent"""
        try:
            # Look for specific column mentions in user message
            mentioned_columns = [col for col in columns if col.lower() in user_message.lower()]

            if chart_type == "pie":
                # For pie charts, we need a categorical column to group by
                categorical_cols = [col for col in columns if any(keyword in col.lower() for keyword in
                                                                  ['name', 'category', 'type', 'status', 'group', 'class',
                                                                   'product', 'region'])]

                if mentioned_columns:
                    group_col = mentioned_columns[0]
                elif categorical_cols:
                    group_col = categorical_cols[0]
                else:
                    group_col = columns[1] if len(columns) > 1 else columns[0]  # Skip ID column

                return f"SELECT {group_col}, COUNT(*) as count FROM {table_name} GROUP BY {group_col} ORDER BY count DESC LIMIT 10"

            elif chart_type in ["bar", "horizontal_bar"]:
                # For bar charts, try to find categorical and numeric columns
                categorical_cols = [col for col in columns if any(keyword in col.lower() for keyword in
                                                                  ['name', 'category', 'type', 'status', 'group', 'class',
                                                                   'product', 'region'])]
                numeric_cols = [col for col in columns if any(keyword in col.lower() for keyword in
                                                              ['price', 'amount', 'value', 'count', 'total', 'sum',
                                                               'quantity', 'score'])]

                if mentioned_columns and len(mentioned_columns) >= 2:
                    x_col, y_col = mentioned_columns[0], mentioned_columns[1]
                elif categorical_cols and numeric_cols:
                    x_col, y_col = categorical_cols[0], numeric_cols[0]
                elif categorical_cols:
                    x_col = categorical_cols[0]
                    y_col = "COUNT(*)"
                    return f"SELECT {x_col}, COUNT(*) as count FROM {table_name} GROUP BY {x_col} ORDER BY count DESC LIMIT 15"
                else:
                    x_col, y_col = columns[1] if len(columns) > 1 else columns[0], "COUNT(*)"
                    return f"SELECT {x_col}, COUNT(*) as count FROM {table_name} GROUP BY {x_col} ORDER BY count DESC LIMIT 15"

                return f"SELECT {x_col}, {y_col} FROM {table_name} ORDER BY {y_col} DESC LIMIT 15"

            elif chart_type == "line":
                # For line charts, try to find date/time and numeric columns
                date_cols = [col for col in columns if any(keyword in col.lower() for keyword in
                                                           ['date', 'time', 'created', 'updated', 'year', 'month'])]
                numeric_cols = [col for col in columns if any(keyword in col.lower() for keyword in
                                                              ['price', 'amount', 'value', 'count', 'total', 'sum',
                                                               'quantity', 'score'])]

                if mentioned_columns and len(mentioned_columns) >= 2:
                    x_col, y_col = mentioned_columns[0], mentioned_columns[1]
                elif date_cols and numeric_cols:
                    x_col, y_col = date_cols[0], numeric_cols[0]
                elif date_cols:
                    x_col = date_cols[0]
                    return f"SELECT {x_col}, COUNT(*) as count FROM {table_name} GROUP BY {x_col} ORDER BY {x_col} LIMIT 20"
                else:
                    # Fallback to first two columns
                    x_col = columns[0]
                    y_col = columns[1] if len(columns) > 1 else "COUNT(*)"
                    if y_col == "COUNT(*)":
                        return f"SELECT {x_col}, COUNT(*) as count FROM {table_name} GROUP BY {x_col} ORDER BY {x_col} LIMIT 20"

                return f"SELECT {x_col}, {y_col} FROM {table_name} ORDER BY {x_col} LIMIT 20"

            # Default fallback
            return f"SELECT * FROM {table_name} LIMIT 10"

        except Exception as e:
            logger.error(f"Error generating chart query: {e}")
            # Simple fallback
            return f"SELECT * FROM {table_name} LIMIT 10"

# Initialize LangChain analyzer
langchain_analyzer = LangChainAnalyzer()


# Routes with automatic base URL detection
@app.route('/')
@app.route('/chat2data/')
def index():
    base_url = get_base_url()
    return render_template('index.html', base_url=base_url)


@app.route('/upload', methods=['POST'])
@app.route('/chat2data/upload', methods=['POST'])
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

                # 成功重定向处理
                base_url = get_base_url()
                redirect_url = f'{base_url}/chat' if base_url else '/chat'

                return jsonify({
                    'success': True,
                    'message': success_msg,
                    'data_info': session['data_info'],
                    'redirect_url': redirect_url
                })

            except Exception as e:
                connection.close()
                return jsonify({'success': False, 'message': f'Import failed: {str(e)}'})

    except Exception as e:
        logger.error(f"File upload error: {e}")
        return jsonify({'success': False, 'message': f'File processing failed: {str(e)}'})

    return jsonify({'success': False, 'message': 'File format not supported'})


@app.route('/chat')
@app.route('/chat2data/chat')
def chat():
    """Chat page"""
    if 'data_info' not in session:
        base_url = get_base_url()
        if base_url:
            return redirect(f'{base_url}/')
        else:
            return redirect('/')
    base_url = get_base_url()
    return render_template('chat.html', data_info=session['data_info'], base_url=base_url)


@app.route('/send_message', methods=['POST'])
@app.route('/chat2data/send_message', methods=['POST'])
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

        chart_keywords = ['chart', 'graph', 'plot', 'visualiz', 'pie', 'bar', 'line']
        if any(keyword in user_message.lower() for keyword in chart_keywords):
            logger.info(f"Chart request detected: {user_message}")

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
@app.route('/chat2data/get_data_summary')
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
@app.route('/chat2data/cleanup_table', methods=['POST'])
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


# 添加静态文件服务路由用于图表显示
@app.route('/uploads/<filename>')
@app.route('/chat2data/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files (including generated charts)"""
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except FileNotFoundError:
        return "File not found", 404


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3001)