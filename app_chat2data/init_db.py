#!/usr/bin/env python3
"""
Database initialization script for Data Analysis Chat Tool
This script creates the necessary database and tables for the application.
"""

import mysql.connector
import os
from dotenv import load_dotenv

load_dotenv()


def create_database():
    """Create the main database for storing imported CSV data"""
    try:
        # Connect to MySQL server (without specifying database)
        connection = mysql.connector.connect(
            host=os.environ.get('MYSQL_HOST', 'localhost'),
            user=os.environ.get('MYSQL_USER', 'root'),
            password=os.environ.get('MYSQL_PASSWORD', ''),
            charset='utf8mb4'
        )

        cursor = connection.cursor()

        # Create database if it doesn't exist
        database_name = os.environ.get('MYSQL_DATABASE', 'data_analysis')
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database_name}")
        print(f"✅ Database '{database_name}' created successfully!")

        # Use the database
        cursor.execute(f"USE {database_name}")

        # Create a metadata table to track imported CSV files
        create_metadata_table = """
        CREATE TABLE IF NOT EXISTS csv_metadata (
            id INT AUTO_INCREMENT PRIMARY KEY,
            table_name VARCHAR(255) UNIQUE NOT NULL,
            original_filename VARCHAR(255) NOT NULL,
            upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            row_count INT NOT NULL,
            column_count INT NOT NULL,
            columns_info JSON,
            file_size BIGINT,
            created_by VARCHAR(100),
            status ENUM('active', 'archived', 'deleted') DEFAULT 'active',
            INDEX idx_table_name (table_name),
            INDEX idx_upload_timestamp (upload_timestamp),
            INDEX idx_status (status)
        )
        """
        cursor.execute(create_metadata_table)
        print("✅ CSV metadata table created successfully!")

        # Create a table for storing analysis history
        create_analysis_history_table = """
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INT AUTO_INCREMENT PRIMARY KEY,
            session_id VARCHAR(255),
            table_name VARCHAR(255),
            user_query TEXT NOT NULL,
            ai_response TEXT,
            sql_executed TEXT,
            execution_time DECIMAL(10,4),
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            success BOOLEAN DEFAULT TRUE,
            error_message TEXT,
            INDEX idx_session_id (session_id),
            INDEX idx_table_name (table_name),
            INDEX idx_timestamp (timestamp),
            FOREIGN KEY (table_name) REFERENCES csv_metadata(table_name) ON DELETE CASCADE
        )
        """
        cursor.execute(create_analysis_history_table)
        print("✅ Analysis history table created successfully!")

        # Create a view for active tables
        create_active_tables_view = """
        CREATE OR REPLACE VIEW active_csv_tables AS
        SELECT 
            table_name,
            original_filename,
            upload_timestamp,
            row_count,
            column_count,
            JSON_EXTRACT(columns_info, '$.columns') as column_names
        FROM csv_metadata 
        WHERE status = 'active'
        ORDER BY upload_timestamp DESC
        """
        cursor.execute(create_active_tables_view)
        print("✅ Active tables view created successfully!")

        cursor.close()
        connection.close()

        print("\n🎉 Database initialization completed successfully!")
        print(f"📊 Database: {database_name}")
        print("📋 Tables created:")
        print("   - csv_metadata (tracks imported files)")
        print("   - analysis_history (stores query history)")
        print("   - active_csv_tables (view of active tables)")

    except mysql.connector.Error as e:
        print(f"❌ Database initialization failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

    return True


def create_cleanup_procedure():
    """Create stored procedure for cleaning up old tables"""
    try:
        connection = mysql.connector.connect(
            host=os.environ.get('MYSQL_HOST', 'localhost'),
            user=os.environ.get('MYSQL_USER', 'root'),
            password=os.environ.get('MYSQL_PASSWORD', ''),
            database=os.environ.get('MYSQL_DATABASE', 'data_analysis'),
            charset='utf8mb4'
        )

        cursor = connection.cursor()

        # Create cleanup procedure
        cleanup_procedure = """
        DELIMITER //
        CREATE PROCEDURE IF NOT EXISTS CleanupOldTables(IN days_old INT)
        BEGIN
            DECLARE done INT DEFAULT FALSE;
            DECLARE table_name_var VARCHAR(255);
            DECLARE table_cursor CURSOR FOR 
                SELECT table_name 
                FROM csv_metadata 
                WHERE upload_timestamp < DATE_SUB(NOW(), INTERVAL days_old DAY)
                AND status = 'active';
            DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = TRUE;

            OPEN table_cursor;

            read_loop: LOOP
                FETCH table_cursor INTO table_name_var;
                IF done THEN
                    LEAVE read_loop;
                END IF;

                SET @sql = CONCAT('DROP TABLE IF EXISTS ', table_name_var);
                PREPARE stmt FROM @sql;
                EXECUTE stmt;
                DEALLOCATE PREPARE stmt;

                UPDATE csv_metadata 
                SET status = 'deleted' 
                WHERE table_name = table_name_var;

            END LOOP;

            CLOSE table_cursor;
        END//
        DELIMITER ;
        """

        # Note: MySQL connector doesn't handle DELIMITER well, so we'll create a simpler version
        simple_cleanup = """
        CREATE PROCEDURE IF NOT EXISTS CleanupOldTables(IN days_old INT)
        BEGIN
            -- This procedure would need to be implemented with proper delimiter handling
            SELECT 'Cleanup procedure created - implement via MySQL command line' as message;
        END
        """

        cursor.execute(simple_cleanup)
        print("✅ Cleanup procedure template created!")

        cursor.close()
        connection.close()

    except mysql.connector.Error as e:
        print(f"⚠️  Warning: Could not create cleanup procedure: {e}")


def test_connection():
    """Test the database connection and show current settings"""
    try:
        connection = mysql.connector.connect(
            host=os.environ.get('MYSQL_HOST', 'localhost'),
            user=os.environ.get('MYSQL_USER', 'root'),
            password=os.environ.get('MYSQL_PASSWORD', ''),
            database=os.environ.get('MYSQL_DATABASE', 'data_analysis'),
            charset='utf8mb4'
        )

        cursor = connection.cursor()

        # Test basic query
        cursor.execute("SELECT DATABASE(), USER(), NOW()")
        result = cursor.fetchone()

        print("\n🔗 Connection Test Results:")
        print(f"   Database: {result[0]}")
        print(f"   User: {result[1]}")
        print(f"   Timestamp: {result[2]}")

        # Show tables
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        print(f"\n📋 Available Tables: {len(tables)}")
        for table in tables:
            print(f"   - {table[0]}")

        cursor.close()
        connection.close()

        return True

    except mysql.connector.Error as e:
        print(f"❌ Connection test failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Starting Database Initialization...")
    print("=" * 50)

    # Load environment variables
    print("📁 Environment variables:")
    print(f"   MYSQL_HOST: {os.environ.get('MYSQL_HOST', 'localhost')}")
    print(f"   MYSQL_USER: {os.environ.get('MYSQL_USER', 'root')}")
    print(f"   MYSQL_DATABASE: {os.environ.get('MYSQL_DATABASE', 'data_analysis')}")
    print(f"   OPENAI_API_KEY: {'Set' if os.environ.get('OPENAI_API_KEY') else 'Not Set'}")
    print()

    # Create database and tables
    if create_database():
        print("\n🧹 Creating cleanup procedures...")
        create_cleanup_procedure()

        print("\n🧪 Testing connection...")
        if test_connection():
            print("\n✅ All systems ready!")
            print("\n🎯 Next steps:")
            print("   1. Start the Flask application: python app.py")
            print("   2. Upload a CSV file or connect to a database")
            print("   3. Start analyzing your data with LangChain!")
        else:
            print("\n❌ Connection test failed. Please check your configuration.")
    else:
        print("\n❌ Database initialization failed. Please check your MySQL configuration.")