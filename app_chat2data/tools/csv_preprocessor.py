#!/usr/bin/env python3
"""
CSV Preprocessing Tool
Analyzes and fixes common CSV import issues, especially date formatting problems
"""

import pandas as pd
import numpy as np
import sys
import os
import re
from datetime import datetime
import chardet


class CSVPreprocessor:
    """Advanced CSV preprocessing for database import"""

    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.encoding = None
        self.issues = []
        self.fixes_applied = []

    def detect_encoding(self):
        """Detect file encoding"""
        encodings_to_try = [
            'utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'gb18030',
            'big5', 'latin1', 'cp1252', 'iso-8859-1', 'windows-1252'
        ]

        # Try chardet first
        try:
            with open(self.file_path, 'rb') as f:
                raw_data = f.read(10000)
                detected = chardet.detect(raw_data)
                if detected['confidence'] > 0.7:
                    self.encoding = detected['encoding']
                    return self.encoding
        except:
            pass

        # Try encodings manually
        for encoding in encodings_to_try:
            try:
                pd.read_csv(self.file_path, encoding=encoding, nrows=5)
                self.encoding = encoding
                return encoding
            except:
                continue

        self.encoding = 'latin1'  # Fallback
        return self.encoding

    def load_csv(self):
        """Load CSV file with detected encoding"""
        if not self.encoding:
            self.detect_encoding()

        try:
            self.df = pd.read_csv(self.file_path, encoding=self.encoding)
            print(f"✅ Loaded CSV: {len(self.df)} rows, {len(self.df.columns)} columns")
            print(f"📁 Encoding: {self.encoding}")
            return True
        except Exception as e:
            print(f"❌ Failed to load CSV: {e}")
            return False

    def analyze_columns(self):
        """Analyze all columns for potential issues"""
        analysis = {}

        for col in self.df.columns:
            col_analysis = {
                'name': col,
                'clean_name': self._clean_column_name(col),
                'dtype': str(self.df[col].dtype),
                'null_count': self.df[col].isnull().sum(),
                'null_percentage': (self.df[col].isnull().sum() / len(self.df)) * 100,
                'unique_count': self.df[col].nunique(),
                'sample_values': self.df[col].dropna().head(5).tolist(),
                'issues': [],
                'suggested_mysql_type': 'TEXT'
            }

            # Check for potential date columns
            if self._is_potential_date_column(col, self.df[col]):
                col_analysis['issues'].append('potential_date_column')
                col_analysis['date_analysis'] = self._analyze_date_column(self.df[col])

            # Check for very long strings
            if self.df[col].dtype == 'object':
                max_length = self.df[col].astype(str).str.len().max()
                if max_length > 255:
                    col_analysis['issues'].append(f'long_strings_max_{max_length}')

                # Suggest MySQL type
                if max_length <= 50:
                    col_analysis['suggested_mysql_type'] = 'VARCHAR(100)'
                elif max_length <= 255:
                    col_analysis['suggested_mysql_type'] = 'VARCHAR(500)'
                elif max_length <= 65535:
                    col_analysis['suggested_mysql_type'] = 'TEXT'
                else:
                    col_analysis['suggested_mysql_type'] = 'LONGTEXT'

            # Check for problematic characters in column names
            if col != self._clean_column_name(col):
                col_analysis['issues'].append('problematic_column_name')

            analysis[col] = col_analysis

        return analysis

    def _clean_column_name(self, col_name):
        """Clean column name for MySQL compatibility"""
        clean_name = str(col_name).strip()
        # Remove/replace problematic characters
        replacements = {
            ' ': '_', '-': '_', '(': '', ')': '', '[': '', ']': '',
            '.': '_', ',': '_', '/': '_', '\\': '_', '&': 'and',
            '%': 'percent', '#': 'num', '@': 'at', '$': 'dollar',
            '!': '', '?': '', '*': '', '+': 'plus', '=': 'eq',
            '<': 'lt', '>': 'gt', '|': '_', ';': '_', ':': '_',
            '"': '', "'": '', '`': ''
        }

        for old, new in replacements.items():
            clean_name = clean_name.replace(old, new)

        # Remove non-ASCII characters
        clean_name = clean_name.encode('ascii', 'ignore').decode('ascii')

        # Ensure not empty
        if not clean_name or clean_name == '_':
            clean_name = 'unnamed_column'

        return clean_name

    def _is_potential_date_column(self, col_name, series):
        """Check if a column might contain dates"""
        col_lower = col_name.lower()
        date_keywords = ['date', 'time', 'timestamp', 'created', 'updated', 'modified', 'birth', 'expire']

        # Check by name
        if any(keyword in col_lower for keyword in date_keywords):
            return True

        # Check by content
        sample_values = series.dropna().head(10)
        if len(sample_values) == 0:
            return False

        date_like_count = 0
        for val in sample_values:
            val_str = str(val).strip()
            # Check for date-like patterns
            if (re.search(r'\d{1,4}[/-]\d{1,2}[/-]\d{2,4}', val_str) or
                    re.search(r'\d{4}-\d{2}-\d{2}', val_str) or
                    re.search(r'\d{1,2}:\d{2}', val_str)):
                date_like_count += 1

        return date_like_count / len(sample_values) > 0.5

    def _analyze_date_column(self, series):
        """Analyze a potential date column"""
        analysis = {
            'total_values': len(series),
            'non_null_values': series.notna().sum(),
            'sample_values': series.dropna().head(10).tolist(),
            'conversion_attempts': {}
        }

        # Try different parsing methods
        methods = [
            ('pandas_auto', lambda x: pd.to_datetime(x, errors='coerce', infer_datetime_format=True)),
            ('us_format_mdy', lambda x: pd.to_datetime(x, format='%m/%d/%Y', errors='coerce')),
            ('us_format_mdy_time', lambda x: pd.to_datetime(x, format='%m/%d/%Y %H:%M', errors='coerce')),
            ('iso_format', lambda x: pd.to_datetime(x, format='%Y-%m-%d', errors='coerce')),
            ('iso_format_time', lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S', errors='coerce'))
        ]

        for method_name, method_func in methods:
            try:
                converted = method_func(series)
                success_count = converted.notna().sum()
                success_rate = success_count / analysis['non_null_values'] if analysis['non_null_values'] > 0 else 0

                analysis['conversion_attempts'][method_name] = {
                    'success_count': success_count,
                    'success_rate': success_rate,
                    'sample_converted': converted.dropna().head(3).dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
                }
            except Exception as e:
                analysis['conversion_attempts'][method_name] = {
                    'error': str(e)
                }

        # Find best conversion method
        best_method = None
        best_rate = 0

        for method_name, result in analysis['conversion_attempts'].items():
            if 'success_rate' in result and result['success_rate'] > best_rate:
                best_rate = result['success_rate']
                best_method = method_name

        analysis['recommended_method'] = best_method
        analysis['recommended_success_rate'] = best_rate

        return analysis

    def fix_date_columns(self):
        """Fix date columns based on analysis"""
        analysis = self.analyze_columns()
        fixed_columns = []

        for col_name, col_info in analysis.items():
            if 'potential_date_column' in col_info['issues']:
                date_analysis = col_info['date_analysis']
                recommended_method = date_analysis['recommended_method']

                if recommended_method and date_analysis['recommended_success_rate'] > 0.7:
                    print(f"🔧 Fixing date column '{col_name}' using method '{recommended_method}'")

                    try:
                        if recommended_method == 'pandas_auto':
                            converted = pd.to_datetime(self.df[col_name], errors='coerce', infer_datetime_format=True)
                        elif recommended_method == 'us_format_mdy':
                            converted = pd.to_datetime(self.df[col_name], format='%m/%d/%Y', errors='coerce')
                        elif recommended_method == 'us_format_mdy_time':
                            converted = pd.to_datetime(self.df[col_name], format='%m/%d/%Y %H:%M', errors='coerce')
                        elif recommended_method == 'iso_format':
                            converted = pd.to_datetime(self.df[col_name], format='%Y-%m-%d', errors='coerce')
                        elif recommended_method == 'iso_format_time':
                            converted = pd.to_datetime(self.df[col_name], format='%Y-%m-%d %H:%M:%S', errors='coerce')

                        # Convert to MySQL-friendly format
                        self.df[col_name] = converted.dt.strftime('%Y-%m-%d %H:%M:%S').fillna('')
                        fixed_columns.append(col_name)
                        self.fixes_applied.append(f"Fixed date column '{col_name}' using {recommended_method}")

                    except Exception as e:
                        print(f"❌ Failed to fix date column '{col_name}': {e}")
                else:
                    print(
                        f"⚠️ Date column '{col_name}' has low conversion rate ({date_analysis['recommended_success_rate']:.2f}), keeping as text")

        return fixed_columns

    def clean_column_names(self):
        """Clean all column names"""
        original_names = self.df.columns.tolist()
        clean_names = [self._clean_column_name(name) for name in original_names]

        # Handle duplicates
        final_names = []
        for name in clean_names:
            original_name = name
            counter = 1
            while name in final_names:
                name = f"{original_name}_{counter}"
                counter += 1
            final_names.append(name)

        if original_names != final_names:
            self.df.columns = final_names
            self.fixes_applied.append("Cleaned column names for MySQL compatibility")
            print("🔧 Cleaned column names")

            # Show mapping
            for old, new in zip(original_names, final_names):
                if old != new:
                    print(f"   '{old}' → '{new}'")

        return final_names

    def save_processed_csv(self, output_path=None):
        """Save the processed CSV"""
        if output_path is None:
            base_name = os.path.splitext(self.file_path)[0]
            output_path = f"{base_name}_processed.csv"

        try:
            self.df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"💾 Saved processed CSV to: {output_path}")
            return output_path
        except Exception as e:
            print(f"❌ Failed to save processed CSV: {e}")
            return None

    def generate_report(self):
        """Generate a comprehensive analysis report"""
        print("\n" + "=" * 60)
        print("📊 CSV Analysis Report")
        print("=" * 60)

        print(f"📁 File: {self.file_path}")
        print(f"📊 Dimensions: {len(self.df)} rows × {len(self.df.columns)} columns")
        print(f"🔤 Encoding: {self.encoding}")

        if self.fixes_applied:
            print(f"\n🔧 Fixes Applied:")
            for fix in self.fixes_applied:
                print(f"   ✅ {fix}")

        analysis = self.analyze_columns()

        print(f"\n📋 Column Analysis:")
        for col_name, col_info in analysis.items():
            print(f"\n   Column: {col_name}")
            if col_info['name'] != col_info['clean_name']:
                print(f"     Clean name: {col_info['clean_name']}")
            print(f"     Type: {col_info['dtype']}")
            print(f"     Nulls: {col_info['null_count']} ({col_info['null_percentage']:.1f}%)")
            print(f"     Unique values: {col_info['unique_count']}")
            print(f"     MySQL type: {col_info['suggested_mysql_type']}")

            if col_info['issues']:
                print(f"     Issues: {', '.join(col_info['issues'])}")

            if 'date_analysis' in col_info:
                date_info = col_info['date_analysis']
                print(
                    f"     Date conversion: {date_info['recommended_method']} ({date_info['recommended_success_rate']:.2f} success rate)")


def main():
    """Command line interface"""
    if len(sys.argv) < 2:
        print("Usage: python csv_preprocessor.py <csv_file> [output_file]")
        print("Example: python csv_preprocessor.py data.csv data_fixed.csv")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.exists(input_file):
        print(f"❌ File not found: {input_file}")
        sys.exit(1)

    print("🚀 Starting CSV preprocessing...")

    # Initialize preprocessor
    preprocessor = CSVPreprocessor(input_file)

    # Load and analyze
    if not preprocessor.load_csv():
        sys.exit(1)

    # Apply fixes
    print("\n🔧 Applying fixes...")
    preprocessor.clean_column_names()
    fixed_date_columns = preprocessor.fix_date_columns()

    # Generate report
    preprocessor.generate_report()

    # Save processed file
    if output_file or fixed_date_columns or preprocessor.fixes_applied:
        print(f"\n💾 Saving processed file...")
        saved_path = preprocessor.save_processed_csv(output_file)
        if saved_path:
            print(f"✅ Use the processed file for import: {saved_path}")

    print(f"\n🎯 Ready for import! Use the processed CSV file in your application.")


if __name__ == "__main__":
    main()