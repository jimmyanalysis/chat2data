#!/usr/bin/env python3
"""
CSV Encoding Detection Utility
Helps detect and test different encodings for CSV files
"""

import pandas as pd
import chardet
import os
import sys


class CSVEncodingDetector:
    """Utility class for detecting and testing CSV file encodings"""

    COMMON_ENCODINGS = [
        'utf-8', 'utf-8-sig', 'utf-16', 'utf-16-le', 'utf-16-be',
        'gbk', 'gb2312', 'gb18030', 'big5', 'big5-hkscs',
        'latin1', 'latin2', 'latin3', 'latin4', 'latin5',
        'cp1252', 'cp1251', 'cp1250', 'cp936', 'cp950',
        'iso-8859-1', 'iso-8859-2', 'iso-8859-15',
        'windows-1252', 'windows-1251', 'windows-1250',
        'ascii', 'us-ascii'
    ]

    @staticmethod
    def detect_encoding(file_path, sample_size=10000):
        """
        Detect file encoding using chardet

        Args:
            file_path (str): Path to the CSV file
            sample_size (int): Number of bytes to read for detection

        Returns:
            dict: Detection results with encoding, confidence, and language
        """
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(sample_size)
                result = chardet.detect(raw_data)
                return result
        except Exception as e:
            return {'encoding': None, 'confidence': 0, 'error': str(e)}

    @staticmethod
    def test_encoding(file_path, encoding):
        """
        Test if a file can be read with a specific encoding

        Args:
            file_path (str): Path to the CSV file
            encoding (str): Encoding to test

        Returns:
            dict: Test results with success status and error info
        """
        try:
            df = pd.read_csv(file_path, encoding=encoding, nrows=5)
            return {
                'success': True,
                'encoding': encoding,
                'rows_read': len(df),
                'columns': len(df.columns),
                'column_names': df.columns.tolist(),
                'sample_data': df.head(2).to_dict('records')
            }
        except Exception as e:
            return {
                'success': False,
                'encoding': encoding,
                'error': str(e)
            }

    @classmethod
    def analyze_file(cls, file_path):
        """
        Comprehensive analysis of CSV file encoding

        Args:
            file_path (str): Path to the CSV file

        Returns:
            dict: Complete analysis results
        """
        if not os.path.exists(file_path):
            return {'error': f'File not found: {file_path}'}

        results = {
            'file_path': file_path,
            'file_size': os.path.getsize(file_path),
            'detection_results': {},
            'encoding_tests': {},
            'recommendations': []
        }

        print(f"🔍 Analyzing file: {file_path}")
        print(f"📁 File size: {results['file_size']:,} bytes")
        print()

        # Auto-detect encoding
        print("🤖 Auto-detecting encoding...")
        detection = cls.detect_encoding(file_path)
        results['detection_results'] = detection

        if detection.get('encoding'):
            print(f"   Detected: {detection['encoding']} (confidence: {detection.get('confidence', 0):.2f})")
        else:
            print(f"   Detection failed: {detection.get('error', 'Unknown error')}")
        print()

        # Test common encodings
        print("🧪 Testing common encodings...")
        successful_encodings = []

        for encoding in cls.COMMON_ENCODINGS:
            test_result = cls.test_encoding(file_path, encoding)
            results['encoding_tests'][encoding] = test_result

            if test_result['success']:
                successful_encodings.append(encoding)
                print(f"   ✅ {encoding}: {test_result['rows_read']} rows, {test_result['columns']} columns")
            else:
                print(f"   ❌ {encoding}: {test_result['error']}")

        print()

        # Generate recommendations
        if successful_encodings:
            results['recommendations'] = cls._generate_recommendations(
                detection, successful_encodings, results['encoding_tests']
            )

            print("💡 Recommendations:")
            for i, rec in enumerate(results['recommendations'], 1):
                print(f"   {i}. Use '{rec['encoding']}' - {rec['reason']}")
        else:
            print("⚠️  No suitable encoding found. File may be corrupted or not a valid CSV.")

        return results

    @classmethod
    def _generate_recommendations(cls, detection, successful_encodings, test_results):
        """Generate encoding recommendations based on test results"""
        recommendations = []

        # If auto-detection worked and is in successful list
        detected_encoding = detection.get('encoding')
        if detected_encoding and detected_encoding in successful_encodings:
            confidence = detection.get('confidence', 0)
            if confidence > 0.8:
                recommendations.append({
                    'encoding': detected_encoding,
                    'reason': f'Auto-detected with high confidence ({confidence:.2f})'
                })
            elif confidence > 0.6:
                recommendations.append({
                    'encoding': detected_encoding,
                    'reason': f'Auto-detected with medium confidence ({confidence:.2f})'
                })

        # Prefer UTF-8 if it works
        if 'utf-8' in successful_encodings:
            recommendations.append({
                'encoding': 'utf-8',
                'reason': 'UTF-8 is the most universal encoding'
            })

        # Prefer UTF-8 with BOM if it works
        if 'utf-8-sig' in successful_encodings:
            recommendations.append({
                'encoding': 'utf-8-sig',
                'reason': 'UTF-8 with BOM, good for Excel compatibility'
            })

        # Add other successful encodings
        for encoding in ['latin1', 'cp1252', 'gbk', 'big5']:
            if encoding in successful_encodings and encoding not in [r['encoding'] for r in recommendations]:
                recommendations.append({
                    'encoding': encoding,
                    'reason': f'Alternative encoding that works'
                })

        return recommendations[:3]  # Return top 3 recommendations


def main():
    """Command line interface for the encoding detector"""
    if len(sys.argv) != 2:
        print("Usage: python encoding_detector.py <csv_file_path>")
        print("Example: python encoding_detector.py data/sample.csv")
        sys.exit(1)

    file_path = sys.argv[1]

    print("=" * 60)
    print("🔧 CSV Encoding Detection Tool")
    print("=" * 60)

    results = CSVEncodingDetector.analyze_file(file_path)

    if 'error' in results:
        print(f"❌ Error: {results['error']}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("📊 Analysis Complete!")

    if results['recommendations']:
        print(f"\n🎯 Best encoding to use: {results['recommendations'][0]['encoding']}")

    print("\n🔧 To use in your code:")
    if results['recommendations']:
        best_encoding = results['recommendations'][0]['encoding']
        print(f"   df = pd.read_csv('{file_path}', encoding='{best_encoding}')")
    else:
        print("   # Try manual encoding or file cleanup")


if __name__ == "__main__":
    main()