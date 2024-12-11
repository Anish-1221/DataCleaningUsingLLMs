import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
import json
from datetime import datetime
import time

def create_resilient_session():
    """Create a requests session with retry logic"""
    session = requests.Session()
    
    # Configure retry strategy
    retry_strategy = Retry(
        total=3,  # number of retries
        backoff_factor=1,  # wait 1, 2, 4 seconds between retries
        status_forcelist=[500, 502, 503, 504],  # retry on these HTTP status codes
        allowed_methods=["POST"]  # only retry POST requests
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

def detect_errors_in_row(row, row_number, session=None):
    """Send a single row to LLM for error detection"""
    if session is None:
        session = create_resilient_session()
        
    # More comprehensive but focused prompt
    row_data = (
        f"You are validating healthcare data. Check this row for ALL possible errors including:\n"
        "- Wrong formats (phone: (XXX) XXX-XXXX, ZIP: 5 or 9 digits)\n"
        "- Invalid values (state must be 2-letter code, dates must be MM/DD/YYYY)\n"
        "- Missing or 'nan' values (Note: 'Not Available' is valid)\n"
        "- Typos and misspellings\n"
        "- Invalid characters\n"
        "- Inconsistent formatting\n"
        "- Numbers in text fields\n"
        "- Text in numeric fields\n\n"
        f"ROW DATA:\n"
        f"Facility ID: {row['Facility ID']} (6 chars)\n"
        f"Facility Name: {row['Facility Name']} (check spelling)\n"
        f"Address: {row['Address']} (valid street address)\n"
        f"City/Town: {row['City/Town']} (no numbers)\n"
        f"State: {row['State']} (2-letter code)\n"
        f"ZIP: {row['ZIP Code']} (5 or 9 digits)\n"
        f"County: {row['County/Parish']} (spelling)\n"
        f"Phone: {row['Telephone Number']} ((XXX) XXX-XXXX)\n"
        f"Condition: {row['Condition']} (medical term)\n"
        f"Measure ID: {row['Measure ID']}\n"
        f"Measure Name: {row['Measure Name']}\n"
        f"Score: {row['Score']} ('Not Available' or numeric)\n"
        f"Sample: {row['Sample']} ('Not Available' or numeric)\n"
        f"Footnote: {row['Footnote']} (numeric)\n"
        f"Start Date: {row['Start Date']} (MM/DD/YYYY)\n"
        f"End Date: {row['End Date']} (MM/DD/YYYY)\n\n"
        f"If NO errors found, respond with: {{\n"
        f'"row_number": {row_number},\n'
        '"error_detection": "no error",\n'
        '"errors": [],\n'
        '"reasoning": "No errors found"\n'
        "}}\n\n"
        f"If errors found, respond with: {{\n"
        f'"row_number": {row_number},\n'
        '"error_detection": "error",\n'
        '"errors": [{"field": "field name", "error_type": "the type of the error", "description": "detailed description"}],\n'
        '"reasoning": "brief explanation"\n'
        "}}"
    )

    try:
        response = session.post(
            'http://localhost:11434/api/generate',
            json={
                "model": "neural-chat",
                "prompt": row_data,
                "temperature": 0.1,
                "max_tokens": 500,
                "stop": ["}"]
            },
            stream=True,
            timeout=20
        )

        full_response = ""
        for line in response.iter_lines():
            if line:
                try:
                    json_response = json.loads(line.decode('utf-8'))
                    if 'response' in json_response:
                        full_response += json_response['response']
                except json.JSONDecodeError:
                    continue

        # Ensure the response ends with a closing brace
        if not full_response.strip().endswith("}"):
            full_response += "}"

        try:
            parsed_response = json.loads(full_response)
            # Ensure the response has all required fields
            return {
                "row_number": row_number,
                "error_detection": parsed_response.get('error_detection', 'error'),
                "errors": parsed_response.get('errors', []),
                "reasoning": parsed_response.get('reasoning', 'No specific reasoning provided')
            }
        except json.JSONDecodeError:
            return {
                "row_number": row_number,
                "error_detection": "error",
                "errors": [{
                    "field": "general",
                    "error_type": "parse_error",
                    "description": "Failed to parse model response"
                }],
                "reasoning": full_response.strip()
            }

    except Exception as e:
        return {
            "row_number": row_number,
            "error_detection": "error",
            "errors": [{
                "field": "general",
                "error_type": "processing_error",
                "description": str(e)
            }],
            "reasoning": f"Error processing row: {str(e)}"
        }

def validate_field_length(value, max_length, field_name):
    """Utility function to validate field length"""
    if value and len(str(value)) > max_length:
        return {
            "field": field_name,
            "error_type": "length_violation",
            "description": f"Value exceeds maximum length of {max_length} characters"
        }
    return None

def validate_date_field(value, field_name):
    """Utility function to validate date fields"""
    try:
        if pd.isna(value):
            return None
        date = pd.to_datetime(value)
        if date.year < 1900 or date.year > datetime.now().year + 1:
            return {
                "field": field_name,
                "error_type": "invalid_date",
                "description": f"Date {value} is outside acceptable range"
            }
    except Exception:
        return {
            "field": field_name,
            "error_type": "invalid_date_format",
            "description": f"Invalid date format: {value}"
        }
    return None

def analyze_csv(csv_path, max_rows=None, delay_between_requests=0.5):
    """Analyze healthcare facility CSV file row by row with rate limiting"""
    try:
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Limit rows if specified
        if max_rows:
            df = df.head(max_rows)
        
        results = []
        total_rows = len(df)
        session = create_resilient_session()  # Create one session to reuse
        
        # Process each row
        for index, row in df.iterrows():
            print(f"Processing row {index + 1} of {total_rows}...")
            
            # Add delay before processing each row (except the first one)
            if index > 0:
                time.sleep(delay_between_requests)
            
            # Get PHI-2 analysis
            phi2_result = detect_errors_in_row(row, index + 1, session=session)
            
            # Perform additional validation checks
            additional_errors = []
            
            # Validate field lengths
            field_lengths = {
                'Facility ID': 6,
                'Facility Name': 72,
                'Address': 51,
                'City/Town': 20,
                'State': 2,
                'County/Parish': 25,
                'Telephone Number': 14,
                'Condition': 35,
                'Measure ID': 19,
                'Measure Name': 168,
                'Score': 13,
                'Sample': 13,
                'Footnote': 9
            }
            
            for field, length in field_lengths.items():
                error = validate_field_length(row.get(field), length, field)
                if error:
                    additional_errors.append(error)
            
            # Validate dates
            for date_field in ['Start Date', 'End Date']:
                error = validate_date_field(row.get(date_field), date_field)
                if error:
                    additional_errors.append(error)
            
            # Combine PHI-2 and additional validation results
            if additional_errors:
                phi2_result['error_detection'] = 'error'
                phi2_result['errors'].extend(additional_errors)
            
            results.append(phi2_result)
            
            # Print progress
            print(f"Row {index + 1} result: {phi2_result['error_detection']}")
            if phi2_result['error_detection'] == 'error':
                print(f"Found {len(phi2_result['errors'])} errors")
                print(f"Reasoning: {phi2_result['reasoning']}\n")
        
        return results

    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        return None

def save_results(results, output_path='fullMD_10pct.json'):
    """Save analysis results to a JSON file with summary statistics"""
    if not results:
        return
    
    # Calculate summary statistics
    total_rows = len(results)
    error_rows = sum(1 for r in results if r['error_detection'] == 'error')
    error_types = {}
    field_errors = {}
    
    for result in results:
        if result['error_detection'] == 'error':
            for error in result.get('errors', []):
                # Count error types
                error_type = error.get('error_type')
                error_types[error_type] = error_types.get(error_type, 0) + 1
                
                # Count errors by field
                field = error.get('field')
                field_errors[field] = field_errors.get(field, 0) + 1
    
    # Create summary
    summary = {
        'total_rows_analyzed': total_rows,
        'rows_with_errors': error_rows,
        'error_rate': (error_rows / total_rows) * 100 if total_rows > 0 else 0,
        'error_types_frequency': error_types,
        'errors_by_field': field_errors
    }
    
    # Save detailed results and summary
    output = {
        'summary': summary,
        'detailed_results': results
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Print summary
    print("\nAnalysis Summary:")
    print(f"Total rows analyzed: {total_rows}")
    print(f"Rows with errors: {error_rows}")
    print(f"Error rate: {summary['error_rate']:.2f}%")
    print("\nMost common error types:")
    for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
        print(f"- {error_type}: {count}")
    print("\nErrors by field:")
    for field, count in sorted(field_errors.items(), key=lambda x: x[1], reverse=True):
        print(f"- {field}: {count}")
    print(f"\nDetailed results have been saved to '{output_path}'")

if __name__ == "__main__":
    # Start with a small number of rows for testing
    results = analyze_csv("data_10pct_errors.csv", max_rows=2000, delay_between_requests=0.0)
    
    if results:
        save_results(results)