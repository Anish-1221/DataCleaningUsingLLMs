import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
import json
import time
from datetime import datetime
import re
import multiprocessing as mp
from functools import partial

def create_resilient_session():
    """Create a requests session with retry logic"""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["POST"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def get_facility_examples(df, facility_id):
    """Get all rows with the same Facility ID as they contain correct facility information"""
    matching_rows = df[df['Facility ID'] == facility_id].drop_duplicates([
        'Facility Name', 'Address', 'City/Town', 'State', 
        'ZIP Code', 'County/Parish', 'Telephone Number'
    ])
    
    if not matching_rows.empty:
        facility_info = matching_rows.iloc[0]
        return {
            'facility_name': facility_info['Facility Name'],
            'address': facility_info['Address'],
            'city': facility_info['City/Town'],
            'state': facility_info['State'],
            'zip': facility_info['ZIP Code'],
            'county': facility_info['County/Parish'],
            'phone': facility_info['Telephone Number']
        }
    return None

def process_single_row(row_data, df, session, error_patterns, field_rules):
    """Process a single row of data"""
    result, row_number, row = row_data
    corrections_log = []
    
    facility_examples = get_facility_examples(df, row['Facility ID'])
    
    # Map error types to their descriptions
    formatted_errors = []
    for error in result['errors']:
        error_type = error['error_type'].lower().replace(' ', '_')
        error_desc = error_patterns.get(error_type, 'Unspecified error type')
        formatted_errors.append({
            'field': error['field'],
            'error_type': error_type,
            'description': error_desc
        })
    
    print(f"\nProcessing row {row_number}:")
    print(f"Original errors: {json.dumps(formatted_errors, indent=2)}")

    correction_requirements = """
1. DATES - CRITICAL:
   - ONLY reformat dates to MM/DD/YYYY, NEVER change the actual date
   - Examples of correct formatting:
     * "2023/01/15" → "01/15/2023"
     * "15-01-2023" → "01/15/2023"
     * "2023-01-15" → "01/15/2023"
   - For missing/invalid dates: use "Not Available"

2. PHONE NUMBERS - CRITICAL:
   - ONLY add formatting (XXX) XXX-XXXX to existing digits
   - NEVER change the actual digits
   - Example: "6405587230" → "(640) 558-7230"
   - WRONG: Do not change "6405587230" to a different number

3. MISSING VALUES:
   For Facility-related fields (Name, Address, City/Town, State, ZIP, County/Parish, Phone):
   - IF verified facility info exists: Use those EXACT values
   - IF NO verified info: Use "Not Available"
   
   For other fields:
   - Score and Sample: Use "Not Available"
   - Footnote: Use "1"
   - Other fields: Use "Not Available"

4. Other Rules:
   - Fix obvious typos in text (e.g., "Helth" → "Health")
   - Format ZIP codes: Ensure 5-digit format
   - State codes: Use correct 2-letter format, preserve same state
   - City/Town and County: Remove only numbers, preserve names
   - Clean up invalid characters from all the column values
   - Medical terms: Fix only clear spelling errors"""

    response_format = """
{
    "corrected_fields": {
        "field_name": "corrected_value"
    },
    "correction_details": {
        "field_name": {
            "original": "original_value",
            "corrected": "corrected_value",
            "reason": "explanation",
            "error_pattern": "pattern_name"
        }
    }
}"""

    prompt_parts = [
        "You are a healthcare data correction expert. Fix the following data according to these rules:",
        f"\nCURRENT ERRORS FOUND:\n{json.dumps(formatted_errors, indent=2)}",
        f"\nCURRENT ROW DATA:\n{json.dumps(row.to_dict(), indent=2)}",
        f"\nFIELD RULES:\n{json.dumps(field_rules, indent=2)}",
        f"\nCORRECTION REQUIREMENTS:{correction_requirements}",
        "\nFor each correction, provide:",
        "1. The field being corrected",
        "2. The original value",
        "3. The corrected value",
        "4. The reason for the correction based on the error pattern identified",
        f"\nRespond ONLY with a JSON object in this exact format:{response_format}"
    ]

    correction_prompt = "\n".join(prompt_parts)
    
    if facility_examples:
        correction_prompt += f"\n\nVERIFIED FACILITY INFORMATION (USE THESE EXACT VALUES):\n{json.dumps(facility_examples, indent=2)}"

    try:
        response = session.post(
            'http://localhost:11434/api/generate',
            json={
                "model": "neural-chat",
                "prompt": correction_prompt,
                "temperature": 0.1,
                "max_tokens": 1000,
                "stop": ["}"]
            },
            stream=True,
            timeout=30
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

        if not full_response.strip().endswith("}"):
            full_response += "}"

        corrections = json.loads(full_response)
        row_corrections = {}
        
        # Process corrections
        for field, value in corrections.get('corrected_fields', {}).items():
            old_value = str(row[field])
            
            correction_detail = corrections.get('correction_details', {}).get(field, {})
            error_pattern = correction_detail.get('error_pattern', 'unknown')
            
            row_corrections[field] = value
            
            correction_record = {
                "row_number": row_number,
                "field": field,
                "original_value": old_value,
                "corrected_value": str(value),
                "error_type": next((error['error_type'] for error in result['errors'] 
                                  if error['field'] == field), "unknown"),
                "error_pattern": error_pattern,
                "error_description": error_patterns.get(error_pattern, "Unknown error pattern"),
                "correction_reason": correction_detail.get('reason', 
                                   "Corrected based on field rules and requirements")
            }
            
            corrections_log.append(correction_record)
            
            print(f"Corrected {field}: '{old_value}' → '{value}'")
            print(f"Error Pattern: {error_pattern}")
            print(f"Description: {correction_record['error_description']}")
            print(f"Reason: {correction_record['correction_reason']}")

        return row_number, row_corrections, corrections_log

    except Exception as e:
        print(f"Error processing row {row_number}: {str(e)}")
        return row_number, None, []

def correct_errors(input_csv_path, error_json_path, output_csv_path=None):
    """Correct errors and track all corrections made"""
    print("Reading files...")
    df = pd.read_csv(input_csv_path)
    with open(error_json_path, 'r') as f:
        results = json.load(f)
    
    session = create_resilient_session()
    corrected_df = df.copy()
    
    # Initialize error patterns and field rules properly
    error_patterns = {
        'missing_value': 'Field is empty or contains nan',
        'length_violation': 'Value length does not meet requirements',
        'nan_value': 'Contains nan instead of valid data',
        'typo': 'Contains spelling errors or typos',
        'invalid_value': 'Value does not meet field requirements',
        'invalid_format': 'Format does not match requirements',
        'numeric_value_in_text': 'Contains numbers where text is expected',
        'invalid_date_format': 'Date format does not match MM/DD/YYYY',
        'invalid_characters': 'Contains invalid or special characters'
    }
    
    field_rules = {
        'Facility ID': {
            'type': 'string',
            'min_length': 6,
            'max_length': 6,
            'required': True,
            'format': 'alphanumeric'
        },
        'Facility Name': {
            'type': 'string',
            'required': True,
            'format': 'text'
        },
        'Address': {
            'type': 'string',
            'required': True,
            'format': 'address'
        },
        'City/Town': {
            'type': 'string',
            'required': True,
            'format': 'text',
            'no_numbers': True
        },
        'State': {
            'type': 'string',
            'length': 2,
            'required': True,
            'format': 'state_code'
        },
        'ZIP Code': {
            'type': 'string',
            'required': True,
            'format': 'zip',
            'valid_lengths': [5, 9]
        },
        'County/Parish': {
            'type': 'string',
            'required': True,
            'format': 'text'
        },
        'Telephone Number': {
            'type': 'string',
            'required': True,
            'format': 'phone',
            'pattern': r'\(\d{3}\) \d{3}-\d{4}'
        },
        'Condition': {
            'type': 'string',
            'required': True,
            'format': 'medical_term'
        },
        'Measure ID': {
            'type': 'string',
            'required': True,
            'format': 'measure_id'
        },
        'Measure Name': {
            'type': 'string',
            'required': True,
            'format': 'text'
        },
        'Score': {
            'type': 'string',
            'required': True,
            'format': 'score',
            'valid_formats': ['numeric', 'Not Available']
        },
        'Sample': {
            'type': 'string',
            'required': True,
            'format': 'sample',
            'valid_formats': ['numeric', 'Not Available']
        },
        'Footnote': {
            'type': 'numeric',
            'required': False,
            'format': 'numeric'
        },
        'Start Date': {
            'type': 'string',
            'required': True,
            'format': 'date',
            'pattern': r'\d{2}/\d{2}/\d{4}'
        },
        'End Date': {
            'type': 'string',
            'required': True,
            'format': 'date',
            'pattern': r'\d{2}/\d{2}/\d{4}'
        }
    }
    
    detailed_results = results.get('detailed_results', [])
    print(f"Processing {len(detailed_results)} rows...")
    
    # Prepare data for parallel processing
    row_data = []
    for result in detailed_results:
        if result.get('error_detection') != 'no error' and result.get('errors'):
            row_number = result['row_number']
            row = df.iloc[row_number - 1]
            row_data.append((result, row_number, row))
    
    # Create a partial function with fixed arguments
    process_row = partial(process_single_row, df=df, session=session, 
                         error_patterns=error_patterns, field_rules=field_rules)
    
    # Initialize multiprocessing pool
    num_processes = min(4, mp.cpu_count())
    print(f"Using {num_processes} processes")
    
    corrections_log = []
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(process_row, row_data)
        
        # Process results
        for row_number, row_corrections, row_logs in results:
            if row_corrections:
                for field, value in row_corrections.items():
                    corrected_df.at[row_number - 1, field] = value
                corrections_log.extend(row_logs)
    
    # Generate output paths and save results as before...
    if output_csv_path is None:
        output_csv_path = input_csv_path.replace('.csv', '_corrected.csv')
    
    corrections_json_path = output_csv_path.replace('.csv', '_corrections.json')
    
    # Save corrected CSV
    corrected_df.to_csv(output_csv_path, index=False)
    
    # Prepare corrections summary
    corrections_summary = {
        "metadata": {
            "original_file": input_csv_path,
            "correction_date": datetime.now().isoformat(),
            "total_rows": len(df),
            "rows_corrected": len(set(c["row_number"] for c in corrections_log)),
            "total_corrections": len(corrections_log)
        },
        "error_types": {error['error_type']: len([c for c in corrections_log 
                                                if c['error_type'] == error['error_type']]) 
                       for error in [item for sublist in [r['errors'] for r in detailed_results 
                                                        if r.get('errors')] for item in sublist]},
        "corrections_by_field": {field: len([c for c in corrections_log if c['field'] == field]) 
                               for field in set(c['field'] for c in corrections_log)},
        "corrections": corrections_log
    }
    
    with open(corrections_json_path, 'w') as f:
        json.dump(corrections_summary, f, indent=2)
    
    print(f"\nCorrected data saved to: {output_csv_path}")
    print(f"Corrections log saved to: {corrections_json_path}")
    
    print("\nCorrection Summary:")
    print(f"Total rows in dataset: {len(df)}")
    print(f"Rows with corrections: {corrections_summary['metadata']['rows_corrected']}")
    print(f"Total corrections made: {corrections_summary['metadata']['total_corrections']}")
    print("\nCorrections by error type:")
    for error_type, count in corrections_summary['error_types'].items():
        print(f"- {error_type}: {count}")
    
    return corrected_df, corrections_summary

if __name__ == "__main__":
    corrected_df, corrections_summary = correct_errors(
        input_csv_path="data_10pct_errors.csv",
        error_json_path="fullMD_10pct.json",
        output_csv_path="fullMD_10pct.csv"
    )