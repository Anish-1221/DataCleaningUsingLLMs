# Data Cleaning Using LLMs

## Overview
This project focuses on using large language models (LLMs) for error detection and correction in healthcare data. It employs Intel's neural-chat model available on Ollama2 to detect and correct errors introduced into a dataset. The dataset includes healthcare-related attributes such as Facility ID, Facility Name, Address, and more. Errors have been introduced into 10% of the total rows, with each row possibly containing multiple errors. While the datasets provided have around 23k rows, it is suggested to truncate them and use about 2k rows for efficiency.

---

## Data Description
### Dataset Information
- **Folder Name:** `Test File`
- **File Types:** CSV
- **Total Rows:** 23,705
- **Files in the Folder:**
  - Ground truth data
  - Data file with errors

### Columns in the Dataset
| Column Name         | Data Type    | Length |
|---------------------|--------------|--------|
| Facility ID         | Char         | 6      |
| Facility Name       | Char         | 72     |
| Address             | Char         | 51     |
| City/Town           | Char         | 20     |
| State               | Char         | 2      |
| ZIP Code            | Num          | 8      |
| County/Parish       | Char         | 25     |
| Telephone Number    | Char         | 14     |
| Condition           | Char         | 35     |
| Measure ID          | Char         | 19     |
| Measure Name        | Char         | 168    |
| Score               | Char         | 13     |
| Sample              | Char         | 13     |
| Footnote            | Char         | 9      |
| Start Date          | Date         |        |
| End Date            | Date         |        |

### Types of Errors Introduced
#### Facility Name (Char(72))
- **Spelling errors:** e.g., `Hospitl` instead of `Hospital`.
- **Special characters:** Including symbols like `#` or `!`.
- **Missing values:** Leaving it blank.

#### Address (Char(51))
- **Typos:** e.g., `Strret` instead of `Street`.
- **Swapping:** Replacing with unrelated addresses.
- **Special characters:** Adding invalid symbols.
- **Missing values:** Leaving it blank.

#### City/Town (Char(20))
- **Typos:** e.g., `Dalass` instead of `Dallas`.
- **Special characters:** e.g., `Austin%`.
- **Swapping:** Using unrelated names.
- **Missing values:** Leaving it blank.

#### State (Char(2))
- **Invalid abbreviations:** e.g., `XY` instead of `TX`.
- **Case sensitivity issues:** e.g., `tx` instead of `TX`.
- **Swapping:** Replacing with unrelated combinations.
- **Missing values.**

#### ZIP Code (Num(8))
- **Invalid characters:** e.g., `78X67`.
- **Format issues:** Introducing hyphens or spaces.
- **Out-of-range values:** Beyond valid ZIP codes.
- **Missing values.**

#### County/Parish (Char(25))
- **Typos:** e.g., `Travis` becoming `Trviz`.
- **Special characters or numbers.**
- **Missing values.**

#### Telephone Number (Char(14))
- **Formatting errors:** e.g., `5123456789` instead of `(512) 345-6789`.
- **Invalid characters:** e.g., `ABC12345678`.
- **Missing values.**

#### Condition (Char(35))
- **Typographical errors:** e.g., `Pnemonia` instead of `Pneumonia`.
- **Special characters or numbers.**
- **Missing values.**

#### Measure ID (Char(19))
- **Typographical errors.**
- **Invalid characters.**
- **Case sensitivity issues.**
- **Extra characters.**
- **Missing values.**

#### Measure Name (Char(168))
- **Typographical errors.**
- **Special characters or numbers.**
- **Missing values.**

#### Score (Char(13))
- **Non-numeric values:** Adding text instead of valid numeric scores.
- **Out-of-range values.**
- **Inconsistent formats:** Mixing percentages with raw numbers.
- **Missing values.**

#### Sample (Char(13))
- **Invalid numeric values:** Negative numbers, non-integers, or zeros.
- **Non-numeric text.**
- **Extreme values:** Unrealistic sample sizes.
- **Format inconsistency.**
- **Missing values.**

#### Footnote (Char(9))
- **Invalid numeric entries.**
- **Out-of-scope values.**
- **Duplications.**
- **Missing values.**

#### Start Date & End Date (Date)
- **Invalid formats:** e.g., `YYYY/DD/MM`.
- **Out-of-range dates.**
- **Invalid strings:** e.g., `Start123`.
- **Missing values.**

---

## Error Detection Methodology
### Stages of Error Detection
#### Stage 1: No Metadata (MD)
- Column names masked as `Attribute 1`, `Attribute 2`, etc.
- **Prompt (used in `detector_noMD.py`):**
```python
row_data = (
    f"You are validating healthcare data. Check this row for ALL possible errors:\n"
    f"ROW DATA:\n"
    f"Attribute 1: {row['Facility ID']}\n"
    f"Attribute 2: {row['Facility Name']}\n"
    f"Attribute 3: {row['Address']}\n"
    f"Attribute 4: {row['City/Town']}\n"
    f"Attribute 5: {row['State']}\n"
    f"Attribute 6: {row['ZIP Code']}\n"
    f"Attribute 7: {row['County/Parish']}\n"
    f"Attribute 8: {row['Telephone Number']}\n"
    f"Attribute 9: {row['Condition']}\n"
    f"Attribute 10: {row['Measure ID']}\n"
    f"Attribute 12: {row['Measure Name']}\n"
    f"Attribute 13: {row['Score']}\n"
    f"Attribute 14: {row['Sample']}\n"
    f"Attribute 15: {row['Footnote']}\n"
    f"Attribute 16: {row['Start Date']}\n"
    f"Attribute 17: {row['End Date']}\n"
    f"If NO errors found, respond with: {{\n"
    f'"row_number": {row_number},\n'
    '"error_detection": "no error",\n'
    '"errors": [],\n'
    '"reasoning": "No errors found"\n'
    "}}\n\n"
    f"If errors found, respond with: {{\n"
    f'"row_number": {row_number},\n'
    '"error_detection": "error",\n'
    '"errors": [{{"field": "field name", "error_type": "the type of the error", "description": "detailed description"}}],\n'
    '"reasoning": "brief explanation"\n'
    "}}"
)
```

#### Stage 2: With Column Names
- Column names provided without expected value metadata.
- **Prompt (used in `detector_columnMD.py`):**
```python
row_data = (
    f"You are validating healthcare data. Check this row for ALL possible errors:\n"
    f"ROW DATA:\n"
    f"Facility ID: {row['Facility ID']}\n"
    f"Facility Name: {row['Facility Name']}\n"
    f"Address: {row['Address']}\n"
    f"City/Town: {row['City/Town']}\n"
    f"State: {row['State']}\n"
    f"ZIP: {row['ZIP Code']}\n"
    f"County: {row['County/Parish']}\n"
    f"Phone: {row['Telephone Number']}\n"
    f"Condition: {row['Condition']}\n"
    f"Measure ID: {row['Measure ID']}\n"
    f"Measure Name: {row['Measure Name']}\n"
    f"Score: {row['Score']}\n"
    f"Sample: {row['Sample']}\n"
    f"Footnote: {row['Footnote']}\n"
    f"Start Date: {row['Start Date']}\n"
    f"End Date: {row['End Date']}\n"
    f"If NO errors found, respond with: {{\n"
    f'"row_number": {row_number},\n'
    '"error_detection": "no error",\n'
    '"errors": [],\n'
    '"reasoning": "No errors found"\n'
    "}}\n\n"
    f"If errors found, respond with: {{\n"
    f'"row_number": {row_number},\n'
    '"error_detection": "error",\n'
    '"errors": [{{"field": "field name", "error_type": "the type of the error", "description": "detailed description"}}],\n'
    '"reasoning": "brief explanation"\n'
    "}}"
)
```

#### Stage 3: Full Metadata
- Full metadata provided, including expected values and error types.
- **Prompt (used in `detector_fullMD.py`):**
```python
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
    '"errors": [{{"field": "field name", "error_type": "the type of the error", "description": "detailed description"}}],\n'
    '"reasoning": "brief explanation"\n'
    "}}"
)
```
Prompts are structured to output a JSON file storing error details.

---

## Error Correction Methodology
- Errors are corrected using the `correct_errors.py` script.
- Prompts are broken into multiple parts to:
  1. Reformat dates to `MM/DD/YYYY`.
  2. Format phone numbers as `(XXX) XXX-XXXX`.
  3. Use verified facility information to fill missing fields.
  4. Fix typos and remove invalid characters.

### Important Information:
1. **Detection Requirement:** If you want to use this correction script, ensure you have first run the detection scripts to generate the required JSON files containing error details.
2. **Adaptability:** While the provided code demonstrates one way to handle error correction, you do not need to use these exact implementations. You can extract the essence from the prompts and modify them to fit your models.


### Correction Prompt (used in `correct_errors.py`):
```python
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
   - Medical terms: Fix only clear spelling errors
"""
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
}
"""

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
```  
Prompts provide outputs in JSON format, including corrected fields and explanations.

---

## How to Run the Project
1. Install Dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Pull the neural-chat Model:
   ```bash
   ollama pull neural-chat
   ```

3. Update File Paths:
   Ensure file paths in the code match your environment.

4. Run Error Detection:
   Use the error detection prompts (`detector_noMD.py`, `detector_columnMD.py`, `detector_fullMD.py`).

5. Run Error Correction:
   Use the correction prompt provided in `correct_errors.py`.

---

## Additional Notes
- For quicker testing, use a small subset of rows instead of the full dataset.
- All prompts are included in this document to avoid hunting through the code.
- Prompts are designed for structured, JSON-compatible outputs.

---

## Dependencies
- Python 3.8+
- Intel's neural-chat model on Ollama
