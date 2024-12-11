import pandas as pd
import numpy as np
import random
import string
from datetime import datetime, timedelta

class ErrorGenerator:
    def __init__(self, error_rate=0.3):
        self.error_rate = error_rate
        self.state_codes = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 
                           'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 
                           'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 
                           'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 
                           'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY']

    def _should_introduce_error(self):
        return random.random() < self.error_rate

    def _random_string(self, length):
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

    def _introduce_typos(self, text, error_count=1):
        if not isinstance(text, str):
            return text
        text = list(text)
        for _ in range(error_count):
            pos = random.randint(0, len(text)-1)
            text[pos] = random.choice(string.ascii_letters)
        return ''.join(text)

    def facility_name_error(self, value):
        error_types = [
            lambda x: self._introduce_typos(x),
            lambda x: x + random.choice('!@#$%^&*'),
            lambda x: ''
        ]
        return random.choice(error_types)(value)

    def address_error(self, value):
        error_types = [
            lambda x: self._introduce_typos(x),
            lambda x: self._random_string(30),
            lambda x: x + random.choice('!@#$%^&*'),
            lambda x: ''
        ]
        return random.choice(error_types)(value)

    def city_error(self, value):
        error_types = [
            lambda x: self._introduce_typos(x),
            lambda x: x + '%',
            lambda x: random.choice(['NewCity', 'OtherTown', 'SomePlace']),
            lambda x: ''
        ]
        return random.choice(error_types)(value)

    def state_error(self, value):
        error_types = [
            lambda x: 'XY',
            lambda x: x.lower(),
            lambda x: random.choice(string.ascii_uppercase) + random.choice(string.ascii_uppercase),
            lambda x: ''
        ]
        return random.choice(error_types)(value)

    def zip_error(self, value):
        error_types = [
            lambda x: f"{random.randint(10000, 99999)}X",
            lambda x: f"{random.randint(10000, 99999)}-{random.randint(1000, 9999)}",
            lambda x: str(random.randint(100000000, 999999999)),
            lambda x: ''
        ]
        return str(random.choice(error_types)(value))

    def county_error(self, value):
        error_types = [
            lambda x: self._introduce_typos(x),
            lambda x: x + str(random.randint(1, 99)),
            lambda x: ''
        ]
        return random.choice(error_types)(value)

    def phone_error(self, value):
        error_types = [
            lambda x: f"{random.randint(100, 999)}{random.randint(100, 999)}{random.randint(1000, 9999)}",
            lambda x: f"ABC{random.randint(10000000, 99999999)}",
            lambda x: ''
        ]
        return random.choice(error_types)(value)

    def condition_error(self, value):
        error_types = [
            lambda x: self._introduce_typos(x),
            lambda x: x + random.choice(string.digits),
            lambda x: ''
        ]
        return random.choice(error_types)(value)

    def measure_id_error(self, value):
        error_types = [
            lambda x: x.replace('_', '-'),
            lambda x: x + '#',
            lambda x: x.lower(),
            lambda x: '',
            lambda x: x + '!'
        ]
        return random.choice(error_types)(value)

    def measure_name_error(self, value):
        error_types = [
            lambda x: self._introduce_typos(x),
            lambda x: x + random.choice('!@#$%^&*'),
            lambda x: ''
        ]
        return random.choice(error_types)(value)

    def score_error(self, value):
        error_types = [
            lambda x: random.choice(['high', 'low', 'medium']),
            lambda x: str(random.randint(-100, -1)),
            lambda x: f"{random.randint(0, 100)}%",
            lambda x: '',
            lambda x: f"Score: {random.randint(0, 100)}"
        ]
        return random.choice(error_types)(value)

    def sample_error(self, value):
        error_types = [
            lambda x: str(random.randint(-1000, -1)),
            lambda x: 'Not Available',
            lambda x: str(random.randint(1000000, 9999999)),
            lambda x: f"{random.randint(1000, 9999):,}",
            lambda x: ''
        ]
        return random.choice(error_types)(value)

    def footnote_error(self, value):
        error_types = [
            lambda x: random.choice(string.ascii_uppercase),
            lambda x: str(random.randint(100, 999)),
            lambda x: '1,1,2',
            lambda x: '',
            lambda x: ' '.join(str(random.randint(1, 9)) for _ in range(3))
        ]
        return random.choice(error_types)(value)

    def date_error(self, value):
        error_types = [
            lambda x: x.strftime('%Y/%d/%m'),
            lambda x: (datetime.now() + timedelta(days=random.randint(365*10, 365*20))).strftime('%Y-%m-%d'),
            lambda x: 'Invalid Date',
            lambda x: ''
        ]
        return random.choice(error_types)(value if isinstance(value, datetime) else datetime.now())

def introduce_errors(df, error_rate):
    """
    Introduce errors into the dataset at the specified error rate
    """
    error_gen = ErrorGenerator(error_rate)
    df_copy = df.copy()

    # Ensure ZIP Code column is string type before introducing errors
    if 'ZIP Code' in df_copy.columns:
        df_copy['ZIP Code'] = df_copy['ZIP Code'].astype(str)
    
    # Calculate number of rows to modify
    rows_to_modify = int(len(df) * error_rate)
    rows_to_modify_indices = random.sample(range(len(df)), rows_to_modify)
    
    error_functions = {
        'Facility Name': error_gen.facility_name_error,
        'Address': error_gen.address_error,
        'City/Town': error_gen.city_error,
        'State': error_gen.state_error,
        'ZIP Code': error_gen.zip_error,
        'County/Parish': error_gen.county_error,
        'Telephone Number': error_gen.phone_error,
        'Condition': error_gen.condition_error,
        'Measure ID': error_gen.measure_id_error,
        'Measure Name': error_gen.measure_name_error,
        'Score': error_gen.score_error,
        'Sample': error_gen.sample_error,
        'Footnote': error_gen.footnote_error,
        'Start Date': error_gen.date_error,
        'End Date': error_gen.date_error
    }
    
    for idx in rows_to_modify_indices:
        # Randomly select columns to modify for this row
        columns_to_modify = random.sample(list(error_functions.keys()), 
                                        random.randint(1, len(error_functions)))
        
        for col in columns_to_modify:
            if col in df_copy.columns:
                try:
                    df_copy.at[idx, col] = error_functions[col](df_copy.at[idx, col])
                except Exception as e:
                    print(f"Error modifying {col} at index {idx}: {str(e)}")
                    continue
    
    return df_copy

def main(input_file, output_prefix):
    """
    Main function to read input CSV and generate two error-containing versions
    """
    # Read the original CSV file
    df = pd.read_csv(input_file)
    
    # Generate 30% error version
    df_30pct = introduce_errors(df, 0.3)
    df_30pct.to_csv(f"{output_prefix}_30pct_errors.csv", index=False)
    
    # Generate 10% error version
    df_10pct = introduce_errors(df, 0.1)
    df_10pct.to_csv(f"{output_prefix}_10pct_errors.csv", index=False)

# Usage example:
main("ground_truth.csv", "data")