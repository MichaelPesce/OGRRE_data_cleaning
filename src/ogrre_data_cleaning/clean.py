import re
from pathlib import Path
from datetime import datetime
import torch
import numpy as np
import pandas as pd

from ogrre_data_cleaning.models.encoder import Encoder, Classifier
from ogrre_data_cleaning.models.dataloaders import HoleSize
from ogrre_data_cleaning.models.checkpoints import get_checkpoint_path

def string_to_date(s: str):
    """
    Converts a string to a date after removing non-date characters.
    """
    # If input is already a datetime object, return it
    if isinstance(s, datetime):
        return s
        
    if not isinstance(s, str):
        return None
    
    # Use regex to keep only valid date characters
    cleaned_string = re.sub(r"[^\d-]+", "", s)
    
    # Remove trailing dashes that might be formatting artifacts
    cleaned_string = cleaned_string.rstrip('-')
    
    # Handle edge cases like empty strings or invalid formats
    try:
        # Try various date formats
        date = datetime.strptime(cleaned_string, '%Y-%m-%d')
        if date:
            return date
        date = datetime.strptime(cleaned_string, '%m-%d-%Y')
        if date:
            return date
        date = datetime.strptime(cleaned_string, '%m/%d/%Y')
    except ValueError:
        return None

def string_to_float(s: str):
    """
    Converts a string to a float after removing non-numeric characters.
    
    Args:
        s (str): The string to convert.
        
    Returns:
        float: The converted float value.
        None: If the conversion fails or no valid number can be extracted.
    """
    # If input is already a float, return it
    if isinstance(s, float):
        return s
        
    if not isinstance(s, str):
        return None
    
    # Use regex to keep only valid numeric characters, including '-' for negatives and '.' for decimals
    cleaned_string = re.sub(r"[^\d.-]+", "", s)
    
    # Remove trailing dashes that might be formatting artifacts
    cleaned_string = cleaned_string.rstrip('-')
    
    # Handle edge cases like empty strings or invalid formats
    try:
        return float(cleaned_string)
    except ValueError:
        return None

def string_to_int(s: str):
    """
    Converts a string to an integer after removing non-numeric characters.
    
    Args:
        s (str): The string to convert.
        
    Returns:
        int: The converted integer value.
        None: If the conversion fails or no valid number can be extracted.
    """
    # If input is already an int, return it
    if isinstance(s, int):
        return s
        
    if not isinstance(s, str):
        return None
    
    # Use regex to keep only digits and '-' for negatives
    cleaned_string = re.sub(r"[^\d-]+", "", s)
    
    # Remove trailing dashes that might be formatting artifacts
    cleaned_string = cleaned_string.rstrip('-')
    
    # Handle edge cases like empty strings or invalid formats
    try:
        return int(cleaned_string)
    except ValueError:
        return None

def llm_clean(s, model_name='holesize', model_version='0'):
    """
    Converts a string to desired final data form for various pre-trained 
    language models.

    Args:
        s (str): The string to convert.
        model_name (str): The specific pre-trained model to use.
        model_version (str): Version of the model to use.
        
    Returns:
        pred (float): The cleaned output from the model.
    """
    # If input is already a cleaned value (assuming it's a string), return it
    if not isinstance(s, float):
        return s
    
    # Check devices
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    # Load pre-trained model checkpoint
    checkpoint_path = get_checkpoint_path(model_name, model_version)
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model_config = checkpoint['model_config']
    data_parameters = checkpoint['data_parameters']
    labels = checkpoint['dataset_labels']
    num_tokens = checkpoint['num_tokens']

    # Define model, loss function and optimizer
    model_encoder = Encoder(
        vocab_dim=num_tokens, # Normally vocab but here num of unicode characters
        sequence_dim=data_parameters['sequence_size'],
        embed_dim=model_config['emb_dim'],
        ff_dim=model_config['ff_dim'],
        num_heads=model_config['num_heads'],
        num_blocks=model_config['num_blocks'],
        dropout=model_config['dropout'],
        norm_eps=model_config['layer_eps'],
        device=device
        ).to(device)

    # Define classifier
    model_classifier = Classifier(
        model_encoder, 
        sequence_dim=data_parameters['sequence_size'], 
        embedding_dim=model_config['emb_dim'], 
        num_classes=len(labels)
        ).to(device)
    
    # Load pre-trained model weights
    model_classifier.load_state_dict(checkpoint['model_state_dict'])

    # Encode the input
    dataset = HoleSize(
        None, labels, max_length=data_parameters['sequence_size']
        )
    
    # Tokenize the input
    X = np.array(dataset.tokenize(s)).reshape((1, -1))
    # print(X)

    model_classifier.eval()
    with torch.no_grad():
        y_pred = model_classifier(torch.tensor(X).to(device)).argmax(-1).to('cpu').numpy()[0]
    
    return dataset.classes[y_pred]

def clean_date(date_str: str) -> datetime | None:
    """
    Clean and standardize date strings into datetime objects.
    
    Args:
        date_str: String containing a date
        
    Returns:
        datetime object if successful, None if invalid date
    """
    # If input is already a datetime object, return it
    if isinstance(date_str, datetime):
        return date_str
        
    if not date_str or date_str in ['N/A', 'illegible', '-', 'BEFORE', 'SAME AS BEFORE']:
        return None
        
    # Remove any extra whitespace
    date_str = date_str.strip()
    
    # List of formats to try, in order of most to least common
    formats = [
        '%Y/%m/%d',           # 2020/8/1
        '%Y-%m-%d',           # 1973-02-29
        '%m/%d/%Y',           # 7/30/1971
        '%m/%d/%y',           # 11/29/54
        '%m-%d-%Y',           # 10-17-1983
        '%m-%d-%y',           # 5-27-66
        '%d-%b-%y',           # 29-Jul-71
        '%B %d, %Y',          # July 7, 1977 (full month with space after comma)
        '%B %d %Y',           # July 7 1977 (full month without comma)
        '%b %d, %Y',          # March 30, 1963 (abbreviated month with space after comma)
        '%B %d,%Y',           # April 28,1958 (full month without space after comma)
        '%b. %d, %Y',         # Sept. 11, 1957
        '%d/%m/%Y',           # 17/18/95 (ambiguous, assumes DD/MM/YYYY)
        '%Y'                  # 1949
    ]
    
    # Clean up some common variations
    date_str = re.sub(r'\.(?=\s|$)', '', date_str)  # Remove trailing periods
    date_str = re.sub(r'\s+', ' ', date_str)        # Normalize spaces
    date_str = date_str.rstrip('-')                  # Remove trailing dashes
    
    # Normalize non-standard month abbreviations
    month_replacements = {
        'Sept.': 'Sep.',
        'Sept ': 'Sep ',
    }
    for old, new in month_replacements.items():
        date_str = date_str.replace(old, new)
    
    # Try each format
    for fmt in formats:
        try:
            date = datetime.strptime(date_str, fmt)
            
            # Handle two-digit years - assume years > 50 are in the 1900s, <= 50 are in the 2000s
            if fmt.endswith('%y'):
                year = date.year
                if year >= 2050:  # If year is 2050 or later (from parsing '50' to '99')
                    # Adjust to 1950-1999
                    date = date.replace(year=year-100)
            
            # Convert to common format
            date_str = date.strftime('%m/%d/%Y')
            return date_str
        except ValueError:
            continue
            
    # Handle special cases
    if re.match(r'^\d{3,4}$', date_str):  # Handle year-only entries
        try:
            year = int(date_str)
            if 1900 <= year <= 2100:  # Reasonable year range
                return datetime(year, 1, 1).strftime('%m/%d/%Y')
        except ValueError:
            pass
            
    return None

def clean_bool(checkbox_str: str):
    '''
    check if string is valid representation of boolean
    
    args:
        checkbox_str: string of checkbox field
    returns:
        boolean: True if the input represents a positive/checked value, False otherwise
    '''
    # If input is already a boolean, return it
    if isinstance(checkbox_str, bool):
        return checkbox_str
        
    # If input is None or empty, return False
    if checkbox_str is None or (isinstance(checkbox_str, str) and not checkbox_str.strip()):
        return False
        
    try:
        # Normalize the string
        checkbox_str = checkbox_str.strip().upper()
        
        # Values that should be interpreted as True
        true_values = ['X', 'YES', 'TRUE', 'T', 'Y', '1', '\u2611']

        # Evaluate length 1 string
        if len(checkbox_str) == 1:
            # Check for True values
            if checkbox_str in true_values:
                return True
            
            # Check for checkbox symbols       
            if ord(checkbox_str) == 9745:  # Checked box symbol
                return True
            elif ord(checkbox_str) == 9744:  # Unchecked box symbol
                return False

        # Evaluate length > 1 strings
        else:
            # Check for True values
            for true_char in true_values:
                if true_char in checkbox_str:
                    return True
            # Check for checkbox symbols
            for box_char in checkbox_str:
                if ord(box_char) == 9745:
                    return True
             
        # All other values are considered False
        return False
    except AttributeError:
        # If it's not a string and not a boolean, return False
        return False

def clean_depth(depth_str):
    """
    Clean depth field values, converting 'surface' variations to 0 and processing numeric values.
    
    Many forms contain the word "surface" or variants (Surf., Surf, surf, surf., etc.) 
    to represent situations where the well is cemented or plugged back to surface.
    This function converts these variations to 0 for database storage.
    
    Args:
        depth_str: String or numeric value representing a depth
        
    Returns:
        float: Depth value as a float, with 'surface' variants converted to 0
        None: If the input cannot be parsed
    """
    # If input is already a numeric type, return it as float
    if isinstance(depth_str, (int, float)):
        return float(depth_str)
    
    # If input is None or empty, return None
    if depth_str is None or (isinstance(depth_str, str) and not depth_str.strip()):
        return None
    
    # Convert to string and normalize
    depth_str = str(depth_str).strip()
    
    # Check for 'surface' variations (case-insensitive)
    # Match: surface, surf, surf., Surf, Surf., SURFACE, etc.
    surface_pattern = r'^surf(ace)?\.?$'
    if re.match(surface_pattern, depth_str, re.IGNORECASE):
        return 0.0
    
    # Try to parse as a regular float (handles numbers with trailing dashes, etc.)
    result = string_to_float(depth_str)
    return result

def clean_size_string(size_str):
    """Clean the size string by removing unwanted characters and normalizing format."""
    if not size_str or pd.isna(size_str):
        return None
        
    # Convert to string and clean
    size_str = str(size_str)
    
    # Remove common suffixes and extra characters
    size_str = size_str.replace('"', '').replace('″', '').replace('OD', '').strip()
    
    # If multiple sizes are given (comma-separated), take the first one
    if ',' in size_str:
        size_str = size_str.split(',')[0].strip()
        
    # Remove any trailing numbers after space (e.g., "8 3/4 4265" -> "8 3/4")
    parts = size_str.split()
    if len(parts) > 2 and any(c.isdigit() for c in parts[-1]):
        size_str = ' '.join(parts[:-1])
        
    return size_str

def normalize_special_hole_size(size_str):
    """Normalize special hole size patterns."""
    if not size_str:
        return size_str
        
    # Clean the string first
    size_str = clean_size_string(size_str)
    if not size_str:
        return None
    
    # Handle special cases like '85/', '95/', '133/'
    if size_str.endswith('/'):
        base = size_str[:-1]  # Remove the trailing slash
        if len(base) == 2:  # e.g., '85'
            return f"{base[0]} {base[1]}/8"
        elif len(base) == 3:  # e.g., '133'
            return f"{base[0]}{base[1]} {base[2]}/8"
    
    # Replace hyphens with spaces
    size_str = size_str.replace('-', ' ')
    
    # If the string ends with a partial fraction (e.g., "8 3/"), add "8"
    if size_str.endswith('/'):
        size_str += '8'
    elif ' ' in size_str and size_str.split()[-1].endswith('/'):
        size_str = size_str + '8'
    
    return size_str

def convert_hole_size_to_decimal(size_str):
    """Convert hole size string to decimal value."""
    if not size_str or pd.isna(size_str):
        return None
    
    # Handle special text values
    if isinstance(size_str, str):
        size_str = size_str.strip().lower()
        if size_str in ['none', 'n/a', 'na', 'null', '-', '', 'unknown']:
            print(f"DEBUG: Special text value found: {size_str}")
            return None
    
    # First normalize the string
    size_str = normalize_special_hole_size(str(size_str))
    print(f"DEBUG: After special normalization - size_str='{size_str}', type={type(size_str)}")
    
    if not size_str:
        return None
    
    # Replace any Unicode fractions with standard ASCII
    size_str = size_str.replace('½', '1/2').replace('¼', '1/4').replace('¾', '3/4')
    print(f"DEBUG: After Unicode replacement - size_str='{size_str}', type={type(size_str)}")
    
    # Remove any remaining whitespace and quotes
    size_str = size_str.strip().strip('"\'')
    
    # Check for common hole sizes first (without spaces)
    common_hole_sizes = {
        '85/8': 8.625,
        '95/8': 9.625,
        '133/8': 13.375
    }
    
    # Remove spaces for comparison
    no_space_str = size_str.replace(' ', '')
    if no_space_str in common_hole_sizes:
        print(f"DEBUG: Found match in common_hole_sizes: {no_space_str}")
        return common_hole_sizes[no_space_str]
    
    # If input is already a float, return it
    if isinstance(size_str, float):
        return size_str
    try:
        return float(size_str)
    except ValueError:
        pass
        
    # Handle fractions
    try:
        # Split on space to separate whole number from fraction
        parts = size_str.split()
        
        if len(parts) == 1:
            # Just a fraction like "1/2"
            if '/' in parts[0]:
                num, denom = map(int, parts[0].split('/'))
                return num / denom
            return float(parts[0])
            
        elif len(parts) == 2:
            # Whole number and fraction like "8 3/4"
            whole = float(parts[0])
            if '/' in parts[1]:
                num, denom = map(int, parts[1].split('/'))
                return whole + (num / denom)
            return whole + float(parts[1])
            
        else:
            print(f"DEBUG: Invalid format - too many parts: {parts}")
            return None
            
    except (ValueError, ZeroDivisionError) as e:
        print(f"DEBUG: Error parsing size: {e}")
        return None


if __name__ == '__main__':

    # LLM hole size cleaning
    input = '12-1/4'
    pred = llm_clean(input)
    print('Input: {}'.format(input))
    print('Cleaned hole size: {}\n'.format(pred))

    # Date cleaning
    input = '6/25/1971'
    date = clean_date(input)
    print('Input: {}'.format(input))
    print('Cleaned date: {}\n'.format(date))

    input = '25/10/1971'
    date = clean_date(input)
    print('Input: {}'.format(input))
    print('Cleaned date: {}\n'.format(date))

    input = '2020/8/1'
    date = clean_date(input)
    print('Input: {}'.format(input))
    print('Cleaned date: {}\n'.format(date))

    input = 'April 28,1958'
    date = clean_date(input)
    print('Input: {}'.format(input))
    print('Cleaned date: {}\n'.format(date))
    
    # Test the new m-dd-yy format
    input = '5-27-66'
    date = clean_date(input)
    print('Input: {}'.format(input))
    print('Cleaned date (m-dd-yy format): {}\n'.format(date))
    
    # Test more two-digit year formats
    input = '11/29/54'
    date = clean_date(input)
    print('Input: {}'.format(input))
    print('Cleaned date (two-digit year 50s): {}\n'.format(date))
    
    input = '3-15-22'
    date = clean_date(input)
    print('Input: {}'.format(input))
    print('Cleaned date (two-digit year 20s): {}\n'.format(date))
    
    input = '7/4/99'
    date = clean_date(input)
    print('Input: {}'.format(input))
    print('Cleaned date (two-digit year 90s): {}\n'.format(date))
    
    # Boolean cleaning
    input = ' yes '
    checkbox = clean_bool(input)
    print('Input: {}'.format(input))
    print('Cleaned boolean: {}\n'.format(checkbox))
    
    input = 'true'
    checkbox = clean_bool(input)
    print('Input: {}'.format(input))
    print('Cleaned boolean: {}\n'.format(checkbox))
    
    input = 't'
    checkbox = clean_bool(input)
    print('Input: {}'.format(input))
    print('Cleaned boolean: {}\n'.format(checkbox))
    
    input = 'y'
    checkbox = clean_bool(input)
    print('Input: {}'.format(input))
    print('Cleaned boolean: {}\n'.format(checkbox))
    
    input = '1'
    checkbox = clean_bool(input)
    print('Input: {}'.format(input))
    print('Cleaned boolean: {}\n'.format(checkbox))
    
    input = 'no'
    checkbox = clean_bool(input)
    print('Input: {}'.format(input))
    print('Cleaned boolean: {}\n'.format(checkbox))
    
    input = 'test'
    checkbox = clean_bool(input)
    print('Input: {}'.format(input))
    print('Cleaned boolean: {}\n'.format(checkbox))
    
    input = None
    checkbox = clean_bool(input)
    print('Input: {}'.format(input))
    print('Cleaned boolean: {}\n'.format(checkbox))
    
    input = ''
    checkbox = clean_bool(input)
    print('Input: empty string')
    print('Cleaned boolean: {}\n'.format(checkbox))
    
    # Hole size cleaning
    input = "8 3/4"
    hole_size = convert_hole_size_to_decimal(input)
    print('Input: {}'.format(input))
    print('Hole size: {}\n'.format(hole_size))

    input = "7-7/8"
    hole_size = convert_hole_size_to_decimal(input)
    print('Input: {}'.format(input))
    print('Hole size: {}\n'.format(hole_size))

    input = "13 3/8"
    hole_size = convert_hole_size_to_decimal(input)
    print('Input: {}'.format(input))
    print('Hole size: {}\n'.format(hole_size))

    input = None
    hole_size = convert_hole_size_to_decimal(input)
    print('Input: None')
    print('Hole size: {}\n'.format(hole_size))
    
    # Test with empty string
    input = ""
    hole_size = convert_hole_size_to_decimal(input)
    print('Input: empty string')
    print('Hole size: {}\n'.format(hole_size))
    
    # Test cases for problem Unicode characters
    print("Testing Unicode character handling:")
    input = '8-3/4\u2033'  # With inch symbol
    hole_size = convert_hole_size_to_decimal(input)
    print('Input: {}'.format(input))
    print('Hole size: {}\n'.format(hole_size))
    
    input = '5\u00bd'  # With ½ symbol
    hole_size = convert_hole_size_to_decimal(input)
    print('Input: {}'.format(input))
    print('Hole size: {}\n'.format(hole_size))
    
    # Test cases for special formats
    print("Testing special format handling:")
    input = '85/8'  # Missing space
    hole_size = convert_hole_size_to_decimal(input)
    print('Input: {}'.format(input))
    print('Hole size: {}\n'.format(hole_size))
    
    input = '95/8'  # Missing space
    hole_size = convert_hole_size_to_decimal(input)
    print('Input: {}'.format(input))
    print('Hole size: {}\n'.format(hole_size))
    
    input = '133/8'  # Missing space
    hole_size = convert_hole_size_to_decimal(input)
    print('Input: {}'.format(input))
    print('Hole size: {}\n'.format(hole_size))
    
    input = '8 3/4" OD'  # With OD text
    hole_size = convert_hole_size_to_decimal(input)
    print('Input: {}'.format(input))
    print('Hole size: {}\n'.format(hole_size))
    
    input = 'None'  # Text value
    hole_size = convert_hole_size_to_decimal(input)
    print('Input: {}'.format(input))
    print('Hole size: {}\n'.format(hole_size))
    
    input = 'N/A'  # Text value
    hole_size = convert_hole_size_to_decimal(input)
    print('Input: {}'.format(input))
    print('Hole size: {}\n'.format(hole_size))
    
    # Test cases for compound values
    print("Testing compound values:")
    input = '17 1/2, 12 1/4, 7-7/8'  # Multiple sizes
    hole_size = convert_hole_size_to_decimal(input)
    print('Input: {}'.format(input))
    print('Hole size: {}\n'.format(hole_size))
    
    input = '8 3/4 4265'  # With depth
    hole_size = convert_hole_size_to_decimal(input)
    print('Input: {}'.format(input))
    print('Hole size: {}\n'.format(hole_size))
    
    # Test cases for handling inputs that are already in the target data type
    print("Testing functions with inputs already in target data type:\n")
    
    # Test string_to_float with float input
    float_val = 123.45
    result = string_to_float(float_val)
    print(f"Input: {float_val} (float)")
    print(f"string_to_float result: {result}, same object: {result is float_val}\n")
    
    # Test string_to_int with int input
    int_val = 42
    result = string_to_int(int_val)
    print(f"Input: {int_val} (int)")
    print(f"string_to_int result: {result}, same object: {result is int_val}\n")
    
    # Test clean_date with datetime input
    date_val = datetime.now()
    result = clean_date(date_val)
    print(f"Input: {date_val} (datetime)")
    print(f"clean_date result: {result}, same object: {result is date_val}\n")
    
    # Test clean_bool with boolean input
    bool_val = True
    result = clean_bool(bool_val)
    print(f"Input: {bool_val} (boolean)")
    print(f"clean_bool result: {result}, same object: {result is bool_val}\n")
    
    # Test convert_hole_size_to_decimal with float input
    float_val = 8.75
    result = convert_hole_size_to_decimal(float_val)
    print(f"Input: {float_val} (float)")
    print(f"convert_hole_size_to_decimal result: {result}, same object: {result is float_val}\n")
    
    # Test cases for trailing dash fix
    print("Testing trailing dash handling (fixed issue):")
    
    # Test the original problematic cases
    trailing_dash_cases = ['3/31/61-', '70-', 'NF1424-']
    
    for case in trailing_dash_cases:
        print(f"Input: '{case}'")
        print(f"  string_to_float: {string_to_float(case)}")
        print(f"  string_to_int: {string_to_int(case)}")
        print(f"  clean_date: {clean_date(case)}")
        print()
    
    # Test additional edge cases with trailing dashes
    print("Additional trailing dash test cases:")
    edge_cases = ['123--', '12.34-', '-456-', '1/1/2020-', 'April 28,1958-']
    
    for case in edge_cases:
        print(f"Input: '{case}'")
        print(f"  string_to_float: {string_to_float(case)}")
        print(f"  string_to_int: {string_to_int(case)}")
        print(f"  clean_date: {clean_date(case)}")
        print()
    
    # Test cases for long format date fix
    print("Testing long format date handling (fixed issue):")
    
    # Test the original problematic cases and variations
    long_date_cases = [
        'July 7, 1977',      # Original issue - full month with space after comma
        'July 7 1977',       # Full month without comma
        'March 30, 1963',    # Another full month with space after comma
        'Sept. 11, 1957',    # Non-standard month abbreviation (normalized)
        'Sept 11, 1957',     # Non-standard month abbreviation without period
        'December 25, 2000', # Long month name
        'Jan 15, 2020',      # Standard abbreviated month with space after comma
        'October 31 1999',   # Long month without comma
    ]
    
    for case in long_date_cases:
        result = clean_date(case)
        print(f"Input: '{case}' -> Output: {result}")
    print()
    
    # Test cases for clean_depth function
    print("Testing clean_depth handling (surface variations):")
    
    depth_cases = [
        'surface',       # Full word lowercase
        'Surface',       # Full word capitalized
        'SURFACE',       # Full word uppercase
        'surf',          # Abbreviated lowercase
        'Surf',          # Abbreviated capitalized
        'SURF',          # Abbreviated uppercase
        'surf.',         # Abbreviated with period lowercase
        'Surf.',         # Abbreviated with period capitalized
        'SURF.',         # Abbreviated with period uppercase
        'surface.',      # Full word with period
        '0',             # Already zero
        '1234',          # Regular depth
        '1234.5',        # Depth with decimal
        '1234-',         # Depth with trailing dash
        '  surf  ',      # With whitespace
        None,            # None value
        '',              # Empty string
        'invalid',       # Invalid text
    ]
    
    for case in depth_cases:
        result = clean_depth(case)
        print(f"Input: '{case}' -> Output: {result}")
    print()
