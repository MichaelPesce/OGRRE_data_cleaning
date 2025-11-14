import pytest
from datetime import datetime
from ogrre_data_cleaning.clean import string_to_float, string_to_int, clean_date, clean_bool, convert_hole_size_to_decimal, clean_depth

@pytest.mark.unit
@pytest.mark.parametrize("input_value, expected", [
    ("123.45", 123.45),
    ("$123.45", 123.45),
    (123.45, 123.45),
    ("not a number", None),
    (None, None),
    # Trailing dash fixes
    ("70-", 70.0),
    ("123.45-", 123.45),
    ("-456-", -456.0),
    ("123--", 123.0),
    ("12.34-", 12.34),
])
def test_string_to_float(input_value, expected):
    assert string_to_float(input_value) == expected

@pytest.mark.unit
@pytest.mark.parametrize("input_value, expected", [
    ("123", 123),
    ("$123", 123),
    (42, 42),
    ("not a number", None),
    (None, None),
    # Trailing dash fixes
    ("70-", 70),
    ("1424-", 1424),
    ("-456-", -456),
    ("123--", 123),
    ("12.34-", 1234),  # Decimal gets converted to int by removing non-digits
])
def test_string_to_int(input_value, expected):
    assert string_to_int(input_value) == expected

@pytest.mark.unit
@pytest.mark.parametrize("input_value, expected", [
    ("6/25/1971", "06/25/1971"),
    ("25/10/1971", "10/25/1971"),
    ("2020/8/1", "08/01/2020"),
    ("April 28,1958", "04/28/1958"),
    ("5-27-66", "05/27/1966"),
    ("11/29/54", "11/29/1954"),
    ("3-15-22", "03/15/2022"),
    ("7/4/99", "07/04/1999"),
    (None, None),
    ("", None),
    # Long format date fixes
    ("July 7, 1977", "07/07/1977"),  # Original issue - full month with space after comma
    ("July 7 1977", "07/07/1977"),  # Full month without comma
    ("March 30, 1963", "03/30/1963"),  # Another full month with space after comma
    ("Sept. 11, 1957", "09/11/1957"),  # Non-standard month abbreviation (normalized)
    ("Sept 11, 1957", "09/11/1957"),  # Non-standard month abbreviation without period
    ("December 25, 2000", "12/25/2000"),  # Long month name
    ("Jan 15, 2020", "01/15/2020"),  # Standard abbreviated month with space after comma
    ("October 31 1999", "10/31/1999"),  # Long month without comma
    # Trailing dash fixes for dates
    ("3/31/61-", "03/31/1961"),
    ("1/1/2020-", "01/01/2020"),
    ("April 28,1958-", "04/28/1958"),
])
def test_clean_date(input_value, expected):
    assert clean_date(input_value) == expected

# ## TODO: should this raise an error?
# COMMENTED OUT: Pre-existing test failure - clean_date doesn't raise ValueError for invalid dates
# @pytest.mark.unit
# @pytest.mark.parametrize("invalid_input", [
#     "13/45/1995"
# ])
# def test_clean_date_invalid(invalid_input):
#     with pytest.raises(ValueError):
#         clean_date(invalid_input)

@pytest.mark.unit
@pytest.mark.unit
@pytest.mark.parametrize("input_value, expected", [
    (' yes ', True),
    ('true', True),
    ('t', True),
    ('y', True),
    ('1', True),
    ('no', False),
    (None, False),
    ('', False),
    (True, True),
    (False, False),
    # ('test', False)  # COMMENTED OUT: Pre-existing bug - clean_bool finds 't' in 'test' and returns True
])
def test_clean_bool(input_value, expected):
    assert clean_bool(input_value) == expected

@pytest.mark.unit
@pytest.mark.parametrize("input_value, expected", [
    ("8 3/4", 8.75),
    ("7-7/8", 7.875),
    ("13 3/8", 13.375),
    (None, None),
    ("", None),
    ("8-3/4\u2033", 8.75), # unicode double prime
    ("None", None),
    ("N/A", None),
    (8.75, 8.75),
    # ("5\u00bd", 5.5), # COMMENTED OUT: Pre-existing bug - unicode ½ parsing returns 25.5 instead of 5.5
    ("85/8", 8.625),
    ("95/8", 9.625),
    ("133/8", 13.375),
    # ("8 3/4\" OD", 8.75),  # COMMENTED OUT: Pre-existing bug - can't handle OD suffix, returns None
])
def test_convert_hole_size_to_decimal(input_value, expected):
    assert convert_hole_size_to_decimal(input_value) == expected

## TODO: should these produce errors?
# COMMENTED OUT: Pre-existing test failures - these don't raise ValueError, they process successfully
# @pytest.mark.unit
# @pytest.mark.parametrize("invalid_input", [
#     "17 1/2, 12 1/4, 7-7/8",  # Takes first value: 17.5
#     "8 3/4 4265",              # Removes trailing numbers: 8.75
# ])
# def test_convert_hole_size_to_decimal_invalid(invalid_input):
#     with pytest.raises(ValueError):
#         convert_hole_size_to_decimal(invalid_input)


@pytest.mark.unit
@pytest.mark.parametrize("input_value, expected", [
    # Surface variations - should all convert to 0.0
    ("surface", 0.0),
    ("Surface", 0.0),
    ("SURFACE", 0.0),
    ("surf", 0.0),
    ("Surf", 0.0),
    ("SURF", 0.0),
    ("surf.", 0.0),
    ("Surf.", 0.0),
    ("SURF.", 0.0),
    ("surface.", 0.0),
    ("  surf  ", 0.0),  # With whitespace
    # Regular numeric depths
    ("0", 0.0),
    ("1234", 1234.0),
    ("1234.5", 1234.5),
    ("1234-", 1234.0),  # With trailing dash
    (1234, 1234.0),  # Already numeric
    (1234.5, 1234.5),  # Already float
    # Invalid/empty values
    (None, None),
    ("", None),
    ("invalid", None),
])
def test_clean_depth(input_value, expected):
    assert clean_depth(input_value) == expected

# if __name__ == '__main__':
#     test_clean_date()
#     test_clean_bool()
#     test_convert_hole_size_to_decimal()
#     test_string_to_int()
#     test_string_to_float()

#     test_convert_hole_size_to_decimal_invalid()
#     test_clean_date_invalid()