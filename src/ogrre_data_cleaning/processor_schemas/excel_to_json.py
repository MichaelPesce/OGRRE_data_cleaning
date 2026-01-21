# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 09:51:53 2025

@author: shayj
"""

"Requires openpyxl"
import json
import pandas as pd
import numpy as np
import os

def write_json(Dataframe, Filepath, key_replacements=None):
    DF_json = Dataframe.to_json(orient="records")
    json_dict = json.loads(DF_json)
    
    if key_replacements:
        for i, item in enumerate(json_dict):
            for key, value in list(item.items()):
                if key in key_replacements:
                    json_dict[i][key_replacements[key]] = item.pop(key)
    
    with open(Filepath, "w", encoding="utf-8") as json_file:
        json.dump(json_dict, json_file, indent=4, default=str)

def Excel_to_Json(excel_file_path = "ISGS Well Completion Schema.xlsx", Test_Sheet_Names = False):
    Organization = excel_file_path[:excel_file_path.find(' ')].lower()
    # Load the Excel file
    excel_file = pd.ExcelFile(excel_file_path)
    sheet_names = excel_file.sheet_names
    print(sheet_names)
    
    Trained_Models_df = pd.read_excel(excel_file_path, sheet_name="Trained Models")
    Extractors = Trained_Models_df[(Trained_Models_df["Processor Type"] == "Extractor")*(Trained_Models_df["Primary Model in Processor"] == "primary")]

    if Test_Sheet_Names:
        for processor in Extractors["Processor Name"]:
            print(np.any(processor in sheet_names))

    key_replacements = {
        "Google Processor Name": "google_processor_name",
        "OGRRE Document Type": "document_type",
        "Page Order Sort": "page_order_sort",
        "Name": "name",
        "Google Data type": "google_data_type",
        "Occurrence": "occurrence",
        "Grouping": "grouping",
        "Database Data Type": "database_data_type",
        "Cleaning Function": "cleaning_function",
        "Accepted Range": "accepted_range",
        "Field Specific Notes": "field_specific_notes",
        "Model Enabled": "model_enabled",
        "Alias": "alias"
    }

    
    extractors_dir = f"{Organization}_extractors"
    if not os.path.exists(extractors_dir):
        os.makedirs(extractors_dir)
    Extractors = Extractors.astype({'Training Documents': 'float64','Testing Documents': 'float64'})
    write_json(Dataframe = Extractors.reset_index()[Extractors.columns], Filepath = f"{Organization}_extractors/Extractor Processors.json", key_replacements=key_replacements)
    
    for processor in Extractors["Processor Name"]:
        processor_df = pd.read_excel(excel_file_path, sheet_name=processor)
        processor_df = processor_df.astype({'Page Order Sort': 'float64'})
        write_json(Dataframe =  processor_df, Filepath = f"{Organization}_extractors/{processor}.json", key_replacements=key_replacements)
    
    try:
        Obsolete_df = pd.read_excel(excel_file_path, sheet_name="ObsoleteFields")
        write_json(Dataframe =  Obsolete_df, Filepath = f"{Organization}_extractors/01Obsolete Fields.json", key_replacements=key_replacements)
    except:
        pass
        
if __name__ == '__main__':
    Excel_to_Json(excel_file_path = "ISGS Well Completion Schema.xlsx", Test_Sheet_Names = False)
    Excel_to_Json(excel_file_path = "CALGEM Well Summary Schema.xlsx", Test_Sheet_Names = False)
    Excel_to_Json(excel_file_path = "Osage Nation Schema.xlsx", Test_Sheet_Names = False)
    """OSAGE NATION SCHEMA: 
        Following running this script two occurences of 'Osage_V1_Final_Report_Compl_Dep' should be edited to be 
        'Osage_V1_Final_Report_Compl_Deep'. 
        This is caused by excel's sheet name character limit that is not present in Google DocAI or Google Sheets.
        Occurences: 
            Inside Extractor Processors.json as "Processor Name": "Osage_V1_Final_Report_Compl_Deep", and
            File name Osage_V1_Final_Report_Compl_Dep.json
    """
    
