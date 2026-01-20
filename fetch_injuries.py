import pandas as pd
import requests
import re
from io import StringIO

URL = "https://www.cbssports.com/nba/injuries/"

print("--- FETCHING INJURY REPORT ---")
print(f"Source: {URL}")

try:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    response = requests.get(URL, headers=headers)
    response.raise_for_status()
    
    dfs = pd.read_html(StringIO(response.text))
    
    if not dfs:
        print("[ERROR] No injury tables found.")
        exit()
        
    all_injuries = pd.concat(dfs, ignore_index=True)
    
    if len(all_injuries.columns) >= 5:
        all_injuries.columns = ['Player', 'Pos', 'Date', 'Injury', 'Status']
    
    # --- THE FIX: ADVANCED NAME CLEANING ---
    def clean_player_name(raw_name):
        raw_name = str(raw_name).strip()
        
        # RULE 1: Handle Suffixes (Jr., Sr., II, III, IV)
        # Matches "Jr.Derrick" or "IIIRobert"
        # We look for the suffix followed immediately by a Capital Letter
        suffix_pattern = r'(?:Jr\.|Sr\.|III|II|IV)(?=[A-Z])'
        match = re.search(suffix_pattern, raw_name)
        if match:
            # We found the split point (e.g., end of "Jr.")
            # We take everything AFTER that split point
            split_index = match.end()
            return raw_name[split_index:].strip()

        # RULE 2: Handle Standard "Smushed" Names (Lower -> Upper)
        # Matches "TatumJayson"
        # We use the previous logic but protect "Van", "De", "Mc"
        # Look for lowercase letter followed by Capital (excluding Mc/De/Van/etc)
        standard_pattern = r'(?<=[a-z])(?!(?:Van|De|Mc|Mac|La|Le|Di|St)[A-Z])(?=[A-Z])'
        match = re.search(standard_pattern, raw_name)
        if match:
            split_index = match.end()
            return raw_name[split_index:].strip()

        # If no weird pattern found, return as is
        return raw_name

    # Apply the cleaning
    all_injuries['Player'] = all_injuries['Player'].apply(clean_player_name)
    
    # Save
    all_injuries.to_csv("injuries.csv", index=False)
    print(f"[OK] Found {len(all_injuries)} injured players.")
    print("[OK] Saved cleaned data to 'injuries.csv'")
    
    # Verification: Print specifically the tricky ones if found
    print("\n--- Verification Check ---")
    tricky_names = ["Jones Jr.", "Lively II", "Williams III", "Tatum", "VanVleet"]
    for name in all_injuries['Player']:
        if any(x in name for x in tricky_names):
            print(f"Cleaned: {name}")

except Exception as e:
    print(f"[ERROR] Could not fetch injuries: {e}")