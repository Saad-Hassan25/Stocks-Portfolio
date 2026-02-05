# A mapping of common KSE-100 and KMI-30 stocks to their sectors.
# This list covers major market cap players.

SECTOR_MAP = {
    # Oil & Gas Exploration
    "OGDC": "Oil & Gas Exploration",
    "PPL": "Oil & Gas Exploration",
    "MARI": "Oil & Gas Exploration",
    "POL": "Oil & Gas Exploration",
    
    # Fertilizer
    "ENGRO": "Fertilizer",
    "EFERT": "Fertilizer",
    "FFC": "Fertilizer",
    "FATIMA": "Fertilizer",
    "FFBL": "Fertilizer",

    # Commercial Banks
    "MEBL": "Commercial Banks",
    "MCB": "Commercial Banks",
    "UBL": "Commercial Banks",
    "HBL": "Commercial Banks",
    "BAHL": "Commercial Banks",
    "BAFL": "Commercial Banks",
    "AKBL": "Commercial Banks",
    "FABL": "Commercial Banks",
    "BOP": "Commercial Banks",

    # Cement
    "LUCK": "Cement",
    "DGKC": "Cement",
    "MLCF": "Cement",
    "CHCC": "Cement",
    "FCCL": "Cement",
    "PIOC": "Cement",
    "KOHC": "Cement",

    # Power Generation
    "HUBC": "Power Generation",
    "KAPCO": "Power Generation",
    "LPL": "Power Generation",
    "NCPL": "Power Generation",
    "NPL": "Power Generation",

    # Technology
    "SYS": "Technology & Communication",
    "TRG": "Technology & Communication",
    "AVN": "Technology & Communication",
    "AIRLINK": "Technology & Communication",
    "PTC": "Technology & Communication",

    # Oil & Gas Marketing
    "PSO": "Oil & Gas Marketing",
    "SNGP": "Oil & Gas Marketing",
    "SSGC": "Oil & Gas Marketing",
    "SHEL": "Oil & Gas Marketing",
    "APL": "Oil & Gas Marketing",

    # Chemical
    "EPCL": "Chemical",
    "LOTCHEM": "Chemical",
    "COLG": "Chemical",
    "ICI": "Chemical",

    # Refinery
    "ATRL": "Refinery",
    "NRL": "Refinery",
    "PRL": "Refinery",
    "CNERGY": "Refinery",

    # Automobile
    "MTL": "Automobile Assembler",
    "INDU": "Automobile Assembler",
    "HCAR": "Automobile Assembler",
    "PSMC": "Automobile Assembler",

    # Food & Personal Care
    "UNITY": "Food & Personal Care",
    "NESTLE": "Food & Personal Care",
    "FRIESLAND": "Food & Personal Care",

    # Textile
    "ILP": "Textile Composite",
    "NML": "Textile Composite",
    "GATM": "Textile Composite",
    "KTML": "Textile Composite",

    # Pharmaceuticals
    "SEARLE": "Pharmaceuticals",
    "AGP": "Pharmaceuticals",
    "HALEON": "Pharmaceuticals",
    "ABOT": "Pharmaceuticals",
}

def get_sector(stock_symbol):
    """Returns the sector for a given stock symbol, or 'Others' if not found."""
    # Try direct match
    if stock_symbol in SECTOR_MAP:
        return SECTOR_MAP[stock_symbol]
    
    # Try splitting potential suffixes (though KSE usually is just ticker)
    base_symbol = stock_symbol.split('.')[0] 
    return SECTOR_MAP.get(base_symbol, "Others")
