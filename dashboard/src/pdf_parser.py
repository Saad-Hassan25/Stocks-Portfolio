"""
PDF Parser for Stock Contract Documents
========================================
Parses PDF contract documents (like JSBL contracts) to extract stock transactions.
Extracts: Stock Symbol, Quantity, Price, Transaction Type (Buy/Sell), Date, Fees
"""

import re
from typing import List, Dict, Optional, Tuple
from datetime import datetime

# Try importing PDF libraries
try:
    import pdfplumber
    PDF_LIBRARY = 'pdfplumber'
except ImportError:
    try:
        import PyPDF2
        PDF_LIBRARY = 'PyPDF2'
    except ImportError:
        PDF_LIBRARY = None


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all text from a PDF file.
    Supports both pdfplumber and PyPDF2.
    """
    if PDF_LIBRARY == 'pdfplumber':
        return _extract_with_pdfplumber(pdf_path)
    elif PDF_LIBRARY == 'PyPDF2':
        return _extract_with_pypdf2(pdf_path)
    else:
        raise ImportError("No PDF library available. Install pdfplumber: pip install pdfplumber")


def _extract_with_pdfplumber(pdf_path: str) -> str:
    """Extract text using pdfplumber (better for tables)."""
    import pdfplumber
    
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def _extract_with_pypdf2(pdf_path: str) -> str:
    """Extract text using PyPDF2."""
    import PyPDF2
    
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def extract_tables_from_pdf(pdf_path: str) -> List[List[List[str]]]:
    """
    Extract tables from PDF using pdfplumber.
    Returns list of tables, each table is a list of rows.
    """
    if PDF_LIBRARY != 'pdfplumber':
        return []
    
    import pdfplumber
    
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_tables = page.extract_tables()
            if page_tables:
                tables.extend(page_tables)
    return tables


def parse_jsbl_contract(pdf_path: str) -> Dict:
    """
    Parse JSBL (or similar broker) contract PDF.
    
    Expected format based on typical broker contracts:
    - Contract details (date, contract number)
    - Transaction details (Buy/Sell, Symbol, Quantity, Rate, Amount)
    - Fees and charges
    
    Returns:
    --------
    Dict with:
        'contract_number': str
        'trade_date': str
        'settlement_date': str
        'transactions': List[Dict] with keys:
            - symbol: str
            - quantity: int
            - price: float
            - amount: float
            - type: 'BUY' or 'SELL'
        'fees': Dict with various fee types
        'total_amount': float
        'raw_text': str (for debugging)
    """
    # Extract text
    raw_text = extract_text_from_pdf(pdf_path)
    
    result = {
        'contract_number': None,
        'trade_date': None,
        'settlement_date': None,
        'transactions': [],
        'fees': {},
        'total_amount': 0,
        'raw_text': raw_text
    }
    
    # Try to extract tables first (more reliable)
    tables = extract_tables_from_pdf(pdf_path)
    
    # Parse contract details from text
    result.update(_parse_contract_header(raw_text))
    
    # Parse transactions - try table first, then regex
    if tables:
        result['transactions'] = _parse_transactions_from_tables(tables, raw_text)
    
    if not result['transactions']:
        result['transactions'] = _parse_transactions_from_text(raw_text)
    
    # Parse fees
    result['fees'] = _parse_fees(raw_text)
    
    # Calculate total
    if result['transactions']:
        result['total_amount'] = sum(t.get('amount', 0) for t in result['transactions'])
    
    return result


def _parse_contract_header(text: str) -> Dict:
    """Extract contract header information."""
    header = {}
    
    # Contract Number patterns
    contract_patterns = [
        r'Contract\s*[#:]\s*(\w+)',
        r'Contract\s+No[.:]\s*(\w+)',
        r'CONTRACT[_\s]*(\w+)',
        r'Confirmation\s*[#:]\s*(\w+)',
    ]
    for pattern in contract_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            header['contract_number'] = match.group(1)
            break
    
    # Date patterns
    date_patterns = [
        r'Trade\s*Date[:\s]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
        r'Trade\s*Date[:\s]*(\d{1,2}\s+\w+\s+\d{4})',
        r'Date[:\s]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
        r'(\d{1,2}[-/]\w{3}[-/]\d{4})',  # DD-MMM-YYYY
    ]
    for pattern in date_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            header['trade_date'] = match.group(1)
            break
    
    # Settlement date - JSBL format: "Sett. Date" followed by dates in the table like "06-02-26"
    settlement_patterns = [
        r'Settlement\s*Date[:\s]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
        r'Settlement[:\s]*(\d{1,2}[-/]\w{3}[-/]\d{4})',
        r'Sett\.\s*Date.*?Ready\s+(\d{2}-\d{2}-\d{2})',  # JSBL format
    ]
    for pattern in settlement_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            header['settlement_date'] = match.group(1)
            break
    
    return header


def _parse_transactions_from_tables(tables: List, raw_text: str) -> List[Dict]:
    """
    Parse transactions from extracted tables.
    Supports PDFs with both Purchase and Sale sections.
    """
    transactions = []
    
    # Find positions of Purchase and Sale sections in raw text
    text_lower = raw_text.lower()
    purchase_pos = -1
    sale_pos = -1
    
    # Look for section markers
    purchase_markers = ['purchase order', 'your purchase', 'bought']
    sale_markers = ['sale order', 'your sale', 'sold']
    
    for marker in purchase_markers:
        pos = text_lower.find(marker)
        if pos != -1 and (purchase_pos == -1 or pos < purchase_pos):
            purchase_pos = pos
    
    for marker in sale_markers:
        pos = text_lower.find(marker)
        if pos != -1 and (sale_pos == -1 or pos < sale_pos):
            sale_pos = pos
    
    # Track current position in text for each transaction
    current_text_pos = 0
    
    for table in tables:
        if not table:
            continue
            
        # Find header row
        header_row = None
        header_idx = 0
        
        for idx, row in enumerate(table):
            if row is None:
                continue
            row_text = ' '.join(str(cell or '').lower() for cell in row)
            # Look for typical column headers
            if any(h in row_text for h in ['symbol', 'scrip', 'stock', 'security', 'quantity', 'rate', 'price']):
                header_row = row
                header_idx = idx
                break
        
        if header_row is None:
            continue
        
        # Map columns
        col_map = {}
        for i, cell in enumerate(header_row):
            if cell is None:
                continue
            cell_lower = str(cell).lower().strip()
            if any(k in cell_lower for k in ['symbol', 'scrip', 'security', 'stock']):
                col_map['symbol'] = i
            elif 'qty' in cell_lower or 'quantity' in cell_lower or 'shares' in cell_lower:
                col_map['quantity'] = i
            elif 'rate' in cell_lower or 'price' in cell_lower:
                col_map['price'] = i
            elif 'amount' in cell_lower or 'value' in cell_lower or 'total' in cell_lower:
                col_map['amount'] = i
        
        # Parse data rows
        for row in table[header_idx + 1:]:
            if row is None or all(cell is None or str(cell).strip() == '' for cell in row):
                continue
            
            try:
                # Get symbol first to determine position in text
                symbol = ''
                if 'symbol' in col_map:
                    symbol = str(row[col_map['symbol']] or '').strip()
                    symbol = re.sub(r'[-\s]*(XD|XR|XB).*$', '', symbol, flags=re.IGNORECASE)
                    symbol = symbol.upper()
                
                # Find position of this symbol in raw text to determine BUY/SELL
                trans_type = 'BUY'  # Default
                if symbol:
                    symbol_pos = raw_text.upper().find(symbol, current_text_pos)
                    if symbol_pos != -1:
                        current_text_pos = symbol_pos  # Update for next search
                        
                        # Determine type based on position relative to section headers
                        if purchase_pos != -1 and sale_pos != -1:
                            # Both sections exist
                            if sale_pos > purchase_pos:
                                # Purchase comes first, then Sale
                                trans_type = 'SELL' if symbol_pos > sale_pos else 'BUY'
                            else:
                                # Sale comes first, then Purchase
                                trans_type = 'BUY' if symbol_pos > purchase_pos else 'SELL'
                        elif sale_pos != -1:
                            trans_type = 'SELL'
                        elif purchase_pos != -1:
                            trans_type = 'BUY'
                
                trans = {'type': trans_type, 'symbol': symbol}
                
                if 'quantity' in col_map:
                    qty_str = str(row[col_map['quantity']] or '0')
                    qty_str = re.sub(r'[^\d.]', '', qty_str)
                    trans['quantity'] = int(float(qty_str)) if qty_str else 0
                
                if 'price' in col_map:
                    price_str = str(row[col_map['price']] or '0')
                    price_str = re.sub(r'[^\d.]', '', price_str)
                    trans['price'] = float(price_str) if price_str else 0
                
                if 'amount' in col_map:
                    amount_str = str(row[col_map['amount']] or '0')
                    amount_str = re.sub(r'[^\d.]', '', amount_str)
                    trans['amount'] = float(amount_str) if amount_str else 0
                elif trans.get('quantity') and trans.get('price'):
                    trans['amount'] = trans['quantity'] * trans['price']
                
                # Validate - must have symbol and quantity
                if trans.get('symbol') and trans.get('quantity', 0) > 0:
                    transactions.append(trans)
                    
            except (ValueError, IndexError) as e:
                continue
    
    return transactions


def _parse_transactions_from_text(text: str) -> List[Dict]:
    """
    Fallback: Parse transactions from raw text using regex patterns.
    Handles various formats commonly found in broker contracts.
    Supports PDFs with both Purchase and Sale sections.
    """
    transactions = []
    
    # Find positions of Purchase and Sale sections
    text_lower = text.lower()
    purchase_pos = -1
    sale_pos = -1
    
    # Look for section markers
    purchase_markers = ['purchase order', 'your purchase', 'bought']
    sale_markers = ['sale order', 'your sale', 'sold']
    
    for marker in purchase_markers:
        pos = text_lower.find(marker)
        if pos != -1 and (purchase_pos == -1 or pos < purchase_pos):
            purchase_pos = pos
    
    for marker in sale_markers:
        pos = text_lower.find(marker)
        if pos != -1 and (sale_pos == -1 or pos < sale_pos):
            sale_pos = pos
    
    # JSBL specific format pattern (extended to capture date and fees):
    # Contract# Ready DD-MM-YY SYMBOL QTY RATE BROK.RATE BROK.AMT NET.RATE SST LEVIES AMOUNT
    # Example: 00115943 Ready 13-01-26 LUCK 2 507.7400 1.0155 2.03 508.7555 0.30 0.12 1,017.94
    # Groups: 1=Date, 2=Symbol, 3=Qty, 4=Rate, 5=BrokAmt, 6=SST, 7=Levies
    jsbl_pattern = r'\d{8}\s+Ready\s+(\d{2}-\d{2}-\d{2})\s+([A-Z]+)\s+(\d+)\s+([\d.]+)\s+[\d.]+\s+([\d.]+)\s+[\d.]+\s+([\d.]+)\s+([\d.]+)'
    
    # Use finditer to get positions along with matches
    for match in re.finditer(jsbl_pattern, text):
        try:
            match_pos = match.start()
            sett_date = match.group(1)  # DD-MM-YY format
            symbol = match.group(2).upper()
            quantity = int(match.group(3))
            price = float(match.group(4))
            brok_amt = float(match.group(5))
            sst = float(match.group(6))
            levies = float(match.group(7))
            total_fees = brok_amt + sst + levies
            
            # Convert date from DD-MM-YY to YYYY-MM-DD
            try:
                date_parts = sett_date.split('-')
                year = int(date_parts[2])
                year = 2000 + year if year < 100 else year
                formatted_date = f"{year}-{date_parts[1]}-{date_parts[0]}"
            except:
                formatted_date = None
            
            # Determine transaction type based on position relative to section headers
            trans_type = 'BUY'  # Default
            if purchase_pos != -1 and sale_pos != -1:
                # Both sections exist - check which section this transaction is in
                if match_pos > sale_pos:
                    trans_type = 'SELL'
                elif match_pos > purchase_pos:
                    trans_type = 'BUY'
            elif sale_pos != -1 and match_pos > sale_pos:
                # Only sale section found
                trans_type = 'SELL'
            elif purchase_pos != -1 and match_pos > purchase_pos:
                # Only purchase section found
                trans_type = 'BUY'
            
            if quantity > 0 and price > 0:
                transactions.append({
                    'symbol': symbol,
                    'quantity': quantity,
                    'price': price,
                    'amount': quantity * price,
                    'type': trans_type,
                    'date': formatted_date,
                    'fees': total_fees,
                    'brokerage': brok_amt,
                    'sst': sst,
                    'levies': levies
                })
        except (ValueError, IndexError):
            continue
    
    # If JSBL pattern didn't work, try generic patterns
    if not transactions:
        # Common PSX stock symbols (3-6 uppercase letters)
        # Pattern: SYMBOL followed by quantity and price
        patterns = [
            # Pattern 1: SYMBOL QTY @ PRICE or SYMBOL QTY PRICE
            r'([A-Z]{2,6})\s+(\d{1,3}(?:,\d{3})*|\d+)\s+(?:@\s*)?(\d+(?:\.\d+)?)',
            # Pattern 2: More flexible with PKR/Rs
            r'([A-Z]{2,6})\s+(?:shares?:?\s*)?(\d{1,3}(?:,\d{3})*|\d+)\s+(?:PKR|Rs\.?|@)?\s*(\d+(?:\.\d+)?)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    symbol = match[0].upper()
                    quantity = int(match[1].replace(',', ''))
                    price = float(match[2])
                    
                    # Skip if looks like a date or other number
                    if quantity > 0 and price > 0 and len(symbol) >= 2:
                        transactions.append({
                            'symbol': symbol,
                            'quantity': quantity,
                            'price': price,
                            'amount': quantity * price,
                            'type': trans_type
                        })
                except (ValueError, IndexError):
                    continue
            
            if transactions:
                break
    
    # Remove duplicates (keep first occurrence)
    seen = set()
    unique_transactions = []
    for t in transactions:
        key = (t['symbol'], t['quantity'], t['price'])
        if key not in seen:
            seen.add(key)
            unique_transactions.append(t)
    
    return unique_transactions


def _parse_fees(text: str) -> Dict:
    """Extract fees and charges from the contract."""
    fees = {}
    
    fee_patterns = [
        (r'Commission[:\s]*(?:PKR|Rs\.?)?\s*([\d,]+(?:\.\d+)?)', 'commission'),
        (r'Brokerage[:\s]*(?:PKR|Rs\.?)?\s*([\d,]+(?:\.\d+)?)', 'brokerage'),
        (r'CDC\s*(?:Charges?)?[:\s]*(?:PKR|Rs\.?)?\s*([\d,]+(?:\.\d+)?)', 'cdc_charges'),
        (r'SECP\s*(?:Fee)?[:\s]*(?:PKR|Rs\.?)?\s*([\d,]+(?:\.\d+)?)', 'secp_fee'),
        (r'CVT[:\s]*(?:PKR|Rs\.?)?\s*([\d,]+(?:\.\d+)?)', 'cvt'),
        (r'WHT[:\s]*(?:PKR|Rs\.?)?\s*([\d,]+(?:\.\d+)?)', 'wht'),
        (r'Sales\s*Tax[:\s]*(?:PKR|Rs\.?)?\s*([\d,]+(?:\.\d+)?)', 'sales_tax'),
        (r'Fed[:\s]*(?:PKR|Rs\.?)?\s*([\d,]+(?:\.\d+)?)', 'fed'),
        (r'Total\s*(?:Charges?|Fees?)[:\s]*(?:PKR|Rs\.?)?\s*([\d,]+(?:\.\d+)?)', 'total_fees'),
    ]
    
    for pattern, fee_name in fee_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                amount = float(match.group(1).replace(',', ''))
                fees[fee_name] = amount
            except ValueError:
                continue
    
    # JSBL-specific: Extract total fees from the grand total section
    # Look for patterns like "Total : X X.XX X.XX X.XX X,XXX.XX"
    # Format: Total : QTY BROK SST LEVIES AMOUNT
    total_brok_pattern = r'Total\s*:\s*\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)'
    total_matches = re.findall(total_brok_pattern, text)
    
    if total_matches:
        total_brokerage = sum(float(m[0]) for m in total_matches)
        total_sst = sum(float(m[1]) for m in total_matches)
        total_levies = sum(float(m[2]) for m in total_matches)
        
        if total_brokerage > 0:
            fees['brokerage'] = total_brokerage
        if total_sst > 0:
            fees['sst'] = total_sst
        if total_levies > 0:
            fees['levies'] = total_levies
        
        fees['total_fees'] = total_brokerage + total_sst + total_levies
    
    return fees


def consolidate_transactions(transactions: List[Dict]) -> List[Dict]:
    """
    Consolidate multiple transactions of the same symbol into weighted average.
    
    For example, if you bought FATIMA 9 @ 190.47 and FATIMA 6 @ 190.48,
    this returns FATIMA 15 @ 190.4740 (weighted average).
    
    Fees are summed, and the earliest date is used.
    """
    if not transactions:
        return transactions
    
    # Group by symbol and type
    grouped = {}
    for t in transactions:
        key = (t['symbol'], t['type'])
        if key not in grouped:
            grouped[key] = {
                'total_qty': 0, 
                'total_amount': 0, 
                'total_fees': 0,
                'dates': [],
                'transactions': []
            }
        grouped[key]['total_qty'] += t['quantity']
        grouped[key]['total_amount'] += t['amount']
        grouped[key]['total_fees'] += t.get('fees', 0)
        if t.get('date'):
            grouped[key]['dates'].append(t['date'])
        grouped[key]['transactions'].append(t)
    
    # Create consolidated transactions
    consolidated = []
    for (symbol, trans_type), data in grouped.items():
        avg_price = data['total_amount'] / data['total_qty'] if data['total_qty'] > 0 else 0
        # Use the earliest date
        earliest_date = min(data['dates']) if data['dates'] else None
        consolidated.append({
            'symbol': symbol,
            'quantity': data['total_qty'],
            'price': round(avg_price, 4),
            'amount': data['total_amount'],
            'type': trans_type,
            'date': earliest_date,
            'fees': data['total_fees'],
            'consolidated_from': len(data['transactions'])
        })
    
    return consolidated


def parse_pdf_file(file_path: str, consolidate: bool = False) -> Dict:
    """
    Main entry point for parsing a PDF file.
    
    Parameters:
    -----------
    file_path : str
        Path to the PDF file
    consolidate : bool
        If True, consolidate multiple transactions of the same symbol
    
    Returns:
    --------
    Dict with parsed data including transactions and metadata
    """
    try:
        result = parse_jsbl_contract(file_path)
        
        # Optionally consolidate transactions
        if consolidate and result.get('transactions'):
            result['transactions_raw'] = result['transactions'].copy()
            result['transactions'] = consolidate_transactions(result['transactions'])
        
        result['success'] = True
        result['error'] = None
        return result
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'transactions': [],
            'raw_text': ''
        }


def parse_pdf_bytes(pdf_bytes, filename: str = "uploaded.pdf", consolidate: bool = False) -> Dict:
    """
    Parse PDF from bytes (for Streamlit file uploader).
    
    Parameters:
    -----------
    pdf_bytes : bytes
        PDF file content as bytes
    filename : str
        Original filename for reference
    consolidate : bool
        If True, consolidate multiple transactions of the same symbol
    
    Returns:
    --------
    Dict with parsed data
    """
    import tempfile
    import os
    
    try:
        # Write to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name
        
        # Parse the temp file
        result = parse_pdf_file(tmp_path, consolidate=consolidate)
        result['filename'] = filename
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        return result
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'transactions': [],
            'filename': filename,
            'raw_text': ''
        }


# ============================================================================
# TESTING
# ============================================================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        print(f"Parsing: {pdf_path}")
        print("=" * 60)
        
        result = parse_pdf_file(pdf_path)
        
        if result['success']:
            print(f"Contract #: {result.get('contract_number', 'N/A')}")
            print(f"Trade Date: {result.get('trade_date', 'N/A')}")
            print(f"Settlement Date: {result.get('settlement_date', 'N/A')}")
            print()
            print("TRANSACTIONS:")
            print("-" * 70)
            for t in result['transactions']:
                date_str = t.get('date', 'N/A')
                fees_str = f"Fees: {t.get('fees', 0):.2f}" if 'fees' in t else ""
                print(f"  {t['type']} {t['symbol']}: {t['quantity']} @ {t['price']:.2f} = {t.get('amount', 0):,.2f}  |  Date: {date_str}  |  {fees_str}")
            print()
            # Show total per-transaction fees
            total_trans_fees = sum(t.get('fees', 0) for t in result['transactions'])
            print(f"TOTAL FEES (per transaction): {total_trans_fees:,.2f}")
            print()
            print("FEES (from PDF header):")
            for fee_name, amount in result.get('fees', {}).items():
                print(f"  {fee_name}: {amount:,.2f}")
            print()
            print(f"Total Amount: {result.get('total_amount', 0):,.2f}")
        else:
            print(f"ERROR: {result['error']}")
        
        print("\n" + "=" * 60)
        print("RAW TEXT (first 2000 chars):")
        print(result.get('raw_text', '')[:2000])
    else:
        print("Usage: python pdf_parser.py <path_to_pdf>")
