import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta

# Existing modules
from scraper import fetch_all_data
from data_loader import save_scraped_data, load_data_from_file, save_common_stocks
from calculations import clean_and_normalize_data, calculate_allocation, rebalance_weights

# New modules
from database import (add_transaction, get_transactions, delete_transaction,
                      add_dividend, get_dividends, delete_dividend, get_total_dividends,
                      get_realized_pnl, get_total_realized_pnl,
                      clear_all_portfolio_data, get_portfolio_stats)
from market_data import get_historical_price, get_current_price, get_stock_chart_data, add_technical_indicators
from portfolio_manager import PortfolioManager
from advanced_indicators import SupportResistanceDetector, PatternRecognizer, add_support_resistance_to_df
from pdf_parser import parse_pdf_bytes
from auth import register_user, login_user, get_user_by_id, change_password
from forecasting import forecast_stock_price, get_forecast_summary, is_model_available, get_available_models

st.set_page_config(page_title="PSX Portfolio Manager", layout="wide")

# ================= AUTHENTICATION =================
def init_session_state():
    """Initialize session state variables for authentication."""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'username' not in st.session_state:
        st.session_state.username = None

def render_login_page():
    """Render the login/signup page."""
    st.title("🔐 PSX Portfolio Manager")
    st.markdown("### Welcome! Please login or create an account.")
    
    tab_login, tab_signup = st.tabs(["🔑 Login", "📝 Sign Up"])
    
    with tab_login:
        st.subheader("Login to Your Account")
        with st.form("login_form"):
            login_user_input = st.text_input("Username or Email", key="login_username")
            login_password = st.text_input("Password", type="password", key="login_password")
            login_submitted = st.form_submit_button("Login", type="primary")
            
            if login_submitted:
                if login_user_input and login_password:
                    result = login_user(login_user_input, login_password)
                    if result['success']:
                        st.session_state.logged_in = True
                        st.session_state.user_id = result['user_id']
                        st.session_state.username = result['username']
                        st.success(f"Welcome back, {result['username']}!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(result['message'])
                else:
                    st.error("Please enter username/email and password.")
    
    with tab_signup:
        st.subheader("Create New Account")
        with st.form("signup_form"):
            signup_username = st.text_input("Username (min 3 characters)", key="signup_username")
            signup_email = st.text_input("Email", key="signup_email")
            signup_password = st.text_input("Password (min 6 characters)", type="password", key="signup_password")
            signup_confirm = st.text_input("Confirm Password", type="password", key="signup_confirm")
            signup_submitted = st.form_submit_button("Create Account", type="primary")
            
            if signup_submitted:
                if not signup_username or not signup_email or not signup_password:
                    st.error("Please fill in all fields.")
                elif signup_password != signup_confirm:
                    st.error("Passwords do not match.")
                elif '@' not in signup_email:
                    st.error("Please enter a valid email address.")
                else:
                    result = register_user(signup_username, signup_email, signup_password)
                    if result['success']:
                        st.success("Account created successfully! Please login.")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(result['message'])

def render_user_sidebar():
    """Render user info and logout in sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"👤 **{st.session_state.username}**")
    if st.sidebar.button("🚪 Logout", key="logout_btn"):
        st.session_state.logged_in = False
        st.session_state.user_id = None
        st.session_state.username = None
        st.rerun()

# Initialize session state
init_session_state()

# Check if user is logged in
if not st.session_state.logged_in:
    render_login_page()
    st.stop()

# Get current user ID for all operations
current_user_id = st.session_state.user_id

# ================= NAVIGATION =================
st.sidebar.title("PSX Manager")
render_user_sidebar()
page = st.sidebar.radio("Navigate", ["Portfolio Tracker", "📈 Stock Explorer", "Investment Planner"])

# ================= PORTFOLIO TRACKER (NEW) =================
def render_portfolio_tracker():
    st.header("My Portfolio Tracker")    
    # --- Metrics Section ---
    user_id = st.session_state.user_id
    pm = PortfolioManager(user_id)
    df_holdings, metrics = pm.get_portfolio_summary()
    
    # Get additional metrics
    total_dividends = get_total_dividends(user_id)
    total_realized = get_total_realized_pnl(user_id)
    
    if not df_holdings.empty:
        # Row 1: Main metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Net Worth", f"PKR {metrics['Current Value']:,.2f}", delta=f"{metrics['Total P&L']:,.2f}")
        col2.metric("Total Investment", f"PKR {metrics['Total Investment']:,.2f}")
        col3.metric("Unrealized P&L", f"PKR {metrics['Total P&L']:,.2f}", 
                    delta=f"{metrics['Total Return %']:.2f}%",
                    delta_color="normal" if metrics['Total P&L'] >= 0 else "inverse")
        col4.metric("Realized P&L", f"PKR {total_realized:,.2f}", 
                    delta_color="normal" if total_realized >= 0 else "inverse")
        
        # Row 2: Dividends
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Total Dividends", f"PKR {total_dividends:,.2f}")
        
        st.divider()

    # --- Add Transaction Form ---
    with st.expander("➕ Add New Transaction", expanded=df_holdings.empty):
        st.subheader("Record a Buy/Sell")
        
        with st.form("transaction_form"):
            c1, c2, c3, c4 = st.columns(4)
            symbol = c1.text_input("Stock Symbol (e.g. OGDC)", help="Enter ticker symbol").upper()
            date = c2.date_input("Date", datetime.today())
            quantity_input = c3.number_input("Quantity", min_value=1.0, step=1.0)
            
            # Note: We handle price 0 = autofetch on submit
            price_input = c4.number_input("Price (Set 0 to Auto-Fetch Close Price)", min_value=0.0, step=0.01)
            
            c_type, c_fee = st.columns(2)
            trans_type = c_type.selectbox("Type", ["BUY", "SELL"])
            fees_input = c_fee.number_input("Commission/Fees", min_value=0.0, step=1.0, value=0.0)
            
            submitted = st.form_submit_button("Add Transaction")
            
            if submitted:
                if not symbol:
                    st.error("Please enter a symbol.")
                else:
                    final_price = price_input
                    
                    # Logic: If price is 0, try to fetch it
                    if final_price == 0:
                        with st.spinner(f"Fetching closing price for {symbol} on {date}..."):
                            fetched = get_historical_price(symbol, date)
                            if fetched:
                                final_price = fetched
                                st.info(f"Auto-fetched price: {final_price}")
                            else:
                                st.warning(f"Could not fetch price for {symbol} on {date}. Please enter manually.")
                                st.stop()
                    
                    add_transaction(symbol, date.strftime('%Y-%m-%d'), quantity_input, final_price, trans_type, fees_input, user_id)
                    st.success(f"Transaction added: {trans_type} {quantity_input} {symbol} @ {final_price}")
                    time.sleep(1) 
                    st.rerun()

    # --- Add Dividend Form ---
    with st.expander("💰 Record Dividend"):
        st.subheader("Record Dividend Payment")
        
        with st.form("dividend_form"):
            d1, d2, d3, d4 = st.columns(4)
            div_symbol = d1.text_input("Stock Symbol", help="Enter ticker symbol", key="div_symbol").upper()
            div_date = d2.date_input("Payment Date", datetime.today(), key="div_date")
            div_amount = d3.number_input("Dividend per Share (PKR)", min_value=0.0, step=0.01, key="div_amount")
            div_shares = d4.number_input("Number of Shares", min_value=1.0, step=1.0, key="div_shares")
            
            div_type = st.selectbox("Dividend Type", ["CASH", "STOCK"], key="div_type")
            
            div_submitted = st.form_submit_button("Record Dividend")
            
            if div_submitted:
                if not div_symbol or div_amount <= 0:
                    st.error("Please enter valid symbol and amount.")
                else:
                    add_dividend(div_symbol, div_date.strftime('%Y-%m-%d'), div_amount, div_shares, div_type, user_id)
                    st.success(f"Dividend recorded: {div_symbol} - PKR {div_amount * div_shares:,.2f}")
                    time.sleep(1)
                    st.rerun()

    # --- PDF Contract Upload (Single File - Original) ---
    with st.expander("📄 Import from PDF Contract (Single)"):
        st.subheader("Upload Broker Contract PDF")
        st.caption("Upload your broker's contract note (JSBL, etc.) to automatically extract and import transactions.")
        
        pdf_col1, pdf_col2 = st.columns([3, 1])
        with pdf_col1:
            uploaded_pdf = st.file_uploader("Choose a PDF file", type=['pdf'], key="pdf_uploader_single")
        with pdf_col2:
            consolidate_single = st.checkbox("Consolidate same symbols", value=True, key="pdf_consolidate_single",
                                            help="Combine multiple trades of the same stock into one (weighted average price)")
        
        if uploaded_pdf is not None:
            # Get bytes from uploaded file
            pdf_bytes = uploaded_pdf.getvalue()
            
            # Parse the PDF
            with st.spinner("Parsing PDF..."):
                pdf_result = parse_pdf_bytes(pdf_bytes, uploaded_pdf.name, consolidate=consolidate_single)
            
            if pdf_result.get('success') and pdf_result.get('transactions'):
                st.success(f"✅ Found {len(pdf_result['transactions'])} transaction(s) in the contract")
                
                # Show contract info
                info_cols = st.columns(3)
                info_cols[0].write(f"**Contract #:** {pdf_result.get('contract_number', 'N/A')}")
                info_cols[1].write(f"**Trade Date:** {pdf_result.get('trade_date', 'N/A')}")
                info_cols[2].write(f"**Settlement Date:** {pdf_result.get('settlement_date', 'N/A')}")
                
                # Create editable dataframe for validation
                st.markdown("#### 📋 Extracted Transactions (Review & Edit)")
                st.caption("Edit any incorrect values before importing. Check the 'Import' column for transactions you want to add.")
                
                # Convert to DataFrame for editing
                trans_df = pd.DataFrame(pdf_result['transactions'])
                trans_df['Import'] = True  # Checkbox column
                
                # Ensure all columns exist
                for col in ['symbol', 'quantity', 'price', 'amount', 'type', 'date', 'fees']:
                    if col not in trans_df.columns:
                        if col == 'date':
                            trans_df[col] = datetime.today().strftime('%Y-%m-%d')
                        elif col in ['symbol', 'type']:
                            trans_df[col] = ''
                        else:
                            trans_df[col] = 0
                
                # Fill any None dates with today's date
                trans_df['date'] = trans_df['date'].fillna(datetime.today().strftime('%Y-%m-%d'))
                trans_df['fees'] = trans_df['fees'].fillna(0)
                
                # Reorder columns
                trans_df = trans_df[['Import', 'type', 'symbol', 'quantity', 'price', 'amount', 'date', 'fees']]
                
                # Use data editor for validation
                edited_df = st.data_editor(
                    trans_df,
                    column_config={
                        "Import": st.column_config.CheckboxColumn("Import", default=True),
                        "type": st.column_config.SelectboxColumn("Type", options=["BUY", "SELL"], required=True),
                        "symbol": st.column_config.TextColumn("Symbol", required=True),
                        "quantity": st.column_config.NumberColumn("Quantity", min_value=1, step=1, required=True),
                        "price": st.column_config.NumberColumn("Price (PKR)", min_value=0.01, format="%.2f", required=True),
                        "amount": st.column_config.NumberColumn("Amount (PKR)", format="%.2f", disabled=True),
                        "date": st.column_config.TextColumn("Trade Date", required=True),
                        "fees": st.column_config.NumberColumn("Fees (PKR)", min_value=0, format="%.2f"),
                    },
                    hide_index=True,
                    use_container_width=True,
                    key="pdf_trans_editor_single"
                )
                
                # Show total fees summary
                total_fees_sum = edited_df[edited_df['Import']]['fees'].sum()
                st.info(f"💰 **Total Fees for Selected Transactions:** PKR {total_fees_sum:,.2f}")
                
                # Show fees if available (summary from PDF header)
                if pdf_result.get('fees'):
                    with st.expander("📊 Fees Summary from PDF"):
                        fee_cols = st.columns(4)
                        fees = pdf_result['fees']
                        fee_items = list(fees.items())
                        for i, (fee_name, amount) in enumerate(fee_items):
                            fee_cols[i % 4].metric(fee_name.replace('_', ' ').title(), f"PKR {amount:,.2f}")
                
                # Import button
                if st.button("✅ Import Selected Transactions", type="primary", key="pdf_import_btn_single"):
                    import_count = 0
                    for _, row in edited_df.iterrows():
                        if row['Import'] and row['symbol'] and row['quantity'] > 0 and row['price'] > 0:
                            # Use per-transaction date and fees from the dataframe
                            trans_date = row['date'] if pd.notna(row['date']) else datetime.today().strftime('%Y-%m-%d')
                            trans_fees = float(row['fees']) if pd.notna(row['fees']) else 0
                            
                            add_transaction(
                                symbol=row['symbol'].upper(),
                                date=trans_date,
                                quantity=int(row['quantity']),
                                price=float(row['price']),
                                type=row['type'],
                                fees=trans_fees,
                                user_id=user_id
                            )
                            import_count += 1
                    
                    if import_count > 0:
                        st.success(f"🎉 Successfully imported {import_count} transaction(s)!")
                        time.sleep(1.5)
                        st.rerun()
                    else:
                        st.warning("No valid transactions selected for import.")
                
                # Debug: Show raw text
                with st.expander("🔍 Debug: Raw PDF Text"):
                    st.text(pdf_result.get('raw_text', 'No text extracted')[:3000])
                    
            elif pdf_result.get('success'):
                st.warning("No transactions found in the PDF. The format may not be recognized.")
                st.markdown("**Expected format:** Broker contract notes with tables containing Symbol, Quantity, Price/Rate columns.")
                
                # Show raw text for debugging
                with st.expander("🔍 Raw PDF Text (for debugging)"):
                    st.text(pdf_result.get('raw_text', 'No text extracted')[:3000])
            else:
                st.error(f"Failed to parse PDF: {pdf_result.get('error', 'Unknown error')}")
                st.info("Make sure you have `pdfplumber` installed: `pip install pdfplumber`")

    # --- PDF Contract Upload (Bulk Import) ---
    with st.expander("📄 Bulk Import from PDF Contracts (Multiple Files)"):
        st.subheader("Upload Multiple Broker Contract PDFs")
        st.caption("Upload up to 15 broker contract notes (JSBL, etc.) to automatically extract and import transactions in bulk.")
        
        pdf_col1, pdf_col2 = st.columns([3, 1])
        with pdf_col1:
            uploaded_pdfs = st.file_uploader(
                "Choose PDF files (max 15)", 
                type=['pdf'], 
                key="pdf_uploader_bulk",
                accept_multiple_files=True
            )
        with pdf_col2:
            consolidate_trans = st.checkbox("Consolidate same symbols", value=True, key="pdf_consolidate",
                                            help="Combine multiple trades of the same stock into one (weighted average price)")
            consolidate_across_pdfs = st.checkbox("Consolidate across all PDFs", value=False, key="pdf_consolidate_all",
                                            help="Combine same symbols from different PDFs (uses weighted average)")
        
        if uploaded_pdfs:
            # Limit to 15 PDFs
            if len(uploaded_pdfs) > 15:
                st.warning(f"⚠️ Maximum 15 PDFs allowed. Only the first 15 will be processed.")
                uploaded_pdfs = uploaded_pdfs[:15]
            
            st.info(f"📁 **{len(uploaded_pdfs)} PDF(s) selected for import**")
            
            # Parse all PDFs
            all_transactions = []
            all_fees_summary = {}
            pdf_summaries = []
            failed_pdfs = []
            
            with st.spinner(f"Parsing {len(uploaded_pdfs)} PDF(s)..."):
                for pdf_file in uploaded_pdfs:
                    try:
                        # Use getvalue() to properly get bytes from UploadedFile
                        pdf_bytes = pdf_file.getvalue()
                        pdf_result = parse_pdf_bytes(pdf_bytes, pdf_file.name, consolidate=consolidate_trans)
                        
                        if pdf_result.get('success') and pdf_result.get('transactions'):
                            # Add source filename to each transaction for tracking
                            for trans in pdf_result['transactions']:
                                trans['source_file'] = pdf_file.name
                                # Ensure all required fields exist
                                if 'date' not in trans or not trans['date']:
                                    trans['date'] = pdf_result.get('trade_date', datetime.today().strftime('%Y-%m-%d'))
                                if 'fees' not in trans:
                                    trans['fees'] = 0
                                if 'type' not in trans:
                                    trans['type'] = 'BUY'
                            
                            all_transactions.extend(pdf_result['transactions'])
                            
                            # Track PDF summary
                            pdf_summaries.append({
                                'filename': pdf_file.name,
                                'contract_number': pdf_result.get('contract_number', 'N/A'),
                                'trade_date': pdf_result.get('trade_date', 'N/A'),
                                'transactions': len(pdf_result['transactions']),
                                'buy_count': sum(1 for t in pdf_result['transactions'] if t.get('type') == 'BUY'),
                                'sell_count': sum(1 for t in pdf_result['transactions'] if t.get('type') == 'SELL'),
                                'total_fees': sum(t.get('fees', 0) for t in pdf_result['transactions'])
                            })
                            
                            # Aggregate fees from PDF headers
                            for fee_name, amount in pdf_result.get('fees', {}).items():
                                all_fees_summary[fee_name] = all_fees_summary.get(fee_name, 0) + amount
                        else:
                            failed_pdfs.append({
                                'filename': pdf_file.name,
                                'error': pdf_result.get('error', 'No transactions found')
                            })
                    except Exception as e:
                        failed_pdfs.append({
                            'filename': pdf_file.name,
                            'error': str(e)
                        })
            
            # Show parsing summary
            if pdf_summaries:
                st.success(f"✅ Successfully parsed {len(pdf_summaries)} PDF(s) with {len(all_transactions)} total transaction(s)")
                
                # Show summary table of parsed PDFs
                with st.expander("📊 PDF Parsing Summary", expanded=True):
                    summary_df = pd.DataFrame(pdf_summaries)
                    summary_df.columns = ['File', 'Contract #', 'Trade Date', 'Transactions', 'Buys', 'Sells', 'Fees']
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)
                    
                    # Totals
                    total_buys = sum(s['buy_count'] for s in pdf_summaries)
                    total_sells = sum(s['sell_count'] for s in pdf_summaries)
                    total_trans_fees = sum(s['total_fees'] for s in pdf_summaries)
                    
                    tc1, tc2, tc3, tc4 = st.columns(4)
                    tc1.metric("Total Transactions", len(all_transactions))
                    tc2.metric("Total Buys", total_buys)
                    tc3.metric("Total Sells", total_sells)
                    tc4.metric("Total Fees", f"PKR {total_trans_fees:,.2f}")
            
            # Show failed PDFs
            if failed_pdfs:
                with st.expander(f"❌ Failed to parse {len(failed_pdfs)} PDF(s)", expanded=False):
                    for fp in failed_pdfs:
                        st.error(f"**{fp['filename']}**: {fp['error']}")
            
            # Process transactions if any were found
            if all_transactions:
                # Consolidate across PDFs if requested
                if consolidate_across_pdfs:
                    all_transactions = _consolidate_transactions_across_pdfs(all_transactions)
                    st.info(f"📦 After consolidation: {len(all_transactions)} unique stock transactions")
                
                # Create editable dataframe for validation
                st.markdown("#### 📋 All Extracted Transactions (Review & Edit)")
                st.caption("Edit any incorrect values before importing. Uncheck 'Import' for transactions you want to skip.")
                
                # Convert to DataFrame for editing
                trans_df = pd.DataFrame(all_transactions)
                trans_df['Import'] = True  # Checkbox column
                
                # Ensure all columns exist
                for col in ['symbol', 'quantity', 'price', 'amount', 'type', 'date', 'fees', 'source_file']:
                    if col not in trans_df.columns:
                        if col == 'date':
                            trans_df[col] = datetime.today().strftime('%Y-%m-%d')
                        elif col in ['symbol', 'type', 'source_file']:
                            trans_df[col] = ''
                        else:
                            trans_df[col] = 0
                
                # Fill any None dates with today's date
                trans_df['date'] = trans_df['date'].fillna(datetime.today().strftime('%Y-%m-%d'))
                trans_df['fees'] = trans_df['fees'].fillna(0)
                
                # Reorder columns - put source_file at the end for reference
                display_cols = ['Import', 'type', 'symbol', 'quantity', 'price', 'amount', 'date', 'fees', 'source_file']
                trans_df = trans_df[[c for c in display_cols if c in trans_df.columns]]
                
                # Use data editor for validation
                edited_df = st.data_editor(
                    trans_df,
                    column_config={
                        "Import": st.column_config.CheckboxColumn("Import", default=True),
                        "type": st.column_config.SelectboxColumn("Type", options=["BUY", "SELL"], required=True),
                        "symbol": st.column_config.TextColumn("Symbol", required=True),
                        "quantity": st.column_config.NumberColumn("Quantity", min_value=1, step=1, required=True),
                        "price": st.column_config.NumberColumn("Price (PKR)", min_value=0.01, format="%.2f", required=True),
                        "amount": st.column_config.NumberColumn("Amount (PKR)", format="%.2f", disabled=True),
                        "date": st.column_config.TextColumn("Trade Date", required=True),
                        "fees": st.column_config.NumberColumn("Fees (PKR)", min_value=0, format="%.2f"),
                        "source_file": st.column_config.TextColumn("Source PDF", disabled=True),
                    },
                    hide_index=True,
                    use_container_width=True,
                    key="pdf_trans_editor_bulk",
                    height=400
                )
                
                # Show summary of selected transactions
                selected_df = edited_df[edited_df['Import']]
                buy_selected = len(selected_df[selected_df['type'] == 'BUY'])
                sell_selected = len(selected_df[selected_df['type'] == 'SELL'])
                total_fees_sum = selected_df['fees'].sum()
                
                st.markdown("---")
                sum_c1, sum_c2, sum_c3, sum_c4 = st.columns(4)
                sum_c1.metric("Selected for Import", f"{len(selected_df)} / {len(edited_df)}")
                sum_c2.metric("Buy Transactions", buy_selected)
                sum_c3.metric("Sell Transactions", sell_selected)
                sum_c4.metric("Total Fees", f"PKR {total_fees_sum:,.2f}")
                
                # Show aggregated fees from PDF headers
                if all_fees_summary:
                    with st.expander("📊 Aggregated Fees Summary from All PDFs"):
                        fee_cols = st.columns(4)
                        fee_items = list(all_fees_summary.items())
                        for i, (fee_name, amount) in enumerate(fee_items):
                            fee_cols[i % 4].metric(fee_name.replace('_', ' ').title(), f"PKR {amount:,.2f}")
                
                # Import button
                col_import, col_clear = st.columns([1, 1])
                with col_import:
                    if st.button("✅ Import All Selected Transactions", type="primary", key="pdf_import_btn_bulk"):
                        import_count = 0
                        buy_count = 0
                        sell_count = 0
                        total_fees_imported = 0
                        
                        for _, row in edited_df.iterrows():
                            if row['Import'] and row['symbol'] and row['quantity'] > 0 and row['price'] > 0:
                                # Use per-transaction date and fees from the dataframe
                                trans_date = row['date'] if pd.notna(row['date']) else datetime.today().strftime('%Y-%m-%d')
                                trans_fees = float(row['fees']) if pd.notna(row['fees']) else 0
                                trans_type = row['type'] if row['type'] in ['BUY', 'SELL'] else 'BUY'
                                
                                add_transaction(
                                    symbol=row['symbol'].upper().strip(),
                                    date=trans_date,
                                    quantity=int(row['quantity']),
                                    price=float(row['price']),
                                    type=trans_type,
                                    fees=trans_fees,
                                    user_id=user_id
                                )
                                import_count += 1
                                total_fees_imported += trans_fees
                                if trans_type == 'BUY':
                                    buy_count += 1
                                else:
                                    sell_count += 1
                        
                        if import_count > 0:
                            st.success(f"""
                            🎉 **Successfully imported {import_count} transaction(s)!**
                            - 📈 **{buy_count}** Buy transactions
                            - 📉 **{sell_count}** Sell transactions  
                            - 💰 **PKR {total_fees_imported:,.2f}** in fees recorded
                            """)
                            time.sleep(2)
                            st.rerun()
                        else:
                            st.warning("No valid transactions selected for import.")
                
                with col_clear:
                    if st.button("🗑️ Clear Selection", key="pdf_clear_btn"):
                        st.rerun()
                
            elif not failed_pdfs:
                st.warning("No transactions found in any of the uploaded PDFs.")

    # --- Clear Portfolio (Danger Zone) ---
    with st.expander("⚠️ Danger Zone", expanded=False):
        st.markdown("### 🗑️ Clear Entire Portfolio")
        st.warning("**WARNING:** This will permanently delete ALL your portfolio data including transactions, dividends, and realized P&L records. This action cannot be undone!")
        
        # Show current stats
        stats = get_portfolio_stats(user_id)
        if stats['total'] > 0:
            st.markdown(f"""
            **Data that will be deleted:**
            - 📊 **{stats['transactions']}** transaction(s)
            - 💰 **{stats['dividends']}** dividend record(s)
            - 📈 **{stats['realized_pnl']}** realized P&L record(s)
            """)
            
            # Initialize session state for confirmations
            if 'clear_confirm_1' not in st.session_state:
                st.session_state.clear_confirm_1 = False
            if 'clear_confirm_2' not in st.session_state:
                st.session_state.clear_confirm_2 = False
            
            # First confirmation
            st.checkbox("I understand this will delete ALL my portfolio data", key="clear_confirm_1_cb", 
                       value=st.session_state.clear_confirm_1,
                       on_change=lambda: setattr(st.session_state, 'clear_confirm_1', not st.session_state.clear_confirm_1))
            
            if st.session_state.clear_confirm_1:
                st.error("⚠️ **FINAL CONFIRMATION REQUIRED**")
                
                # Second confirmation
                st.checkbox("I am absolutely sure and want to permanently delete everything", 
                           key="clear_confirm_2_cb",
                           value=st.session_state.clear_confirm_2,
                           on_change=lambda: setattr(st.session_state, 'clear_confirm_2', not st.session_state.clear_confirm_2))
                
                if st.session_state.clear_confirm_2:
                    col_btn1, col_btn2 = st.columns(2)
                    with col_btn1:
                        if st.button("🗑️ DELETE EVERYTHING", type="primary", key="clear_portfolio_btn"):
                            clear_all_portfolio_data(user_id)
                            st.session_state.clear_confirm_1 = False
                            st.session_state.clear_confirm_2 = False
                            st.success("✅ Portfolio cleared successfully!")
                            time.sleep(1.5)
                            st.rerun()
                    with col_btn2:
                        if st.button("❌ Cancel", key="clear_cancel_btn"):
                            st.session_state.clear_confirm_1 = False
                            st.session_state.clear_confirm_2 = False
                            st.rerun()
        else:
            st.info("Your portfolio is already empty. Nothing to clear.")

    # --- Dashboard View ---
    if not df_holdings.empty:
        # Tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Holdings & Performance", "Visual Analysis", "Stock Charts", "Dividends", "Realized P&L"])
        
        with tab1:
            # Display Holdings with formatted columns
            st.markdown("### Current Holdings")
            st.caption("Click on a stock in the 'Stock Charts' tab to view its price trend.")
            
            # Format dataframe for display
            display_df = df_holdings.copy()
            
            st.dataframe(
                display_df.style.format({
                    "Avg Cost": "{:,.2f}",
                    "Total Cost": "{:,.2f}", 
                    "Current Price": "{:,.2f}",
                    "Current Value": "{:,.2f}",
                    "Gain/Loss": "{:,.2f}",
                    "Return %": "{:.2f}%",
                    "Weight %": "{:.2f}%"
                }).background_gradient(subset=['Return %'], cmap="RdYlGn", vmin=-20, vmax=20),
                use_container_width=True
            )
            
            st.divider()
            
            # Transaction History with Delete functionality
            st.markdown("### Transaction History")
            transactions = get_transactions(user_id)
            
            if not transactions.empty:
                t_col1, t_col2 = st.columns([3, 1])
                
                with t_col1:
                    st.dataframe(
                        transactions.style.format({"price": "{:,.2f}", "quantity": "{:.0f}", "fees": "{:.0f}"}), 
                        use_container_width=True
                    )
                
                with t_col2:
                    st.write("Manage Transactions")
                    del_id = st.number_input("Transaction ID to Delete", min_value=0, step=1)
                    if st.button("Delete Transaction", type="primary"):
                        if del_id > 0:
                             delete_transaction(del_id, user_id)
                             st.success(f"Deleted transaction {del_id}")
                             time.sleep(1)
                             st.rerun()

        with tab2:
            st.subheader("📊 Portfolio Visual Analysis")
            
            # Row 1: Allocation Charts
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("##### Allocation by Value")
                if 'Current Value' in df_holdings.columns:
                    fig_pie = px.pie(df_holdings, values='Current Value', names='Stock', hole=0.4,
                                     color_discrete_sequence=px.colors.qualitative.Set3)
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            with c2:
                st.markdown("##### Allocation by Investment Cost")
                if 'Total Cost' in df_holdings.columns:
                    fig_cost_pie = px.pie(df_holdings, values='Total Cost', names='Stock', hole=0.4,
                                          color_discrete_sequence=px.colors.qualitative.Pastel)
                    fig_cost_pie.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_cost_pie, use_container_width=True)
            
            st.divider()
            
            # Row 2: P&L Analysis
            c3, c4 = st.columns(2)
            with c3:
                st.markdown("##### Profit/Loss by Stock")
                if 'Gain/Loss' in df_holdings.columns:
                    df_sorted = df_holdings.sort_values('Gain/Loss', ascending=True)
                    fig_bar = px.bar(df_sorted, x='Gain/Loss', y='Stock', orientation='h',
                                     color='Gain/Loss', color_continuous_scale='RdYlGn',
                                     text=df_sorted['Gain/Loss'].apply(lambda x: f'{x:,.0f}'))
                    fig_bar.update_traces(textposition='outside')
                    fig_bar.update_layout(showlegend=False, yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_bar, use_container_width=True)
            
            with c4:
                st.markdown("##### Return % by Stock")
                if 'Return %' in df_holdings.columns:
                    df_sorted = df_holdings.sort_values('Return %', ascending=True)
                    fig_return = px.bar(df_sorted, x='Return %', y='Stock', orientation='h',
                                        color='Return %', color_continuous_scale='RdYlGn',
                                        text=df_sorted['Return %'].apply(lambda x: f'{x:.1f}%'))
                    fig_return.update_traces(textposition='outside')
                    fig_return.update_layout(showlegend=False, yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_return, use_container_width=True)
            
            st.divider()
            
            # Row 3: Investment vs Current Value & Treemap
            c5, c6 = st.columns(2)
            with c5:
                st.markdown("##### Investment vs Current Value")
                comparison_df = df_holdings[['Stock', 'Total Cost', 'Current Value']].melt(
                    id_vars='Stock', var_name='Type', value_name='Amount'
                )
                fig_compare = px.bar(comparison_df, x='Stock', y='Amount', color='Type',
                                     barmode='group', color_discrete_map={'Total Cost': '#636EFA', 'Current Value': '#00CC96'})
                fig_compare.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.02))
                st.plotly_chart(fig_compare, use_container_width=True)
            
            with c6:
                st.markdown("##### Portfolio Treemap")
                if 'Current Value' in df_holdings.columns:
                    fig_tree = px.treemap(df_holdings, path=['Stock'], values='Current Value',
                                          color='Return %', color_continuous_scale='RdYlGn',
                                          color_continuous_midpoint=0)
                    st.plotly_chart(fig_tree, use_container_width=True)
            
            st.divider()
            
            # Row 4: Weight Distribution & Performance Summary
            c7, c8 = st.columns(2)
            with c7:
                st.markdown("##### Portfolio Weight Distribution")
                if 'Weight %' in df_holdings.columns:
                    df_weight_sorted = df_holdings.sort_values('Weight %', ascending=False)
                    fig_weight = px.bar(df_weight_sorted, x='Stock', y='Weight %',
                                        color='Weight %', color_continuous_scale='Blues',
                                        text=df_weight_sorted['Weight %'].apply(lambda x: f'{x:.1f}%'))
                    fig_weight.update_traces(textposition='outside')
                    fig_weight.update_layout(showlegend=False)
                    st.plotly_chart(fig_weight, use_container_width=True)
            
            with c8:
                st.markdown("##### Performance Summary")
                if len(df_holdings) >= 2:
                    top_performer = df_holdings.loc[df_holdings['Return %'].idxmax()]
                    bottom_performer = df_holdings.loc[df_holdings['Return %'].idxmin()]
                    
                    col_top, col_bot = st.columns(2)
                    with col_top:
                        st.metric("🏆 Top Performer", top_performer['Stock'], 
                                  delta=f"{top_performer['Return %']:.2f}%")
                        st.caption(f"P&L: PKR {top_performer['Gain/Loss']:,.0f}")
                    with col_bot:
                        delta_color = "normal" if bottom_performer['Return %'] >= 0 else "inverse"
                        st.metric("📉 Worst Performer", bottom_performer['Stock'], 
                                  delta=f"{bottom_performer['Return %']:.2f}%", delta_color=delta_color)
                        st.caption(f"P&L: PKR {bottom_performer['Gain/Loss']:,.0f}")
                    
                    st.divider()
                    avg_return = df_holdings['Return %'].mean()
                    positive_stocks = len(df_holdings[df_holdings['Gain/Loss'] > 0])
                    negative_stocks = len(df_holdings[df_holdings['Gain/Loss'] < 0])
                    
                    st.write(f"**Average Return:** {avg_return:.2f}%")
                    st.write(f"**Winning Positions:** {positive_stocks} | **Losing Positions:** {negative_stocks}")
        
        # --- TAB 3: STOCK CHARTS WITH TECHNICAL INDICATORS ---
        with tab3:
            st.subheader("Stock Price Charts with Technical Indicators")
            st.caption("Select a stock to view its historical price trend with RSI, MACD, Bollinger Bands, and Advanced Analysis.")
            
            # Help expander with indicator explanations
            with st.expander("📚 What do these indicators mean? (Click to learn)"):
                help_col1, help_col2 = st.columns(2)
                with help_col1:
                    st.markdown("""
                    **📊 RSI (Relative Strength Index)**
                    - Measures momentum on a scale of 0-100
                    - **Above 70** = Overbought (price may fall soon)
                    - **Below 30** = Oversold (price may rise soon)
                    - **30-70** = Neutral zone
                    
                    **📈 Bollinger Bands**
                    - 3 lines around price: Middle (20-day avg), Upper & Lower bands
                    - Bands **widen** during high volatility, **narrow** during calm
                    - Price near **upper band** = potentially overbought
                    - Price near **lower band** = potentially oversold
                    
                    **📉 MACD (Moving Average Convergence Divergence)**
                    - Shows trend direction and momentum
                    - **MACD above Signal** = Bullish momentum 📈
                    - **MACD below Signal** = Bearish momentum 📉
                    """)
                with help_col2:
                    st.markdown("""
                    **🔴🟢 Support/Resistance Zones**
                    - **Support (Green)**: Price level where buying pressure is strong - price tends to bounce UP from here
                    - **Resistance (Red)**: Price level where selling pressure is strong - price tends to fall back from here
                    - More "touches" = stronger level
                    
                    **🎯 Pattern Recognition**
                    - **Double Bottom (W shape)**: Bullish reversal - price may go UP
                    - **Double Top (M shape)**: Bearish reversal - price may go DOWN
                    - **Head & Shoulders**: Bearish reversal pattern
                    - **Inverse H&S**: Bullish reversal pattern
                    - **Higher Highs/Lows**: Uptrend continuation
                    - **Lower Highs/Lows**: Downtrend continuation
                    """)
            
            stock_list = df_holdings['Stock'].tolist()
            
            col_select, col_period = st.columns([2, 1])
            with col_select:
                selected_stock = st.selectbox("Select Stock", stock_list, key="chart_stock_select")
            with col_period:
                period_options = {"1 Month": "1mo", "3 Months": "3mo", "6 Months": "6mo", "1 Year": "1y", "2 Years": "2y", "5 Years": "5y"}
                selected_period_label = st.selectbox("Time Period", list(period_options.keys()), index=2, key="chart_period_select")
                selected_period = period_options[selected_period_label]
            
            # Indicators and Advanced Analysis row
            col_indicators, col_advanced = st.columns(2)
            with col_indicators:
                show_indicators = st.multiselect("Technical Indicators", ["Bollinger Bands", "RSI", "MACD"], default=["RSI"], key="chart_indicators")
            with col_advanced:
                show_advanced = st.multiselect("Advanced Analysis", ["Support/Resistance Zones", "Pattern Recognition"], default=[], key="chart_advanced")
            
            if selected_stock:
                with st.spinner(f"Loading chart for {selected_stock}..."):
                    chart_data = get_stock_chart_data(selected_stock, period=selected_period)
                    if not chart_data.empty:
                        chart_data = add_technical_indicators(chart_data)
                
                if not chart_data.empty:
                    stock_row = df_holdings[df_holdings['Stock'] == selected_stock].iloc[0]
                    
                    # --- ADVANCED ANALYSIS: Support/Resistance & Patterns ---
                    sr_levels = None
                    pattern_result = None
                    
                    if "Support/Resistance Zones" in show_advanced:
                        sr_detector = SupportResistanceDetector(window=5, tolerance_pct=1.5, min_touches=2)
                        sr_levels = sr_detector.get_nearest_levels(chart_data, n_levels=3)
                    
                    if "Pattern Recognition" in show_advanced:
                        pattern_detector = PatternRecognizer(window=5, tolerance_pct=3.0)
                        pattern_result = pattern_detector.detect_pattern(chart_data)
                    
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Current Price", f"PKR {stock_row['Current Price']:,.2f}")
                    m2.metric("Avg Cost", f"PKR {stock_row['Avg Cost']:,.2f}")
                    m3.metric("Quantity", f"{stock_row['Quantity']:.0f}")
                    m4.metric("Return", f"{stock_row['Return %']:.2f}%", delta=f"{stock_row['Gain/Loss']:,.0f}")
                    
                    num_indicator_rows = sum([1 for i in ["RSI", "MACD"] if i in show_indicators])
                    total_rows = 2 + num_indicator_rows
                    row_heights = [0.5] + [0.15] * (total_rows - 1)
                    
                    fig = make_subplots(
                        rows=total_rows, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.03,
                        row_heights=row_heights,
                        subplot_titles=[f"{selected_stock} Price"] + [""] * (total_rows - 1)
                    )
                    
                    fig.add_trace(go.Candlestick(
                        x=chart_data['Date'],
                        open=chart_data['Open'],
                        high=chart_data['High'],
                        low=chart_data['Low'],
                        close=chart_data['Close'],
                        name='Price'
                    ), row=1, col=1)
                    
                    fig.add_trace(go.Scatter(
                        x=chart_data['Date'], y=chart_data['MA20'],
                        mode='lines', name='20-Day MA',
                        line=dict(color='orange', width=1)
                    ), row=1, col=1)
                    
                    fig.add_trace(go.Scatter(
                        x=chart_data['Date'], y=chart_data['MA50'],
                        mode='lines', name='50-Day MA',
                        line=dict(color='blue', width=1)
                    ), row=1, col=1)
                    
                    if "Bollinger Bands" in show_indicators:
                        fig.add_trace(go.Scatter(
                            x=chart_data['Date'], y=chart_data['BB_Upper'],
                            mode='lines', name='BB Upper',
                            line=dict(color='rgba(128,128,128,0.5)', width=1, dash='dash')
                        ), row=1, col=1)
                        
                        fig.add_trace(go.Scatter(
                            x=chart_data['Date'], y=chart_data['BB_Lower'],
                            mode='lines', name='BB Lower',
                            line=dict(color='rgba(128,128,128,0.5)', width=1, dash='dash'),
                            fill='tonexty', fillcolor='rgba(128,128,128,0.1)'
                        ), row=1, col=1)
                    
                    # --- ADD SUPPORT/RESISTANCE ZONES TO CHART ---
                    if sr_levels:
                        # Add support lines (green, dashed)
                        for i, sup in enumerate(sr_levels.get('support_levels', [])):
                            opacity = 0.8 - (i * 0.2)
                            fig.add_hline(
                                y=sup['level'], 
                                line_dash="dot", 
                                line_color=f"rgba(0, 200, 0, {opacity})",
                                line_width=2,
                                annotation_text=f"S{i+1}: {sup['level']:.2f}",
                                annotation_position="left",
                                row=1, col=1
                            )
                        
                        # Add resistance lines (red, dashed)
                        for i, res in enumerate(sr_levels.get('resistance_levels', [])):
                            opacity = 0.8 - (i * 0.2)
                            fig.add_hline(
                                y=res['level'], 
                                line_dash="dot", 
                                line_color=f"rgba(255, 0, 0, {opacity})",
                                line_width=2,
                                annotation_text=f"R{i+1}: {res['level']:.2f}",
                                annotation_position="right",
                                row=1, col=1
                            )
                    
                    colors = ['green' if chart_data['Close'].iloc[i] >= chart_data['Open'].iloc[i] else 'red' 
                              for i in range(len(chart_data))]
                    fig.add_trace(go.Bar(
                        x=chart_data['Date'], y=chart_data['Volume'],
                        marker_color=colors, name='Volume', showlegend=False
                    ), row=2, col=1)
                    
                    current_row = 3
                    
                    if "RSI" in show_indicators:
                        fig.add_trace(go.Scatter(
                            x=chart_data['Date'], y=chart_data['RSI'],
                            mode='lines', name='RSI',
                            line=dict(color='purple', width=1)
                        ), row=current_row, col=1)
                        fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1)
                        fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_row, col=1)
                        fig.update_yaxes(title_text="RSI", row=current_row, col=1)
                        current_row += 1
                    
                    if "MACD" in show_indicators:
                        fig.add_trace(go.Scatter(
                            x=chart_data['Date'], y=chart_data['MACD'],
                            mode='lines', name='MACD',
                            line=dict(color='blue', width=1)
                        ), row=current_row, col=1)
                        
                        fig.add_trace(go.Scatter(
                            x=chart_data['Date'], y=chart_data['MACD_Signal'],
                            mode='lines', name='Signal',
                            line=dict(color='orange', width=1)
                        ), row=current_row, col=1)
                        
                        colors_macd = ['green' if val >= 0 else 'red' for val in chart_data['MACD_Hist']]
                        fig.add_trace(go.Bar(
                            x=chart_data['Date'], y=chart_data['MACD_Hist'],
                            marker_color=colors_macd, name='MACD Hist', showlegend=False
                        ), row=current_row, col=1)
                        fig.update_yaxes(title_text="MACD", row=current_row, col=1)
                    
                    fig.update_layout(
                        height=600 + (num_indicator_rows * 150),
                        template="plotly_white",
                        xaxis_rangeslider_visible=False,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        showlegend=True
                    )
                    fig.update_yaxes(title_text="Price (PKR)", row=1, col=1)
                    fig.update_yaxes(title_text="Volume", row=2, col=1)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # --- ANALYSIS PANELS ---
                    analysis_cols = st.columns(2)
                    
                    # Left Column: Technical Indicator Interpretations
                    with analysis_cols[0]:
                        st.markdown("##### 📊 Technical Signals")
                        if "RSI" in show_indicators and 'RSI' in chart_data.columns and not chart_data['RSI'].isna().all():
                            latest_rsi = chart_data['RSI'].iloc[-1]
                            if latest_rsi > 70:
                                st.warning(f"⚠️ **RSI: {latest_rsi:.1f}** - Overbought")
                                st.caption("Price has risen rapidly. May pull back soon. Consider taking profits or waiting before buying.")
                            elif latest_rsi < 30:
                                st.success(f"✅ **RSI: {latest_rsi:.1f}** - Oversold")
                                st.caption("Price has fallen significantly. May bounce back. Could be a buying opportunity.")
                            else:
                                st.info(f"ℹ️ **RSI: {latest_rsi:.1f}** - Neutral Zone")
                                st.caption("No extreme momentum. Price moving normally.")
                        
                        if "MACD" in show_indicators and 'MACD' in chart_data.columns:
                            latest_macd = chart_data['MACD'].iloc[-1]
                            latest_signal = chart_data['MACD_Signal'].iloc[-1]
                            if latest_macd > latest_signal:
                                st.success(f"📈 **MACD: Bullish Crossover**")
                                st.caption(f"MACD ({latest_macd:.2f}) is above Signal ({latest_signal:.2f}). Upward momentum building.")
                            else:
                                st.error(f"📉 **MACD: Bearish Crossover**")
                                st.caption(f"MACD ({latest_macd:.2f}) is below Signal ({latest_signal:.2f}). Downward momentum building.")
                        
                        if "Bollinger Bands" in show_indicators and 'BB_Upper' in chart_data.columns:
                            latest_close = chart_data['Close'].iloc[-1]
                            bb_upper = chart_data['BB_Upper'].iloc[-1]
                            bb_lower = chart_data['BB_Lower'].iloc[-1]
                            bb_middle = chart_data['MA20'].iloc[-1]
                            
                            if latest_close >= bb_upper * 0.98:
                                st.warning(f"📈 **Bollinger: Near Upper Band**")
                                st.caption("Price near upper band. May be stretched. Watch for pullback.")
                            elif latest_close <= bb_lower * 1.02:
                                st.success(f"📉 **Bollinger: Near Lower Band**")
                                st.caption("Price near lower band. May be oversold. Watch for bounce.")
                            else:
                                st.info(f"ℹ️ **Bollinger: Mid-Range**")
                                st.caption("Price trading within normal range.")
                    
                    # Right Column: Advanced Analysis
                    with analysis_cols[1]:
                        st.markdown("##### 🎯 Advanced Analysis")
                        # Pattern Recognition Display with explanation
                        if pattern_result and pattern_result.get('pattern'):
                            signal = pattern_result.get('signal', '')
                            pattern_name = pattern_result.get('pattern', '')
                            
                            # Pattern explanations
                            pattern_explanations = {
                                'Bullish Double Bottom': "📈 W-shaped pattern. Price fell twice to similar lows, then bounced. Suggests buyers are stepping in - price may go UP.",
                                'Bearish Double Top': "📉 M-shaped pattern. Price rose twice to similar highs, then fell back. Suggests sellers are stepping in - price may go DOWN.",
                                'Bullish Higher Highs and Higher Lows': "📈 Uptrend in progress. Each high is higher than the previous, showing strong buying pressure.",
                                'Bearish Lower Highs and Lower Lows': "📉 Downtrend in progress. Each high is lower than the previous, showing strong selling pressure.",
                                'Bullish Inverse Head and Shoulders': "📈 Three lows with middle being deepest. Strong reversal signal - price likely to go UP significantly.",
                                'Bearish Head and Shoulders': "📉 Three highs with middle being highest. Strong reversal signal - price likely to go DOWN significantly."
                            }
                            
                            if signal == 'bullish':
                                st.success(f"🎯 **{pattern_name}**")
                            else:
                                st.warning(f"🎯 **{pattern_name}**")
                            
                            explanation = pattern_explanations.get(pattern_name, "Chart pattern detected that may signal a trend change.")
                            st.caption(explanation)
                        elif "Pattern Recognition" in show_advanced:
                            st.info("ℹ️ No clear chart pattern detected")
                            st.caption("No recognizable reversal or continuation pattern at current price levels.")
                        
                        # Support/Resistance Summary
                        if sr_levels:
                            st.markdown("##### 📊 Key Price Levels")
                            sr_col1, sr_col2 = st.columns(2)
                            with sr_col1:
                                st.markdown("**🟢 Support (Buy Zones):**")
                                if sr_levels.get('support_levels'):
                                    for i, s in enumerate(sr_levels['support_levels']):
                                        dist_pct = ((sr_levels['current_price'] - s['level']) / sr_levels['current_price']) * 100
                                        st.write(f"S{i+1}: PKR {s['level']:,.2f}")
                                        st.caption(f"{dist_pct:.1f}% below current price")
                                else:
                                    st.write("No clear support levels")
                            with sr_col2:
                                st.markdown("**🔴 Resistance (Sell Zones):**")
                                if sr_levels.get('resistance_levels'):
                                    for i, r in enumerate(sr_levels['resistance_levels']):
                                        dist_pct = ((r['level'] - sr_levels['current_price']) / sr_levels['current_price']) * 100
                                        st.write(f"R{i+1}: PKR {r['level']:,.2f}")
                                        st.caption(f"{dist_pct:.1f}% above current price")
                                else:
                                    st.write("No clear resistance levels")
                            
                            st.caption("💡 **Tip**: Support levels are good entry points. Resistance levels are profit-taking zones.")
                    
                    # --- FORECASTING SECTION ---
                    st.divider()
                    st.markdown("### 🔮 AI Price Forecast")
                    
                    if not is_model_available():
                        st.warning("⚠️ No forecasting models installed.")
                        st.code('pip install "chronos-forecasting>=2.0"  # For Chronos-2\npip install toto-ts  # For Toto', language="bash")
                        st.caption("Install at least one package and restart the app to enable AI-powered price forecasting.")
                    else:
                        with st.expander("📈 Generate Price Forecast (Click to expand)", expanded=False):
                            # Get available models
                            available_models = get_available_models()
                            model_names = [m[0] for m in available_models]
                            model_descriptions = {m[0]: m[1] for m in available_models}
                            
                            st.caption("Uses AI models to forecast future stock prices based on historical patterns.")
                            
                            # Model selection
                            selected_model = st.selectbox(
                                "Select AI Model",
                                model_names,
                                key=f"model_select_{selected_stock}",
                                help="Choose the forecasting model"
                            )
                            st.caption(f"ℹ️ {model_descriptions.get(selected_model, '')}")
                            
                            # Fetch maximum historical data for forecasting
                            with st.spinner("Loading maximum historical data..."):
                                forecast_chart_data = get_stock_chart_data(selected_stock, period="max")
                            
                            available_days = len(forecast_chart_data) if not forecast_chart_data.empty else 0
                            
                            # Set max context based on model
                            model_max_context = 4096 if selected_model == "Toto" else 8192
                            
                            fc_col1, fc_col2, fc_col3 = st.columns(3)
                            with fc_col1:
                                forecast_days = st.slider("Forecast Period (Days)", min_value=7, max_value=1024, value=30, key=f"forecast_days_{selected_stock}",
                                                         help="Number of days to forecast (max 1024)")
                            with fc_col2:
                                max_context = min(available_days, model_max_context) if available_days > 100 else model_max_context
                                context_days = st.slider("Historical Context (Days)", min_value=100, max_value=max_context, value=min(1024, max_context), key=f"context_days_{selected_stock}",
                                                        help=f"More historical data = better predictions (max {model_max_context})")
                            with fc_col3:
                                st.info(f"📊 Available: {available_days} days")
                            
                            if st.button("🔮 Generate Forecast", key=f"forecast_btn_{selected_stock}", type="primary"):
                                with st.spinner(f"🤖 {selected_model} is analyzing patterns and generating forecast..."):
                                    forecast_result = forecast_stock_price(
                                        forecast_chart_data,
                                        prediction_length=forecast_days,
                                        context_length=context_days,
                                        model_name=selected_model
                                    )
                                
                                if forecast_result.get('success'):
                                    # Get summary
                                    summary = get_forecast_summary(forecast_result)
                                    
                                    # Display summary metrics
                                    if summary:
                                        st.success(f"✅ Forecast generated successfully using {summary.get('model', selected_model)}!")
                                        
                                        sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
                                        sum_col1.metric(
                                            "Current Price", 
                                            f"PKR {summary['current_price']:,.2f}"
                                        )
                                        sum_col2.metric(
                                            f"Forecast ({summary['days_ahead']}d)", 
                                            f"PKR {summary['forecast_end_price']:,.2f}",
                                            delta=f"{summary['price_change_pct']:+.2f}%"
                                        )
                                        sum_col3.metric(
                                            "Forecast Range",
                                            f"{summary['forecast_low']:,.0f} - {summary['forecast_high']:,.0f}"
                                        )
                                        sum_col4.metric(
                                            "Trend Signal",
                                            f"{summary['trend_emoji']} {summary['trend']}"
                                        )
                                    
                                    # Create forecast chart
                                    fig_forecast = go.Figure()
                                    
                                    # Historical prices
                                    fig_forecast.add_trace(go.Scatter(
                                        x=chart_data['Date'],
                                        y=chart_data['Close'],
                                        mode='lines',
                                        name='Historical Price',
                                        line=dict(color='#2196F3', width=2)
                                    ))
                                    
                                    # Forecast dates
                                    forecast_dates = forecast_result['dates']
                                    
                                    # 90% confidence interval (outer band)
                                    if forecast_result.get('lower_90') and forecast_result.get('upper_90'):
                                        fig_forecast.add_trace(go.Scatter(
                                            x=forecast_dates + forecast_dates[::-1],
                                            y=forecast_result['upper_90'] + forecast_result['lower_90'][::-1],
                                            fill='toself',
                                            fillcolor='rgba(255, 165, 0, 0.1)',
                                            line=dict(color='rgba(255,255,255,0)'),
                                            name='90% Confidence',
                                            showlegend=True
                                        ))
                                    
                                    # 50% confidence interval (inner band)
                                    if forecast_result.get('lower_50') and forecast_result.get('upper_50'):
                                        fig_forecast.add_trace(go.Scatter(
                                            x=forecast_dates + forecast_dates[::-1],
                                            y=forecast_result['upper_50'] + forecast_result['lower_50'][::-1],
                                            fill='toself',
                                            fillcolor='rgba(255, 165, 0, 0.2)',
                                            line=dict(color='rgba(255,255,255,0)'),
                                            name='50% Confidence',
                                            showlegend=True
                                        ))
                                    
                                    # Median forecast line
                                    fig_forecast.add_trace(go.Scatter(
                                        x=forecast_dates,
                                        y=forecast_result['median'],
                                        mode='lines',
                                        name='Forecast (Median)',
                                        line=dict(color='#FF9800', width=2, dash='dash')
                                    ))
                                    
                                    # Add vertical line at forecast start (without annotation to avoid Plotly bug)
                                    fig_forecast.add_shape(
                                        type="line",
                                        x0=forecast_result['last_actual_date'],
                                        x1=forecast_result['last_actual_date'],
                                        y0=0,
                                        y1=1,
                                        yref="paper",
                                        line=dict(color="gray", width=2, dash="dash")
                                    )
                                    # Add annotation separately
                                    fig_forecast.add_annotation(
                                        x=forecast_result['last_actual_date'],
                                        y=1.05,
                                        yref="paper",
                                        text="Forecast Start",
                                        showarrow=False,
                                        font=dict(size=10, color="gray")
                                    )
                                    
                                    fig_forecast.update_layout(
                                        title=f"{selected_stock} - {forecast_days}-Day Price Forecast",
                                        xaxis_title="Date",
                                        yaxis_title="Price (PKR)",
                                        template="plotly_white",
                                        height=400,
                                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                        hovermode='x unified'
                                    )
                                    
                                    st.plotly_chart(fig_forecast, use_container_width=True)
                                    
                                    # Disclaimer
                                    st.caption("⚠️ **Disclaimer**: This forecast is generated by an AI model based on historical patterns. "
                                              "It should NOT be used as the sole basis for investment decisions. Past performance does not guarantee future results. "
                                              "Always do your own research and consider consulting a financial advisor.")
                                else:
                                    st.error(f"❌ {forecast_result.get('error', 'Unknown error occurred')}")
                else:
                    st.warning(f"Could not fetch chart data for {selected_stock}. Yahoo Finance may not have data for this stock.")
        
        # --- TAB 4: DIVIDENDS ---
        with tab4:
            st.subheader("💰 Dividend Tracking")
            
            div_summary, total_div = pm.get_dividend_summary()
            
            if not div_summary.empty:
                dc1, dc2, dc3 = st.columns(3)
                dc1.metric("Total Dividends Received", f"PKR {total_div:,.2f}")
                dc2.metric("Stocks with Dividends", len(div_summary))
                avg_yield = div_summary['Yield %'].mean()
                dc3.metric("Average Yield", f"{avg_yield:.2f}%")
                
                st.markdown("### Dividend Summary by Stock")
                st.dataframe(
                    div_summary.style.format({
                        "Total Dividends": "{:,.2f}",
                        "Total Cost": "{:,.2f}",
                        "Yield %": "{:.2f}%"
                    }),
                    use_container_width=True
                )
                
                st.markdown("### Dividend History")
                dividends = get_dividends(user_id)
                if not dividends.empty:
                    div_col1, div_col2 = st.columns([3, 1])
                    with div_col1:
                        st.dataframe(
                            dividends.style.format({
                                "amount_per_share": "{:,.2f}",
                                "total_amount": "{:,.2f}"
                            }),
                            use_container_width=True
                        )
                    with div_col2:
                        del_div_id = st.number_input("Dividend ID to Delete", min_value=0, step=1, key="del_div")
                        if st.button("Delete Dividend", type="secondary"):
                            if del_div_id > 0:
                                delete_dividend(del_div_id, user_id)
                                st.success(f"Deleted dividend {del_div_id}")
                                time.sleep(1)
                                st.rerun()
            else:
                st.info("No dividends recorded yet. Use the 'Record Dividend' form above to add dividend payments.")
        
        # --- TAB 5: REALIZED P&L ---
        with tab5:
            st.subheader("📊 Realized P&L Report")
            st.caption("Track profits and losses from closed (sold) positions.")
            
            realized_summary, total_realized, total_winners, total_losers = pm.get_realized_pnl_summary()
            
            if not realized_summary.empty:
                rc1, rc2, rc3, rc4 = st.columns(4)
                rc1.metric("Total Realized P&L", f"PKR {total_realized:,.2f}",
                          delta_color="normal" if total_realized >= 0 else "inverse")
                rc2.metric("Winning Trades", f"PKR {total_winners:,.2f}")
                rc3.metric("Losing Trades", f"PKR {total_losers:,.2f}")
                win_rate = (len(realized_summary[realized_summary['Realized P&L'] > 0]) / len(realized_summary)) * 100 if len(realized_summary) > 0 else 0
                rc4.metric("Win Rate", f"{win_rate:.1f}%")
                
                st.markdown("### Realized P&L by Stock")
                st.dataframe(
                    realized_summary.style.format({
                        "Realized P&L": "{:,.2f}",
                        "Shares Sold": "{:.0f}"
                    }).background_gradient(subset=['Realized P&L'], cmap="RdYlGn"),
                    use_container_width=True
                )
                
                st.markdown("### Trade History")
                realized_df = get_realized_pnl(user_id)
                if not realized_df.empty:
                    st.dataframe(
                        realized_df.style.format({
                            "buy_avg_price": "{:,.2f}",
                            "sell_price": "{:,.2f}",
                            "realized_pnl": "{:,.2f}"
                        }).background_gradient(subset=['realized_pnl'], cmap="RdYlGn"),
                        use_container_width=True
                    )
                
                if len(realized_summary) > 0:
                    fig_realized = px.bar(realized_summary, x='Stock', y='Realized P&L', 
                                          color='Realized P&L', color_continuous_scale='RdYlGn',
                                          title='Realized P&L by Stock')
                    st.plotly_chart(fig_realized, use_container_width=True)
            else:
                st.info("No realized P&L yet. Sell positions will automatically track realized gains/losses.")

    else:
        st.info("No active holdings. Add a transaction to see your portfolio.")


def _consolidate_transactions_across_pdfs(transactions: list) -> list:
    """
    Consolidate transactions of the same symbol and type across multiple PDFs.
    Uses weighted average for price calculation.
    """
    from collections import defaultdict
    
    # Group by symbol and type
    grouped = defaultdict(list)
    for t in transactions:
        key = (t.get('symbol', '').upper().strip(), t.get('type', 'BUY'))
        grouped[key].append(t)
    
    consolidated = []
    for (symbol, trans_type), trans_list in grouped.items():
        if len(trans_list) == 1:
            consolidated.append(trans_list[0])
        else:
            # Calculate weighted average price
            total_qty = sum(t.get('quantity', 0) for t in trans_list)
            total_amount = sum(t.get('quantity', 0) * t.get('price', 0) for t in trans_list)
            total_fees = sum(t.get('fees', 0) for t in trans_list)
            
            avg_price = total_amount / total_qty if total_qty > 0 else 0
            
            # Use the earliest date
            dates = [t.get('date') for t in trans_list if t.get('date')]
            earliest_date = min(dates) if dates else datetime.today().strftime('%Y-%m-%d')
            
            # Combine source files
            source_files = list(set(t.get('source_file', '') for t in trans_list))
            source_str = ', '.join(source_files[:3])
            if len(source_files) > 3:
                source_str += f" (+{len(source_files) - 3} more)"
            
            consolidated.append({
                'symbol': symbol,
                'type': trans_type,
                'quantity': total_qty,
                'price': avg_price,
                'amount': total_amount,
                'fees': total_fees,
                'date': earliest_date,
                'source_file': source_str
            })
    
    return consolidated


# ================= STOCK EXPLORER (STANDALONE CHARTS FOR ALL STOCKS) =================
def render_stock_explorer():
    st.subheader("📈 Stock Explorer")
    st.caption("Browse and analyze any stock from KSE100 or KMI30 indices.")
    
    # Load all available stocks from scraped data
    kse_df, kmi_df = load_data_from_file()
    
    all_stocks = set()
    stock_prices = {}
    
    if kse_df is not None and not kse_df.empty:
        for _, row in kse_df.iterrows():
            stock = str(row['Stock']).strip()
            all_stocks.add(stock)
            stock_prices[stock] = {'price': row.get('Price_KSE100', 0), 'index': 'KSE100'}
    
    if kmi_df is not None and not kmi_df.empty:
        for _, row in kmi_df.iterrows():
            stock = str(row['Stock']).strip()
            all_stocks.add(stock)
            if stock not in stock_prices:
                stock_prices[stock] = {'price': row.get('Price_KMI30', 0), 'index': 'KMI30'}
            else:
                stock_prices[stock]['index'] = 'Both'
    
    if not all_stocks:
        st.warning("No stock data available. Please go to 'Investment Planner' and fetch/load index data first.")
        return
    
    # Sort alphabetically
    all_stocks_list = sorted(list(all_stocks))
    
    # Filters and Selection
    col_filter, col_select, col_period = st.columns([1, 2, 1])
    
    with col_filter:
        index_filter = st.selectbox("Filter by Index", ["All", "KSE100", "KMI30", "Both"], key="explorer_index_filter")
    
    # Apply filter
    if index_filter != "All":
        filtered_stocks = [s for s in all_stocks_list if stock_prices.get(s, {}).get('index') == index_filter or (index_filter == "Both" and stock_prices.get(s, {}).get('index') == 'Both')]
    else:
        filtered_stocks = all_stocks_list
    
    with col_select:
        selected_stock = st.selectbox("Select Stock", filtered_stocks, key="explorer_stock_select")
    
    with col_period:
        period_options = {"1 Month": "1mo", "3 Months": "3mo", "6 Months": "6mo", "1 Year": "1y", "2 Years": "2y", "5 Years": "5y"}
        selected_period_label = st.selectbox("Time Period", list(period_options.keys()), index=2, key="explorer_period_select")
        selected_period = period_options[selected_period_label]
    
    # Technical Indicator selector
    col_ind1, col_ind2 = st.columns(2)
    with col_ind1:
        show_indicators = st.multiselect("Technical Indicators", ["Bollinger Bands", "RSI", "MACD"], default=[], key="explorer_indicators")
    with col_ind2:
        show_advanced = st.multiselect("Advanced Analysis", ["Support/Resistance Zones", "Pattern Recognition"], default=[], key="explorer_advanced")
    
    if selected_stock:
        with st.spinner(f"Loading chart for {selected_stock}..."):
            chart_data = get_stock_chart_data(selected_stock, period=selected_period)
            if not chart_data.empty:
                chart_data = add_technical_indicators(chart_data)
        
        if not chart_data.empty:
            # --- ADVANCED ANALYSIS: Support/Resistance & Patterns ---
            sr_levels = None
            pattern_result = None
            
            if "Support/Resistance Zones" in show_advanced:
                sr_detector = SupportResistanceDetector(window=5, tolerance_pct=1.5, min_touches=2)
                sr_levels = sr_detector.get_nearest_levels(chart_data, n_levels=3)
            
            if "Pattern Recognition" in show_advanced:
                pattern_detector = PatternRecognizer(window=5, tolerance_pct=3.0)
                pattern_result = pattern_detector.detect_pattern(chart_data)
            
            # Stock Info Metrics
            stock_info = stock_prices.get(selected_stock, {})
            current_price = stock_info.get('price', 0)
            index_membership = stock_info.get('index', 'Unknown')
            
            # Calculate some basic stats from chart data
            if len(chart_data) > 1:
                price_change = chart_data['Close'].iloc[-1] - chart_data['Close'].iloc[0]
                price_change_pct = (price_change / chart_data['Close'].iloc[0]) * 100 if chart_data['Close'].iloc[0] > 0 else 0
                high_52w = chart_data['High'].max()
                low_52w = chart_data['Low'].min()
                avg_volume = chart_data['Volume'].mean()
            else:
                price_change = price_change_pct = high_52w = low_52w = avg_volume = 0
            
            # Metrics Row
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Current Price", f"PKR {current_price:,.2f}")
            m2.metric(f"{selected_period_label} Change", f"{price_change_pct:.2f}%", delta=f"{price_change:,.2f}")
            m3.metric("Period High", f"PKR {high_52w:,.2f}")
            m4.metric("Period Low", f"PKR {low_52w:,.2f}")
            m5.metric("Index", index_membership)
            
            # Chart Type Selection
            chart_type = st.radio("Chart Type", ["Candlestick", "Line"], horizontal=True, key="explorer_chart_type")
            
            # Determine number of rows for subplots
            num_indicator_rows = sum([1 for i in ["RSI", "MACD"] if i in show_indicators])
            total_rows = 2 + num_indicator_rows  # Price + Volume + indicators
            
            row_heights = [0.5] + [0.15] * (total_rows - 1)
            
            fig = make_subplots(
                rows=total_rows, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=row_heights
            )
            
            if chart_type == "Candlestick":
                fig.add_trace(go.Candlestick(
                    x=chart_data['Date'],
                    open=chart_data['Open'],
                    high=chart_data['High'],
                    low=chart_data['Low'],
                    close=chart_data['Close'],
                    name='Price'
                ), row=1, col=1)
            else:
                fig.add_trace(go.Scatter(
                    x=chart_data['Date'], y=chart_data['Close'],
                    mode='lines', name='Close Price',
                    line=dict(color='#2196F3', width=2),
                    fill='tozeroy', fillcolor='rgba(33, 150, 243, 0.1)'
                ), row=1, col=1)
            
            # Add Moving Averages
            fig.add_trace(go.Scatter(
                x=chart_data['Date'], y=chart_data['MA20'],
                mode='lines', name='20-Day MA',
                line=dict(color='orange', width=1)
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=chart_data['Date'], y=chart_data['MA50'],
                mode='lines', name='50-Day MA',
                line=dict(color='purple', width=1)
            ), row=1, col=1)
            
            # Bollinger Bands
            if "Bollinger Bands" in show_indicators:
                fig.add_trace(go.Scatter(
                    x=chart_data['Date'], y=chart_data['BB_Upper'],
                    mode='lines', name='BB Upper',
                    line=dict(color='rgba(128,128,128,0.5)', width=1, dash='dash')
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=chart_data['Date'], y=chart_data['BB_Lower'],
                    mode='lines', name='BB Lower',
                    line=dict(color='rgba(128,128,128,0.5)', width=1, dash='dash'),
                    fill='tonexty', fillcolor='rgba(128,128,128,0.1)'
                ), row=1, col=1)
            
            # Volume
            colors = ['green' if chart_data['Close'].iloc[i] >= chart_data['Open'].iloc[i] else 'red' 
                      for i in range(len(chart_data))]
            fig.add_trace(go.Bar(
                x=chart_data['Date'], y=chart_data['Volume'],
                marker_color=colors, name='Volume', showlegend=False
            ), row=2, col=1)
            
            current_row = 3
            
            # RSI
            if "RSI" in show_indicators:
                fig.add_trace(go.Scatter(
                    x=chart_data['Date'], y=chart_data['RSI'],
                    mode='lines', name='RSI',
                    line=dict(color='purple', width=1)
                ), row=current_row, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_row, col=1)
                fig.update_yaxes(title_text="RSI", row=current_row, col=1)
                current_row += 1
            
            # MACD
            if "MACD" in show_indicators:
                fig.add_trace(go.Scatter(
                    x=chart_data['Date'], y=chart_data['MACD'],
                    mode='lines', name='MACD',
                    line=dict(color='blue', width=1)
                ), row=current_row, col=1)
                
                fig.add_trace(go.Scatter(
                    x=chart_data['Date'], y=chart_data['MACD_Signal'],
                    mode='lines', name='Signal',
                    line=dict(color='orange', width=1)
                ), row=current_row, col=1)
                
                colors_macd = ['green' if val >= 0 else 'red' for val in chart_data['MACD_Hist']]
                fig.add_trace(go.Bar(
                    x=chart_data['Date'], y=chart_data['MACD_Hist'],
                    marker_color=colors_macd, name='MACD Hist', showlegend=False
                ), row=current_row, col=1)
                fig.update_yaxes(title_text="MACD", row=current_row, col=1)
            
            # --- ADD SUPPORT/RESISTANCE ZONES TO CHART ---
            if sr_levels:
                # Add support lines (green, dashed)
                for i, sup in enumerate(sr_levels.get('support_levels', [])):
                    opacity = 0.8 - (i * 0.2)  # Stronger lines for closer levels
                    fig.add_hline(
                        y=sup['level'], 
                        line_dash="dot", 
                        line_color=f"rgba(0, 200, 0, {opacity})",
                        line_width=2,
                        annotation_text=f"S{i+1}: {sup['level']:.2f} ({sup['touches']} touches)",
                        annotation_position="left",
                        row=1, col=1
                    )
                
                # Add resistance lines (red, dashed)
                for i, res in enumerate(sr_levels.get('resistance_levels', [])):
                    opacity = 0.8 - (i * 0.2)
                    fig.add_hline(
                        y=res['level'], 
                        line_dash="dot", 
                        line_color=f"rgba(255, 0, 0, {opacity})",
                        line_width=2,
                        annotation_text=f"R{i+1}: {res['level']:.2f} ({res['touches']} touches)",
                        annotation_position="right",
                        row=1, col=1
                    )
            
            fig.update_layout(
                title=f"{selected_stock} - {selected_period_label} Price Chart",
                height=500 + (num_indicator_rows * 150),
                template="plotly_white",
                xaxis_rangeslider_visible=False,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig.update_yaxes(title_text="Price (PKR)", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # --- ANALYSIS PANELS ---
            analysis_cols = st.columns(2)
            
            # Left Column: Technical Indicator Interpretations
            with analysis_cols[0]:
                # RSI Interpretation
                if "RSI" in show_indicators and 'RSI' in chart_data.columns and not chart_data['RSI'].isna().all():
                    latest_rsi = chart_data['RSI'].iloc[-1]
                    if latest_rsi > 70:
                        st.warning(f"⚠️ **RSI: {latest_rsi:.1f}** - Stock may be **overbought**")
                    elif latest_rsi < 30:
                        st.success(f"✅ **RSI: {latest_rsi:.1f}** - Stock may be **oversold**")
                    else:
                        st.info(f"ℹ️ **RSI: {latest_rsi:.1f}** - Stock is in neutral zone")
                
                # MACD Interpretation
                if "MACD" in show_indicators and 'MACD' in chart_data.columns:
                    latest_macd = chart_data['MACD'].iloc[-1]
                    latest_signal = chart_data['MACD_Signal'].iloc[-1]
                    if latest_macd > latest_signal:
                        st.success(f"📈 **MACD Bullish** - MACD ({latest_macd:.2f}) above Signal ({latest_signal:.2f})")
                    else:
                        st.error(f"📉 **MACD Bearish** - MACD ({latest_macd:.2f}) below Signal ({latest_signal:.2f})")
            
            # Right Column: Advanced Analysis
            with analysis_cols[1]:
                # Pattern Recognition Display
                if pattern_result and pattern_result.get('pattern'):
                    signal = pattern_result.get('signal', '')
                    pattern_name = pattern_result.get('pattern', '')
                    if signal == 'bullish':
                        st.success(f"🎯 **Pattern Detected: {pattern_name}**")
                    else:
                        st.warning(f"🎯 **Pattern Detected: {pattern_name}**")
                elif "Pattern Recognition" in show_advanced:
                    st.info("ℹ️ No clear chart pattern detected at current levels")
                
                # Support/Resistance Summary
                if sr_levels:
                    st.markdown("##### 📊 Key Levels")
                    sr_col1, sr_col2 = st.columns(2)
                    with sr_col1:
                        st.markdown("**Support Zones:**")
                        if sr_levels.get('support_levels'):
                            for i, s in enumerate(sr_levels['support_levels']):
                                dist_pct = ((sr_levels['current_price'] - s['level']) / sr_levels['current_price']) * 100
                                st.write(f"S{i+1}: PKR {s['level']:,.2f} ({dist_pct:.1f}% below)")
                        else:
                            st.write("No clear support levels")
                    with sr_col2:
                        st.markdown("**Resistance Zones:**")
                        if sr_levels.get('resistance_levels'):
                            for i, r in enumerate(sr_levels['resistance_levels']):
                                dist_pct = ((r['level'] - sr_levels['current_price']) / sr_levels['current_price']) * 100
                                st.write(f"R{i+1}: PKR {r['level']:,.2f} ({dist_pct:.1f}% above)")
                        else:
                            st.write("No clear resistance levels")
            
            # --- FORECASTING SECTION FOR STOCK EXPLORER ---
            st.divider()
            st.markdown("### 🔮 AI Price Forecast")
            
            if not is_model_available():
                st.warning("⚠️ No forecasting models installed.")
                st.code('pip install "chronos-forecasting>=2.0"  # For Chronos-2\npip install toto-ts  # For Toto', language="bash")
                st.caption("Install at least one package and restart the app to enable AI-powered price forecasting.")
            else:
                with st.expander("📈 Generate Price Forecast (Click to expand)", expanded=False):
                    # Get available models
                    available_models = get_available_models()
                    model_names = [m[0] for m in available_models]
                    model_descriptions = {m[0]: m[1] for m in available_models}
                    
                    st.caption("Uses AI models to forecast future stock prices based on historical patterns.")
                    
                    # Model selection
                    selected_model = st.selectbox(
                        "Select AI Model",
                        model_names,
                        key=f"explorer_model_select_{selected_stock}",
                        help="Choose the forecasting model"
                    )
                    st.caption(f"ℹ️ {model_descriptions.get(selected_model, '')}")
                    
                    # Fetch maximum historical data for forecasting
                    with st.spinner("Loading maximum historical data..."):
                        forecast_chart_data = get_stock_chart_data(selected_stock, period="max")
                    
                    available_days = len(forecast_chart_data) if not forecast_chart_data.empty else 0
                    
                    # Set max context based on model
                    model_max_context = 4096 if selected_model == "Toto" else 8192
                    
                    fc_col1, fc_col2, fc_col3 = st.columns(3)
                    with fc_col1:
                        forecast_days = st.slider("Forecast Period (Days)", min_value=7, max_value=1024, value=30, key=f"explorer_forecast_days_{selected_stock}",
                                                 help="Number of days to forecast (max 1024)")
                    with fc_col2:
                        max_context = min(available_days, model_max_context) if available_days > 100 else model_max_context
                        context_days = st.slider("Historical Context (Days)", min_value=100, max_value=max_context, value=min(1024, max_context), key=f"explorer_context_days_{selected_stock}",
                                                help=f"More historical data = better predictions (max {model_max_context})")
                    with fc_col3:
                        st.info(f"📊 Available: {available_days} days")
                    
                    if st.button("🔮 Generate Forecast", key=f"explorer_forecast_btn_{selected_stock}", type="primary"):
                        with st.spinner(f"🤖 {selected_model} is analyzing patterns and generating forecast..."):
                            forecast_result = forecast_stock_price(
                                forecast_chart_data,
                                prediction_length=forecast_days,
                                context_length=context_days,
                                model_name=selected_model
                            )
                        
                        if forecast_result.get('success'):
                            summary = get_forecast_summary(forecast_result)
                            
                            if summary:
                                st.success(f"✅ Forecast generated successfully using {summary.get('model', selected_model)}!")
                                
                                sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
                                sum_col1.metric("Current Price", f"PKR {summary['current_price']:,.2f}")
                                sum_col2.metric(
                                    f"Forecast ({summary['days_ahead']}d)", 
                                    f"PKR {summary['forecast_end_price']:,.2f}",
                                    delta=f"{summary['price_change_pct']:+.2f}%"
                                )
                                sum_col3.metric("Forecast Range", f"{summary['forecast_low']:,.0f} - {summary['forecast_high']:,.0f}")
                                sum_col4.metric("Trend Signal", f"{summary['trend_emoji']} {summary['trend']}")
                            
                            # Create forecast chart
                            fig_forecast = go.Figure()
                            
                            fig_forecast.add_trace(go.Scatter(
                                x=chart_data['Date'],
                                y=chart_data['Close'],
                                mode='lines',
                                name='Historical Price',
                                line=dict(color='#2196F3', width=2)
                            ))
                            
                            forecast_dates = forecast_result['dates']
                            
                            if forecast_result.get('lower_90') and forecast_result.get('upper_90'):
                                fig_forecast.add_trace(go.Scatter(
                                    x=forecast_dates + forecast_dates[::-1],
                                    y=forecast_result['upper_90'] + forecast_result['lower_90'][::-1],
                                    fill='toself',
                                    fillcolor='rgba(255, 165, 0, 0.1)',
                                    line=dict(color='rgba(255,255,255,0)'),
                                    name='90% Confidence',
                                    showlegend=True
                                ))
                            
                            if forecast_result.get('lower_50') and forecast_result.get('upper_50'):
                                fig_forecast.add_trace(go.Scatter(
                                    x=forecast_dates + forecast_dates[::-1],
                                    y=forecast_result['upper_50'] + forecast_result['lower_50'][::-1],
                                    fill='toself',
                                    fillcolor='rgba(255, 165, 0, 0.2)',
                                    line=dict(color='rgba(255,255,255,0)'),
                                    name='50% Confidence',
                                    showlegend=True
                                ))
                            
                            fig_forecast.add_trace(go.Scatter(
                                x=forecast_dates,
                                y=forecast_result['median'],
                                mode='lines',
                                name='Forecast (Median)',
                                line=dict(color='#FF9800', width=2, dash='dash')
                            ))
                            
                            fig_forecast.add_vline(
                                x=forecast_result['last_actual_date'],
                                line_dash="dash",
                                line_color="gray",
                                annotation_text="Forecast Start"
                            )
                            
                            fig_forecast.update_layout(
                                title=f"{selected_stock} - {forecast_days}-Day Price Forecast",
                                xaxis_title="Date",
                                yaxis_title="Price (PKR)",
                                template="plotly_white",
                                height=400,
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig_forecast, use_container_width=True)
                            
                            st.caption("⚠️ **Disclaimer**: This forecast is generated by an AI model based on historical patterns. "
                                      "It should NOT be used as the sole basis for investment decisions. Past performance does not guarantee future results.")
                        else:
                            st.error(f"❌ {forecast_result.get('error', 'Unknown error occurred')}")
            
        else:
            st.warning(f"Could not fetch chart data for {selected_stock}. Yahoo Finance may not have data for this stock.")

# ================= INVESTMENT PLANNER (OLD DASHBOARD WRAPPED) =================
def render_investment_planner():
    # STATE MANAGEMENT
    if "data" not in st.session_state:
        st.session_state.data = pd.DataFrame()
    if "kse" not in st.session_state:
        st.session_state.kse = pd.DataFrame()
    if "kmi" not in st.session_state:
        st.session_state.kmi = pd.DataFrame()
    if "last_updated" not in st.session_state:
        st.session_state.last_updated = None
    if "total_investment" not in st.session_state:
        st.session_state.total_investment = 100000
    if "commission_fee" not in st.session_state:
        st.session_state.commission_fee = 0.15

    # SIDEBAR CONTROLS FOR PLANNER
    st.sidebar.markdown("---")
    st.sidebar.subheader("Planner Settings")
    
    st.session_state.total_investment = st.sidebar.number_input(
        "Total Investment (PKR)", 
        min_value=0, 
        value=int(st.session_state.total_investment),
        step=1000,
        key="planner_invest"
    )

    st.session_state.commission_fee = st.sidebar.number_input(
        "Broker Commission (%)",
        min_value=0.0,
        value=float(st.session_state.commission_fee),
        step=0.01,
        format="%.2f",
        key="planner_comm"
    )

    col1, col2 = st.sidebar.columns(2)

    if col1.button("Fetch Latest Index Data"):
        with st.spinner("Scraping data from Sarmaaya.pk..."):
            kse, kmi = fetch_all_data()
            if kse is not None and kmi is not None:
                save_scraped_data(kse, kmi)
                df = clean_and_normalize_data(kse, kmi)
                st.session_state.kse = kse
                st.session_state.kmi = kmi
                st.session_state.data = df
                st.session_state.last_updated = time.strftime("%Y-%m-%d %H:%M:%S")
                st.success("Data fetched!")
                st.rerun()
            else:
                st.error("Failed to scrape.")

    if col2.button("Load Saved Data"):
        kse, kmi = load_data_from_file()
        if kse is not None and kmi is not None:
            df = clean_and_normalize_data(kse, kmi)
            st.session_state.kse = kse
            st.session_state.kmi = kmi
            st.session_state.data = df
            st.session_state.last_updated = "Loaded from file"
            st.success("Data loaded.")
            st.rerun()
        else:
            st.warning("No local files found.")

    if st.sidebar.button("Reset Planner Weights"):
        if not st.session_state.data.empty:
            st.session_state.data["Final Weight"] = st.session_state.data["Default Weight"]
            st.session_state.data = st.session_state.data.sort_values("Final Weight", ascending=False).reset_index(drop=True)
            st.rerun()

    # MAIN PLANNER UI
    st.subheader("Common Stocks Investment Planner (KSE100 & KMI30)")

    if st.session_state.last_updated:
        st.caption(f"Index Data Last Updated: {st.session_state.last_updated}")
    else:
        st.info("Please Fetch or Load data to begin planning.")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["KSE 100 Index", "KMI 30 Index", "Combined Planner"])

    # --- TAB 1: KSE 100 ---
    with tab1:
        st.subheader("KSE 100 Index Data")
        if not st.session_state.kse.empty:
            col_kse1, col_kse2 = st.columns([2, 1])
            with col_kse1:
                st.dataframe(st.session_state.kse, use_container_width=True, hide_index=True)
            with col_kse2:
                try:
                    top_kse = st.session_state.kse.sort_values("Weight_KSE100", ascending=False).head(10)
                    fig_kse = px.pie(top_kse, values="Weight_KSE100", names="Stock", title="Top 10 KSE Stocks")
                    st.plotly_chart(fig_kse, use_container_width=True)
                except Exception as e:
                    st.caption(f"Could not generate chart: {e}")
        else:
            st.write("No data available.")

    # --- TAB 2: KMI 30 ---
    with tab2:
        st.subheader("KMI 30 Index Data")
        if not st.session_state.kmi.empty:
            col_kmi1, col_kmi2 = st.columns([2, 1])
            with col_kmi1:
                st.dataframe(st.session_state.kmi, use_container_width=True, hide_index=True)
            with col_kmi2:
                try:
                    top_kmi = st.session_state.kmi.sort_values("Weight_KMI30", ascending=False).head(10)
                    fig_kmi = px.pie(top_kmi, values="Weight_KMI30", names="Stock", title="Top 10 KMI Stocks")
                    st.plotly_chart(fig_kmi, use_container_width=True)
                except Exception as e:
                    st.caption(f"Could not generate chart: {e}")
        else:
            st.write("No data available.")

    # --- TAB 3: COMBINED ---
    with tab3:
        if st.session_state.data.empty:
             st.caption("No Combined data.")
        else:
            st.subheader("Investment Allocation Calculator")
            
            # 1. Editable Dataframe
            edited_df = st.data_editor(
                st.session_state.data[['Stock', 'Sector', 'Price', 'Default Weight', 'Final Weight']],
                column_config={
                    "Default Weight": st.column_config.NumberColumn("Original %", format="%.2f", disabled=True),
                    "Final Weight": st.column_config.NumberColumn("Your %", format="%.2f", min_value=0, max_value=100),
                    "Price": st.column_config.NumberColumn("Price", format="%.2f", disabled=True)
                },
                disabled=["Stock", 'Sector', 'Price', 'Default Weight'],
                use_container_width=True,
                key="planner_editor",
                hide_index=True,
                height=400
            )

            # Check if weights changed (using session state to track prev version or equality check)
            if not edited_df['Final Weight'].equals(st.session_state.data['Final Weight']):
                # We need to reflect changes. 
                # Note: normalize logic is inside rebalance_weights in calculations.py if we want it
                st.session_state.data['Final Weight'] = edited_df['Final Weight']
                # Re-calculate
                st.session_state.data = calculate_allocation(
                    st.session_state.data, 
                    st.session_state.total_investment, 
                    st.session_state.commission_fee
                )
                st.rerun()

            # 2. Display Results
            calc_df = calculate_allocation(
                st.session_state.data, 
                st.session_state.total_investment, 
                st.session_state.commission_fee
            )
            
            buy_df = calc_df[calc_df["Shares to Buy"] > 0].copy()

            st.divider()
            if buy_df.empty:
                st.warning("No stocks allocated.")
            else:
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.dataframe(
                        buy_df[['Stock', 'Sector', 'Price', 'Final Weight', 'Allocated Amount', 'Shares to Buy']],
                        column_config={
                            "Price": st.column_config.NumberColumn("Price", format="%.2f"),
                            "Allocated Amount": st.column_config.NumberColumn("Cost (PKR)", format="%.0f"),
                            "Shares to Buy": st.column_config.NumberColumn("Qty", format="%.2f")
                        },
                        use_container_width=True
                    )
                with c2:
                    st.metric("Total Investment", f"{st.session_state.total_investment:,.0f}")
                    st.caption(f"Brokerage ({st.session_state.commission_fee}%): {buy_df['Transaction Cost'].sum():,.0f}")

                # Visuals
                col_viz1, col_viz2 = st.columns(2)
                with col_viz1:
                    fig_sector = px.pie(buy_df, values='Allocated Amount', names='Sector', title='Allocation by Sector')
                    st.plotly_chart(fig_sector, use_container_width=True)

                with col_viz2:
                    fig_tree = px.treemap(buy_df, path=['Sector', 'Stock'], values='Allocated Amount', title='Portfolio Treemap')
                    st.plotly_chart(fig_tree, use_container_width=True)

                if st.button("Export Plan to Excel"):
                    save_common_stocks(buy_df)
                    with open("common_stocks.xlsx", "rb") as f:
                        st.download_button("Download Excel", f, "portfolio_plan.xlsx")

if __name__ == "__main__":
    if page == "Portfolio Tracker":
        render_portfolio_tracker()
    elif page == "📈 Stock Explorer":
        render_stock_explorer()
    elif page == "Investment Planner":
        render_investment_planner()
