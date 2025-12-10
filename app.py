import streamlit as st
import pandas as pd
from supabase import create_client, Client
import re
import io
import datetime
import time
import numpy as np
from difflib import SequenceMatcher
from flask_bcrypt import Bcrypt 

# Initialize Bcrypt globally for use in hashing functions
bcrypt = Bcrypt() 

# --- 1. CONFIGURATION & DATABASE CONNECTION ---

# ---------------------------------------------------------
# 🔑 SECURITY: READ SUPABASE CREDENTIALS FROM STREAMLIT SECRETS
# ---------------------------------------------------------
try:
    # IMPORTANT: These must be configured in Streamlit Cloud > Advanced Settings > Secrets
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
except KeyError:
    # Fallback/Error message if not deployed with secrets
    SUPABASE_URL = "https://placeholder.supabase.co" 
    SUPABASE_KEY = "placeholder_key"
    # Only show this error if the placeholder keys are definitely wrong
    if SUPABASE_URL == "https://placeholder.supabase.co":
        st.error("Database connection failed. Supabase keys are missing from Streamlit secrets. Please configure SUPABASE_URL and SUPABASE_KEY.")
    
@st.cache_resource
def init_connection():
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        st.error(f"Failed to connect to Database: {e}")
        return None

supabase = init_connection()

# --- 2. DATABASE HELPER FUNCTIONS (Supabase Version) ---

def get_staff_config(staff_name):
    """Fetches configuration for a specific staff member."""
    try:
        response = supabase.table('staff_config').select("*").eq('staff_name', staff_name).execute()
        if response.data:
            return response.data[0]
        return None
    except Exception:
        return None

def get_all_staff_names():
    """Returns a list of all staff names from config."""
    try:
        response = supabase.table('staff_config').select('staff_name').execute()
        return [row['staff_name'] for row in response.data]
    except Exception:
        return []

def get_log_entry_by_id(log_id):
    """Fetches a single log entry by ID."""
    try:
        response = supabase.table('staff_logs').select("*").eq('id', log_id).execute()
        if response.data:
            return response.data[0]
        return None
    except Exception:
        return None

def get_username_by_staff_name(staff_name):
    """Finds the username associated with a staff name."""
    try:
        response = supabase.table('users').select('username').eq('staff_name', staff_name).execute()
        if response.data:
            return response.data[0]['username']
        return "Unknown"
    except Exception:
        return "Unknown"

def save_log_entry(data):
    """Inserts a new log entry."""
    # Supabase handles the ID generation automatically
    supabase.table('staff_logs').insert(data).execute()

def update_log_entry(log_id, data):
    """Updates an existing log entry."""
    supabase.table('staff_logs').update(data).eq('id', log_id).execute()

def delete_log_entry(log_id):
    """Deletes a log entry."""
    supabase.table('staff_logs').delete().eq('id', log_id).execute()

def update_staff_info(original_name, new_role, new_pay_type, new_direct_rate, new_indirect_rate, new_travel, new_password=None):
    """Updates staff configuration and optionally their password."""
    # 1. Update Config Table
    config_data = {
        'position': new_role,
        'pay_type': new_pay_type,
        'base_rate': new_direct_rate,
        'indirect_rate': new_indirect_rate,
        'travel_fee': new_travel
    }
    supabase.table('staff_config').update(config_data).eq('staff_name', original_name).execute()
    
    # 2. Update Password if provided
    if new_password:
        # Hash this password before sending!
        hashed_password = bcrypt.generate_password_hash(new_password).decode('utf-8')
        supabase.table('users').update({'password': hashed_password}).eq('staff_name', original_name).execute()

def add_new_staff(username, password, staff_name, position, pay_type, direct_rate, indirect_rate, travel_fee):
    """Adds a new staff member to both users and config tables."""
    try:
        # Hash the password before storing it for security
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8') 

        # 1. Insert into Users
        user_data = {
            'username': username,
            'password': hashed_password, # Storing the hash
            'role': 'staff',
            'staff_name': staff_name
        }
        supabase.table('users').insert(user_data).execute()
        
        # 2. Insert into Config
        config_data = {
            'staff_name': staff_name,
            'position': position,
            'pay_type': pay_type,
            'base_rate': direct_rate,
            'indirect_rate': indirect_rate,
            'travel_fee': travel_fee
        }
        supabase.table('staff_config').insert(config_data).execute()
        return True
    except Exception as e:
        st.error(f"Error adding staff: {e}")
        return False

def delete_staff(staff_name):
    """Deletes a staff member from all tables."""
    try:
        supabase.table('users').delete().eq('staff_name', staff_name).execute()
        supabase.table('staff_config').delete().eq('staff_name', staff_name).execute()
        supabase.table('staff_logs').delete().eq('staff_name', staff_name).execute()
    except Exception as e:
        st.error(f"Error deleting staff: {e}")

def get_filtered_logs(staff_filter=None, start_date=None, end_date=None):
    """Fetches logs with optional filtering."""
    query = supabase.table('staff_logs').select("*")
    
    if staff_filter and staff_filter != "All Staff":
        query = query.eq('staff_name', staff_filter)
    
    if start_date:
        query = query.gte('date', str(start_date))
    
    if end_date:
        query = query.lte('date', str(end_date))
        
    response = query.execute()
    return pd.DataFrame(response.data)

def change_user_password(username, new_password):
    """Updates password for the logged-in user."""
    # Hash password before updating
    hashed_password = bcrypt.generate_password_hash(new_password).decode('utf-8')
    supabase.table('users').update({'password': hashed_password}).eq('username', username).execute()

def calculate_total_pay(config, direct_hrs, indirect_hrs, charged_amount, is_home_session):
    direct_rate = config.get('base_rate', 0.0)
    indirect_rate = config.get('indirect_rate', 0.0)
    travel_fee_val = config.get('travel_fee', 0.0) if is_home_session else 0.0
    total_pay = 0.0
    
    if config['pay_type'] == 'Hourly':
        total_pay = (direct_hrs * direct_rate) + (indirect_hrs * indirect_rate) + travel_fee_val
    elif config['pay_type'] == 'Percentage':
        total_pay = (charged_amount * (direct_rate / 100)) + travel_fee_val
    
    return total_pay, travel_fee_val

# --- 3. RECONCILIATION LOGIC (Refined for DataFrames) ---

def run_reconciliation_logic(df_staff, df_sales):
    # Ensure DataFrame columns are lowercase for matching logic
    df_staff.columns = df_staff.columns.str.strip().str.replace(' ', '_').str.lower()
    df_sales.columns = df_sales.columns.str.strip().str.replace(' ', '_').str.lower()

    # --- Helpers ---
    def clean_name_string(name):
        if not isinstance(name, str): return ""
        return re.sub(r'[^a-z]', '', name.lower())

    def extract_name(note):
        if not isinstance(note, str): return ""
        pattern = r'^(.*?)\s+\d{1,2}(?::\d{2})?-\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM)?.*$'
        match = re.search(pattern, note, flags=re.IGNORECASE)
        if match:
            extracted = match.group(1).strip()
            extracted = re.sub(r'\s+(OT|PT|SLP|Assessment|Intervention|Report|Consultation|Session|Writing)\s*$', '', extracted, flags=re.IGNORECASE)
            return extracted.strip()
        pattern = r'\s+\d{1,2}(?::\d{2})?-\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM)?.*$'
        clean_name = re.sub(pattern, '', note, flags=re.IGNORECASE)
        return clean_name.strip()

    def calculate_keyword_score(row):
        if pd.isna(row.get('notes')) or pd.isna(row.get('item')): return 0
        staff_note = str(row['notes']).lower()
        sales_item = str(row['item']).lower()
        keywords = ['report', 'assessment', 'intervention', 'session', 'consultation', 'writing']
        score = 0
        for kw in keywords:
            if kw in staff_note and kw in sales_item: score += 10
        return score

    def extract_expected_hours(item):
        if not isinstance(item, str): return None
        item = item.lower()
        match_min = re.search(r'(\d+)\s*mins?$', item)
        if match_min: return int(match_min.group(1)) / 60.0
        match_hr = re.search(r'(\d+)\s*hours?$', item)
        if match_hr: return float(match_hr.group(1))
        return None

    def check_hours_validation(row):
        if row['Status'] != 'Matched': return 'N/A'
        staff_hrs = row.get('direct_hrs')
        expected_hrs = row.get('expected_hours')
        if pd.isna(staff_hrs) and pd.isna(expected_hrs): return 'Missing Data'
        if pd.isna(staff_hrs) or pd.isna(expected_hrs): return 'Missing Staff/Sales Hrs'
        if round(staff_hrs, 2) == round(expected_hrs, 2): return 'Match'
        return f'Mismatch: Staff Hrs ({staff_hrs}) != Expected Hrs ({expected_hrs})'

    def check_amount_final(row):
        if row['Status'] != 'Matched': return 'N/A'
        staff_charge = row.get('charged_amount', 0)
        travel_fee = row.get('travel_fee_used', 0)
        sales_subtotal = row.get('subtotal', 0)
        expected_hours = row.get('expected_hours', 0)
        outside_clinic = str(row.get('outside_clinic', 'no')).strip().lower()
        
        staff_direct_rate = row.get('Direct_Rate_Config', 160.0) 
        if pd.isna(staff_direct_rate) or staff_direct_rate == 0:
            staff_direct_rate = 160.0
        
        staff_total_charge = staff_charge + travel_fee
        expected_base_price = expected_hours * staff_direct_rate
        staff_flags_home_session = (outside_clinic == 'yes') and (travel_fee > 0)

        if round(staff_total_charge, 2) == round(sales_subtotal, 2):
            if staff_flags_home_session: return 'Match (Inc. Travel Fee)'
            return 'Match'
        
        if staff_flags_home_session:
            if round(sales_subtotal, 2) == round(expected_base_price, 2):
                return f'Mismatch: Staff Log indicates Home Session (+$20), but Sales Subtotal (${sales_subtotal}) is missing Travel Fee.'
            if round(staff_charge, 2) == 0 and round(sales_subtotal, 2) > 0:
                   return f'Mismatch: Staff Log Charged Amount is $0, but Sales is ${sales_subtotal}. (Possible Travel Fee Error)'

        if not staff_flags_home_session and travel_fee == 0:
            if round(sales_subtotal, 2) == round(expected_base_price + 20.0, 2):
                   return f'Mismatch: Sales Subtotal (${sales_subtotal}) suggests Home Session (+$20) not reflected in Staff Log.'

        return f'Mismatch: Staff Total Charge (${staff_total_charge}) != Sales Subtotal (${sales_subtotal})'

    # --- Fetch Staff Config for Rates & Pay Type ---
    # We need to fetch the config again to map it to the staff logs in the dataframe
    try:
        response = supabase.table('staff_config').select("staff_name, pay_type, base_rate").execute()
        df_config = pd.DataFrame(response.data)
    except:
        df_config = pd.DataFrame(columns=['staff_name', 'pay_type', 'base_rate'])

    if not df_config.empty:
        df_config['staff_name_lower'] = df_config['staff_name'].str.lower()
        pay_type_map = df_config.set_index('staff_name_lower')['pay_type'].to_dict()
        rate_map = df_config.set_index('staff_name_lower')['base_rate'].to_dict()
    else:
        pay_type_map = {}
        rate_map = {}

    # 1. Staff Log Processing
    df_staff['staff_name_lower'] = df_staff['staff_name'].astype(str).str.lower()
    df_staff['Pay_Type'] = df_staff['staff_name_lower'].map(pay_type_map).fillna('Unknown')
    df_staff['Direct_Rate_Config'] = df_staff['staff_name_lower'].map(rate_map).fillna(160.0)
    df_staff.drop(columns=['staff_name_lower'], inplace=True)

    df_staff['date_obj'] = pd.to_datetime(df_staff['date'], errors='coerce')
    df_staff['date_str'] = df_staff['date_obj'].dt.normalize().astype(str).str[:10]
    df_staff['extracted_name'] = df_staff['client_name']
    df_staff['name_norm'] = df_staff['extracted_name'].apply(clean_name_string)
    for col in ['travel_fee_used', 'charged_amount', 'direct_hrs']:
        df_staff[col] = pd.to_numeric(df_staff[col], errors='coerce').fillna(0)

    # 2. Sales Record Processing
    df_sales = df_sales.dropna(subset=['patient', 'invoice_date'])
    df_sales['dt_obj'] = pd.to_datetime(df_sales['invoice_date'], errors='coerce')
    if df_sales['dt_obj'].dt.tz is None:
        # Assuming the data is in the local timezone (America/Vancouver)
        df_sales['dt_obj'] = df_sales['dt_obj'].dt.tz_localize('America/Vancouver', ambiguous='NaT')
    df_sales['date_str'] = df_sales['dt_obj'].dt.normalize().astype(str).str[:10]
    df_sales['patient_norm'] = df_sales['patient'].apply(clean_name_string)
    df_sales['expected_hours'] = df_sales['item'].apply(extract_expected_hours)
    df_sales['staff_member_lower'] = df_sales['staff_member'].astype(str).str.lower()

    # 3. Matching
    df_staff['staff_id'] = df_staff.index
    df_sales['sales_id'] = df_sales.index

    potential_matches = pd.merge(df_staff, df_sales, left_on=['name_norm'], right_on=['patient_norm'], how='outer', suffixes=('_staff', '_sales'))
    
    def get_date_diff(row):
        if pd.notna(row['date_obj']) and pd.notna(row['dt_obj']):
            # Compare only the date component
            return abs((row['date_obj'].date() - row['dt_obj'].date()).days)
        return 999 

    potential_matches['date_diff'] = potential_matches.apply(get_date_diff, axis=1)
    candidates = potential_matches[potential_matches['date_diff'] <= 1].copy()
    candidates['service_score'] = candidates.apply(calculate_keyword_score, axis=1)
    
    def get_match_score(row):
        similarity = SequenceMatcher(None, str(row['name_norm']), str(row['patient_norm'])).ratio()
        # High score for date match (0 days diff), lower score for 1 day diff
        score = (100 - row['date_diff']) + row['service_score']
        if similarity < 0.7: score -= 20 
        return score

    candidates['match_score'] = candidates.apply(get_match_score, axis=1)
    candidates = candidates.sort_values(by='match_score', ascending=False)
    
    matched_staff_ids = set()
    matched_sales_ids = set()
    final_rows = []

    for _, row in candidates.iterrows():
        sid, slid = row['staff_id'], row['sales_id']
        if pd.notna(sid) and pd.notna(slid) and sid not in matched_staff_ids and slid not in matched_sales_ids:
            match_dict = row.to_dict()
            match_dict['Status'] = 'Matched'
            match_dict['Match_Type'] = f"Match (Diff: {row['date_diff']} days)"
            match_dict['Staff_Name_Final'] = row['staff_name'] 
            final_rows.append(match_dict)
            matched_staff_ids.add(sid)
            matched_sales_ids.add(slid)

    # Unmatched Staff
    unmatched_staff = df_staff[~df_staff['staff_id'].isin(matched_staff_ids)].copy()
    for _, row in unmatched_staff.iterrows():
        new_row = row.to_dict()
        if 'date_str' in new_row: new_row['date_str_staff'] = new_row.pop('date_str')
        new_row['Status'], new_row['Match_Type'] = 'In Staff Log Only (Missing in Sales)', 'N/A'
        new_row['Staff_Name_Final'] = row['staff_name']
        new_row.update({'invoice_date': None, 'patient': None, 'item': None, 'subtotal': None, 'expected_hours': None, 'dt_obj': None, 'date_str_sales': None, 'patient_norm': None, 'staff_member_lower': None, 'sales_id': None})
        final_rows.append(new_row)

    # Unmatched Sales
    unmatched_sales = df_sales[~df_sales['sales_id'].isin(matched_sales_ids)].copy()
    for _, row in unmatched_sales.iterrows():
        new_row = row.to_dict()
        if 'date_str' in new_row: new_row['date_str_sales'] = new_row.pop('date_str')
        inferred_pay_type = pay_type_map.get(new_row.get('staff_member_lower'), 'Unknown')
        inferred_rate = rate_map.get(new_row.get('staff_member_lower'), 160.0)

        new_row['Staff_Name_Final'] = row['staff_member']
        new_row.update({
            'date_str_staff': None, 'date': None, 'extracted_name': None, 'notes': None, 
            'charged_amount': None, 'direct_hrs': None, 'outside_clinic': None, 
            'travel_fee_used': None, 'Pay_Type': inferred_pay_type, 'Direct_Rate_Config': inferred_rate,
            'date_obj': None, 'name_norm': None, 'staff_id': None, 'client_name': None, 'indirect_hrs': None, 'total_pay': None
        })
        new_row['Status'], new_row['Match_Type'] = 'In Sales Record Only (Missing in Log)', 'N/A'
        final_rows.append(new_row)

    final_df = pd.DataFrame(final_rows)
    if not final_df.empty:
        # Re-map date objects that might be NaT from merge
        final_df['date_obj'] = pd.to_datetime(final_df['date_obj'], errors='coerce')
        final_df['dt_obj'] = pd.to_datetime(final_df['dt_obj'], errors='coerce')
        
        final_df['Amount_Status'] = final_df.apply(check_amount_final, axis=1)
        final_df['Hours_Validation_Status'] = final_df.apply(check_hours_validation, axis=1)
        final_df['Display_Date'] = final_df['date_str_staff'].fillna(final_df['date_str_sales'])
    
    return final_df

# --- 4. PAGE UI FUNCTIONS ---

# Page Configuration
st.set_page_config(page_title="Clinic Portal", layout="wide")

def login_page():
    st.markdown("## 🔐 Clinic Portal Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        try:
            # Correct login logic using bcrypt for secure password checking
            # 1. Retrieve the user by username only
            response = supabase.table('users').select("*").eq('username', username).execute() 
            
            if response.data:
                user = response.data[0]
                stored_hash = user['password']
                
                # 2. Use flask_bcrypt to check the entered password against the stored hash
                if bcrypt.check_password_hash(stored_hash, password):
                    # Login Success
                    st.session_state['logged_in'] = True
                    st.session_state['user_role'] = user['role']
                    st.session_state['staff_name'] = user['staff_name']
                    st.toast(f"Welcome back, {user['staff_name']}!", icon="👋")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    # Password mismatch
                    st.error("Invalid Username or Password")
            else:
                # Username not found
                st.error("Invalid Username or Password")
        except Exception as e:
            st.error(f"Login Error: {e}")

def change_password_section(username):
    with st.expander("🔐 Change Password"):
        with st.form("change_pass_form"):
            new_pass = st.text_input("New Password", type="password")
            confirm_pass = st.text_input("Confirm New Password", type="password")
            if st.form_submit_button("Update Password"):
                if new_pass and new_pass == confirm_pass:
                    change_user_password(username, new_pass)
                    st.toast("Password updated successfully!", icon="✅")
                else:
                    st.error("Passwords do not match or are empty.")

def staff_entry_page():
    st.markdown(f"## 👋 Welcome, {st.session_state['staff_name']}")
    
    config = get_staff_config(st.session_state['staff_name'])
    if config is None:
        st.error("Configuration not found. Please contact Admin.")
        return

    # Monthly Pay Estimate
    today = datetime.date.today()
    first_day_of_month = today.replace(day=1)
    df_monthly = get_filtered_logs(st.session_state['staff_name'], first_day_of_month, today)
    monthly_pay = df_monthly['total_pay'].sum() if not df_monthly.empty else 0.0

    st.metric(label=f"💰 Estimated Pay for {today.strftime('%B')}", value=f"${monthly_pay:,.2f}")
    st.markdown("---")
    
    direct_rate = config.get('base_rate', 0.0)
    indirect_rate = config.get('indirect_rate', 0.0)
    travel_fee = config.get('travel_fee', 0.0)

    st.info(f"**Position:** {config['position']} | **Pay Type:** {config['pay_type']}")
    if config['pay_type'] == 'Hourly':
        st.caption(f"Rates: ${direct_rate}/hr (Direct) | ${indirect_rate}/hr (Indirect) | Travel Fee: ${travel_fee}")
    elif config['pay_type'] == 'Percentage':
        st.caption(f"Rates: {direct_rate}% of Charged Amt | Travel Fee: ${travel_fee}")

    # New Entry Form
    st.markdown("### 📝 Submit New Entry")
    with st.form("staff_log_form"):
        col1, col2 = st.columns(2)
        date_input = col1.date_input("Date of Service")
        client_name = col2.text_input("Client Name (First Last)")
        col3, col4 = st.columns(2)
        direct_hrs = col3.number_input("Direct Hours", min_value=0.0, step=0.5, key="new_direct_hrs")
        indirect_hrs = col4.number_input("Indirect Hours", min_value=0.0, step=0.5, key="new_indirect_hrs")
        
        charged_amount = 0.0
        if config['pay_type'] == 'Percentage':
            charged_amount = st.number_input("Charged Amount ($)", min_value=0.0, step=10.0, key="new_charged_amt")
        
        is_home_session = st.checkbox("Home Session / Outside Clinic?")
        notes = st.text_area("Notes (Optional)")
        submitted = st.form_submit_button("Submit Entry")
        
        if submitted:
            if not client_name:
                st.error("Client Name is required.")
            else:
                total_pay, travel_fee_val = calculate_total_pay(config, direct_hrs, indirect_hrs, charged_amount, is_home_session)
                outside_val = "Yes" if is_home_session else "No"
                log_data = {
                    'date': date_input.strftime('%Y-%m-%d'), 'staff_name': st.session_state['staff_name'],
                    'client_name': client_name, 'direct_hrs': direct_hrs, 'indirect_hrs': indirect_hrs,
                    'charged_amount': charged_amount, 'outside_clinic': outside_val,
                    'travel_fee_used': travel_fee_val, 'total_pay': total_pay, 'notes': f"{client_name} {notes}"
                }
                save_log_entry(log_data)
                st.toast(f"Entry Saved! Total Pay: ${total_pay:.2f}", icon="💾")
                time.sleep(1)
                st.rerun()

    # View & Edit Logs
    st.markdown("### 🔎 View & Edit Your Logs")
    col_f1, col_f2, col_f3 = st.columns(3)
    default_start_date = today - datetime.timedelta(days=30)
    filter_start_date = col_f1.date_input("Start Date", default_start_date, key="staff_filter_start_date")
    filter_end_date = col_f2.date_input("End Date", today, key="staff_filter_end_date")
    client_search = col_f3.text_input("Client Search (Partial Name)", key="client_search")

    df_logs = get_filtered_logs(st.session_state['staff_name'], filter_start_date, filter_end_date)
    if client_search and not df_logs.empty:
        df_logs = df_logs[df_logs['client_name'].str.contains(client_search, case=False, na=False)]
    
    if df_logs.empty:
        st.info("No entries found.")
    else:
        # st.dataframe uses use_container_width=True, which is the correct, modern replacement
        st.dataframe(df_logs[['id', 'date', 'client_name', 'direct_hrs', 'indirect_hrs', 'charged_amount', 'outside_clinic', 'total_pay', 'notes']].sort_values('id', ascending=False), use_container_width=True)

        st.markdown("#### ✏️ Edit or Delete an Entry")
        log_id_to_edit = st.number_input("Enter Log ID to Edit", min_value=0, step=1, key="staff_log_id_edit")
        
        if log_id_to_edit > 0:
            current_row = get_log_entry_by_id(log_id_to_edit)
            if current_row and current_row['staff_name'] == st.session_state['staff_name']:
                st.info(f"Editing Log ID: {log_id_to_edit} - Client: {current_row['client_name']}")
                with st.form("edit_log_form"):
                    col_e1, col_e2 = st.columns(2)
                    e_date = col_e1.date_input("Date", pd.to_datetime(current_row['date']), key="e_date")
                    e_client = col_e2.text_input("Client Name", current_row['client_name'], key="e_client")
                    col_e3, col_e4 = st.columns(2)
                    e_direct = col_e3.number_input("Direct Hrs", value=float(current_row['direct_hrs']), min_value=0.0, step=0.5, key="e_direct")
                    e_indirect = col_e4.number_input("Indirect Hrs", value=float(current_row['indirect_hrs']), min_value=0.0, step=0.5, key="e_indirect")
                    e_charged = float(current_row['charged_amount'])
                    if config['pay_type'] == 'Percentage':
                        e_charged = st.number_input("Charged Amt", value=float(current_row['charged_amount']), min_value=0.0, step=10.0, key="e_charged")
                    e_is_home = (current_row['outside_clinic'] == 'Yes')
                    e_outside = st.checkbox("Home Session?", value=e_is_home, key="e_outside")
                    e_notes = st.text_area("Notes", current_row['notes'], key="e_notes")
                    
                    if st.form_submit_button("Update Log"):
                        recalculated_pay, recalculated_travel = calculate_total_pay(config, e_direct, e_indirect, e_charged, e_outside)
                        updated_data = {
                            'date': e_date.strftime('%Y-%m-%d'), 'client_name': e_client, 'direct_hrs': e_direct, 'indirect_hrs': e_indirect,
                            'charged_amount': e_charged, 'outside_clinic': "Yes" if e_outside else "No", 'travel_fee_used': recalculated_travel, 'total_pay': recalculated_pay, 'notes': e_notes
                        }
                        update_log_entry(log_id_to_edit, updated_data)
                        st.toast(f"Log updated! New Pay: ${recalculated_pay:.2f}", icon="✅")
                        time.sleep(1)
                        st.rerun()
                    
                    # Delete Button OUTSIDE form
                    if st.button("Delete Log Entry", key=f"del_{log_id_to_edit}"):
                        delete_log_entry(log_id_to_edit)
                        st.toast("Log deleted.", icon="🗑️")
                        time.sleep(1)
                        st.rerun()
            elif current_row:
                 st.error("Unauthorized.")
            else:
                st.warning("Log ID not found.")
    
    # Change Password Section for Staff
    st.markdown("---")
    my_username = get_username_by_staff_name(st.session_state['staff_name'])
    change_password_section(my_username)

def admin_page():
    st.markdown(f"## 🛠️ Admin Dashboard ({st.session_state['staff_name']})")
    tab1, tab2, tab3 = st.tabs(["📊 Reconciliation", "👥 Manage Staff", "📝 View Logs"])
    
    with tab1:
        st.subheader("Run Payroll Reconciliation")
        col1, col2 = st.columns(2)
        all_staff = ["All Staff"] + get_all_staff_names()
        selected_staff = col1.selectbox("Filter by Staff", all_staff)
        col3, col4 = col2.columns(2)
        start_date = col3.date_input("Start Date", datetime.date.today().replace(day=1))
        end_date = col4.date_input("End Date", datetime.date.today())
        sales_file = st.file_uploader("Upload Sales Record (CSV)", type=['csv'])
        
        if st.button("Run Reconciliation"):
            if not sales_file:
                st.error("Please upload a Sales Record CSV.")
            else:
                df_staff_db = get_filtered_logs(selected_staff, start_date, end_date)
                if df_staff_db.empty:
                    st.warning("No staff logs found.")
                else:
                    try:
                        # Use io.StringIO to read the uploaded file contents as a string buffer
                        sales_data = sales_file.getvalue().decode("latin-1")
                        df_sales_csv = pd.read_csv(io.StringIO(sales_data), engine='python', on_bad_lines='skip')
                        
                        final_report = run_reconciliation_logic(df_staff_db, df_sales_csv)
                        
                        df_hourly = final_report[final_report['Pay_Type'] == 'Hourly']
                        df_percentage = final_report[final_report['Pay_Type'] == 'Percentage']
                        
                        def get_metrics(df, ptype):
                            matched = len(df[df['Status'] == 'Matched'])
                            staff_only = len(df[df['Status'].str.contains('Staff Log Only', na=False)])
                            sales_only = len(df[df['Status'].str.contains('Sales Record Only', na=False)])
                            err = 0
                            if ptype == 'Hourly':
                                h_err = len(df[(df['Status']=='Matched') & (df['Hours_Validation_Status'].str.startswith('Mismatch'))])
                                amt_err = len(df[(df['Status']=='Matched') & (df['Amount_Status'].str.contains('Mismatch')) & (df['Amount_Status'].str.contains('Travel Fee'))])
                                err = h_err + amt_err
                            else:
                                err = len(df[(df['Status']=='Matched') & (df['Amount_Status'].str.contains('Mismatch'))])
                            return matched, err, staff_only, sales_only

                        h_m, h_e, h_so, h_sro = get_metrics(df_hourly, 'Hourly')
                        p_m, p_e, p_so, p_sro = get_metrics(df_percentage, 'Percentage')
                        
                        st.markdown("#### Hourly Rate Staff Summary")
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Matches", h_m)
                        c2.metric("Critical Errors", h_e)
                        c3.metric("Staff Only", h_so)
                        c4.metric("Sales Only", h_sro)
                        
                        st.markdown("#### Percentage Staff Summary")
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Matches", p_m)
                        c2.metric("Amount Errors", p_e)
                        c3.metric("Staff Only", p_so)
                        c4.metric("Sales Only", p_sro)
                        
                        st.dataframe(final_report)
                        csv = final_report.to_csv(index=False).encode('utf-8')
                        st.download_button("Download Report", csv, "Reconciliation_Report.csv", "text/csv")
                    except Exception as e:
                        st.error(f"Error processing files: {e}")

    with tab2:
        st.subheader("Manage Staff Information")
        st.markdown("### ➕ Add New Staff")
        with st.form("add_staff_form"):
            col_n1, col_n2 = st.columns(2)
            new_staff_name = col_n1.text_input("Staff Full Name", key="new_staff_name")
            new_login_id = col_n2.text_input("Login ID", key="new_login_id")
            new_password = st.text_input("Default Password", type="password", key="new_password")
            
            c_r1, c_r2 = st.columns(2)
            new_role = c_r1.text_input("Position", value="OT", key="new_position")
            new_pay_type = c_r2.selectbox("Pay Type", ["Hourly", "Percentage"], key="new_pay_type")
            c_r3, c_r4 = st.columns(2)
            new_direct_rate = c_r3.number_input("Direct Rate ($/hr or %)", value=80.0, key="new_direct_rate")
            new_indirect_rate = c_r4.number_input("Indirect Rate ($/hr)", value=0.0, key="new_indirect_rate")
            new_travel = st.number_input("Travel Fee ($)", value=20.0, key="new_travel")
            
            if st.form_submit_button("Create New Staff"):
                if not new_staff_name or not new_login_id or not new_password:
                    st.error("Please fill in all fields.")
                else:
                    if add_new_staff(new_login_id, new_password, new_staff_name, new_role, new_pay_type, new_direct_rate, new_indirect_rate, new_travel):
                        st.toast(f"Staff {new_staff_name} created!", icon="🎉")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Failed. Login ID might exist.")
        
        st.markdown("---")
        st.markdown("### ✏️ Edit Existing Staff")
        staff_list = get_all_staff_names()
        selected_staff_edit = st.selectbox("Select Staff to Edit", ["Select..."] + staff_list, key="admin_select_staff_edit")
        
        if selected_staff_edit != "Select...":
            config = get_staff_config(selected_staff_edit)
            # Show Login ID (Read Only)
            current_username = get_username_by_staff_name(selected_staff_edit)
            st.text_input("Login ID (Read-Only)", value=current_username, disabled=True)
            
            with st.form("edit_staff_form"):
                c1, c2 = st.columns(2)
                new_role = c1.text_input("Position", value=config['position'])
                new_pay_type = c2.selectbox("Pay Type", ["Hourly", "Percentage"], index=0 if config['pay_type']=='Hourly' else 1)
                c3, c4 = st.columns(2)
                new_direct_rate = c3.number_input("Direct Rate ($/hr or %)", value=config['base_rate'])
                indirect_val = config['indirect_rate'] if 'indirect_rate' in config else 0.0
                new_indirect_rate = c4.number_input("Indirect Rate ($/hr)", value=indirect_val)
                new_travel = st.number_input("Travel Fee ($)", value=config['travel_fee'])
                
                # Password Reset (Admin only)
                st.markdown("##### Change Password (Optional)")
                new_password = st.text_input("Set New Password", type="password", help="Leave blank if you don't want to change the password.", key="admin_new_pass")
                
                if st.form_submit_button("Update Staff Info"):
                    update_staff_info(selected_staff_edit, new_role, new_pay_type, new_direct_rate, new_indirect_rate, new_travel, new_password)
                    st.toast("Staff info updated!", icon="✅")
                    time.sleep(1)
                    st.rerun()

            if st.button(f"Delete Staff: {selected_staff_edit}"):
                # Add confirmation step
                st.warning("Are you sure you want to delete this staff member and all associated data?")
                if st.button("Confirm Deletion", key="confirm_delete_staff"):
                    delete_staff(selected_staff_edit)
                    st.toast(f"Staff {selected_staff_edit} deleted.", icon="🗑️")
                    time.sleep(1)
                    st.rerun()

    with tab3:
        st.subheader("All Staff Logs")
        all_staff_logs = ["All Staff"] + get_all_staff_names()
        selected_staff_logs = st.selectbox("Filter Logs by Staff", all_staff_logs, key="admin_log_filter")
        
        col_log1, col_log2 = st.columns(2)
        log_start_date = col_log1.date_input("Log Start Date", datetime.date.today().replace(day=1), key="admin_log_start")
        log_end_date = col_log2.date_input("Log End Date", datetime.date.today(), key="admin_log_end")
        
        df_all_logs = get_filtered_logs(selected_staff_logs, log_start_date, log_end_date)
        if df_all_logs.empty:
            st.info("No logs found.")
        else:
            df_display = df_all_logs[['id', 'date', 'staff_name', 'client_name', 'direct_hrs', 'indirect_hrs', 'charged_amount', 'outside_clinic', 'total_pay', 'notes']].sort_values('id', ascending=False)
            st.dataframe(df_display, use_container_width=True)
            
            # Download button
            csv = df_all_logs.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Staff Logs as CSV",
                data=csv,
                file_name=f"Staff_Logs_{log_start_date}_to_{log_end_date}.csv",
                mime="text/csv",
            )
            
    # Admin Change Password
    st.markdown("---")
    my_username = get_username_by_staff_name(st.session_state['staff_name'])
    change_password_section(my_username)


# --- 5. MAIN APPLICATION LOGIC ---

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'user_role' not in st.session_state:
    st.session_state['user_role'] = None
if 'staff_name' not in st.session_state:
    st.session_state['staff_name'] = None

# Sidebar Content
with st.sidebar:
    # FIX: Replaced 'use_column_width=True' with 'width=250' to remove deprecation warning.
    st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQh1_h7nK-HkYt3f3Tf8q-BvL-2L-1e-p7lSg&usqp=CAU", width=250)
    st.markdown("# Clinic Portal")
    if st.session_state['logged_in']:
        if st.button("Logout", key="logout_btn"):
            st.session_state['logged_in'] = False
            st.session_state['user_role'] = None
            st.session_state['staff_name'] = None
            st.rerun()

# Main Page Routing
if not st.session_state['logged_in']:
    login_page()
elif st.session_state['user_role'] == 'admin':
    admin_page()
else:
    staff_entry_page()