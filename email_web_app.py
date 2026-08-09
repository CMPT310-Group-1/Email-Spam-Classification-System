import streamlit as st
import pandas as pd
import joblib

# Cache models: to prevent model from reloading
@st.cache_resource
def load_models():
    stage1 = joblib.load('stage1.pkl')
    stage2 = joblib.load('stage2.pkl')
    return stage1, stage2

spam_model, class_spam = load_models()

# Persistent Mailboxes and Tab State
if 'inbox' not in st.session_state:
    st.session_state.inbox = []
if 'spam' not in st.session_state:
    st.session_state.spam = []
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "Inbox"
if 'file_key' not in st.session_state:
    st.session_state.file_key = str(0)

def switch_tab(tab_name):
    st.session_state.current_tab = tab_name

# Sidebar
with st.sidebar:
    st.subheader("Mailboxes")

    inbox_type = "primary" if st.session_state.current_tab == "Inbox" else "secondary"
    spam_type = "primary" if st.session_state.current_tab == "Spam" else "secondary"

    st.button(f"Inbox ({len(st.session_state.inbox)})", use_container_width=True, type=inbox_type, on_click=switch_tab, args=("Inbox",))
    st.button(f"Spam ({len(st.session_state.spam)})", use_container_width=True, type=spam_type, on_click=switch_tab, args=("Spam",))

# Header and button(s)
col1, col2, col3 = st.columns([4, 1, 1])
with col1:
    st.title("Group 1 Mailbox")

# sending 1 email
with col2:
    with st.popover("Send"):
        st.subheader("Send an Email to us:")
        with st.form(key='single_form', clear_on_submit=True):
            subject = st.text_input("Subject:")
            text = st.text_area("Content:")
            submit_button = st.form_submit_button("Send")
            
            if submit_button:
                if subject and text:
                    user_input = [subject + ' ' + text]
                    
                    spam_prediction = spam_model.predict(user_input)[0]
                    
                    new_email = {
                        "subject": subject,
                        "text": text
                    }
                    
                    # Routing predictions
                    if spam_prediction == 1:
                        # if spam:
                        class_prediction = class_spam.predict(user_input)[0]
                        new_email["classification"] = f"Spam - {class_prediction}"
                        
                        st.session_state.spam.append(new_email)
                        st.session_state.upload_error = f"Spam detected — Category: {class_prediction}"
                    else:
                        # if ham:
                        new_email["classification"] = "Legit"
                        st.session_state.inbox.append(new_email)
                        st.session_state.upload_success = "Legitimate email — Sent to Inbox"
                    st.rerun()
                else:
                    st.warning("Please enter both a subject and text.")

# sending multiple emails (as csv)
with col3:
    with st.popover("File upload"):
        st.subheader("Upload CSV File:")
        st.write("File must contain exactly two columns: **Subject** and **Content**")
        uploaded_file = st.file_uploader("", type=['csv'],key=st.session_state.file_key)
        if uploaded_file is not None:
            if st.button("Upload", use_container_width=True):
                df = pd.read_csv(uploaded_file)

                if 'Subject' in df.columns and 'Content' in df.columns:
                    df = df.dropna(subset=['Subject', 'Content'])
                    
                    df['combined'] = df['Subject'].astype(str) + " " + df['Content'].astype(str)

                    df['is_spam'] = spam_model.predict(df['combined'])
                    spam_mask = df['is_spam'] == 1
                    if spam_mask.any():
                        df.loc[spam_mask, 'category'] = class_spam.predict(df.loc[spam_mask, 'combined'])

                    spam_count, inbox_count = 0, 0
                    
                    for index, row in df.iterrows():
                        new_email = {
                            "subject": row['Subject'],
                            "text": row['Content']
                        }
                        
                        if row['is_spam'] == 1:
                            new_email["classification"] = f"Spam: {row['category']}"
                            st.session_state.spam.append(new_email)
                            spam_count += 1
                        else:
                            new_email["classification"] = "Legit"
                            st.session_state.inbox.append(new_email)
                            inbox_count += 1
                    st.session_state.upload_success = f"{inbox_count} to Inbox, {spam_count} to Spam."

                    # Clear uploaded file
                    st.session_state.file_key = str(int(st.session_state.file_key) + 1)
                    st.rerun()
                else:
                    st.error("Invalid CSV format. The file must have columns named 'Subject' and 'Content'.")

st.divider()
if 'upload_success' in st.session_state:
    st.success(st.session_state.upload_success)
    del st.session_state.upload_success

if 'upload_error' in st.session_state:
    st.error(st.session_state.upload_error)
    del st.session_state.upload_error

# Text area
if st.session_state.current_tab == "Inbox":
    st.header("Inbox")
    if not st.session_state.inbox:
        st.write("Empty.")
    else:
        for i, email in enumerate(reversed(st.session_state.inbox)):
            with st.expander(f"{email['subject']} | {email['classification']}"):
                st.write(f"**Message:** {email['text']}")

elif st.session_state.current_tab == "Spam":
    st.header("Spam Mailbox")
    if not st.session_state.spam:
        st.write("Empty.")
    else:
        for i, email in enumerate(reversed(st.session_state.spam)):
            with st.expander(f"{email['subject']} | {email['classification']}"):
                st.write(f"**Message:** {email['text']}")