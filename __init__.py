import datetime
import pandas as pd

import streamlit as st

def load_streamlit_variables():
        
    if "query_variables" not in st.session_state:
        st.session_state.query_variables = []
        
    if "evidence_variables" not in st.session_state:
        st.session_state.evidence_variables = []
        
    if "count_snr" not in st.session_state:
        st.session_state.count_snr = None
        
    if "count_clients" not in st.session_state:
        st.session_state.count_clients = None
    
    if "count_util" not in st.session_state:
        st.session_state.count_clients = None
        
    if "count_rate" not in st.session_state:
        st.session_state.count_rate = None
        
    if "count_per" not in st.session_state:
        st.session_state.count_per = None
        
    if "count_gput" not in st.session_state:
        st.session_state.count_gput = None
        
    if "cpt_snr" not in st.session_state:
        st.session_state.cpt_snr = pd.DataFrame()
        
    if "cpt_clients" not in st.session_state:
        st.session_state.cpt_clients = None
    
    if "cpt_util" not in st.session_state:
        st.session_state.cpt_util = pd.DataFrame()
        
    if "cpt_rate" not in st.session_state:
        st.session_state.cpt_rate = pd.DataFrame()
        
    if "cpt_per" not in st.session_state:
        st.session_state.cpt_per = pd.DataFrame()
        
    if "cpt_gput" not in st.session_state:
        st.session_state.cpt_gput = pd.DataFrame()
        
    if "query_variables" not in st.session_state:
        st.session_state.query_variables = []
        
    if "observed_variables" not in st.session_state:
        st.session_state.observed_variables = {}
