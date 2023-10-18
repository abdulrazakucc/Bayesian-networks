import random
import pandas as pd

import streamlit as st
import streamlit.components.v1 as components
import networkx as nx
from pyvis.network import Network

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

import distinctipy as distcolors
from matplotlib.colors import rgb2hex

# from IPython.display import display, HTML

from __init__ import load_streamlit_variables

# Streamlit page configurations
st.set_page_config(
    page_title="Bayesian Networks",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Streamlit style
st_style = """
    <style>
    #MainMenu {visibility: visible;}
    footer {visibility: visible;}
    footer:after {content: 'Copyright @ 2023: Abdul Razak';
        display:block;
        position:relative
        
    } 
    header {visibility: hidden;} 
    </style>
"""

st.markdown(st_style, unsafe_allow_html=True)

# Header
st.title("Bayesian Networks")
st.write("This page provides a simple Bayesian Network model.")

# Bayesian Network parameters
BN_VARIABLES = {
    "SNR": 1,
    "N_CLIENTS": 2,
    "CHANNEL UTIL": 3,
    "DATA RATE": 4,
    "PER": 5,
    "GOODPUT": 6,
}
BN_VARIABLE_DOMAIN = {
    "POOR": "poor",
    "MEDIUM": "medium",
    "GOOD": "good",
}
DOMAIN_VALUES = {
    "POOR": 0,
    "MEDIUM": 1,
    "GOOD": 2,
}
BN_EDGES = {
    1: ("SNR", "DATA RATE"),
    2: ("N_CLIENTS", "CHANNEL UTIL"),
    3: ("CHANNEL UTIL", "PER"),
    4: ("N_CLIENTS", "GOODPUT"),
    5: ("DATA RATE", "GOODPUT"),
    6: ("PER", "GOODPUT"),
    7: ("SNR", "PER"),
}
BN_VARIABLE_LEVELS = [3, 3, 2, 2, 2, 1]
BN_DEPENDENCIES = {
    "SNR": [],
    "N_CLIENTS": [],
    "CHANNEL UTIL": ["N_CLIENTS"],
    "DATA RATE": ["SNR"],
    "PER": ["SNR", "CHANNEL UTIL"],
    "GOODPUT": ["N_CLIENTS", "DATA RATE", "PER"],
}


# Bayesian Network Graph
@st.cache_data
def draw_the_network_graph(bn_file):
    net = Network(
        height="1000px", width="100%", directed=True, bgcolor="black", font_color="white"
    )
    # add nodes to the graph
    colors = [rgb2hex(i) for i in distcolors.get_colors(len(BN_VARIABLES))]
    for i, (var, val) in enumerate(BN_VARIABLES.items()):
        net.add_node(
            val,
            shape="ellipse",
            label=var,
            level=BN_VARIABLE_LEVELS[i],
            size=110,
            color=colors[i],
        )

    # add edges between nodes
    for v in BN_EDGES.values():
        net.add_edge(BN_VARIABLES[v[0]], BN_VARIABLES[v[1]])

    # save the network
    net.set_edge_smooth("dynamic")
    net.save_graph(bn_file)


# Get the variable's CPT values
def get_the_distribution_vals(df):
    distribution = []
    df = df.sort_index(ascending=False)
    columns = list(df.columns)
    columns.sort(reverse=True)
    for col in columns:
        distribution.append(df[col].values.tolist())
    return distribution


# Bayesian model
def get_the_bn_model():
    # Bayesian Network graph
    model = BayesianNetwork(list(BN_EDGES.values()))

    # Define conditional probability distributions (CPDs)
    # SNR distribution
    cpd_snr = TabularCPD(
        variable="SNR",
        variable_card=len(BN_VARIABLE_DOMAIN),
        values=get_the_distribution_vals(st.session_state.cpt_snr),
    )
    # n_clients distribution
    cpd_clients = TabularCPD(
        variable="N_CLIENTS",
        variable_card=len(BN_VARIABLE_DOMAIN),
        values=get_the_distribution_vals(st.session_state.cpt_clients),
    )
    # Utilization distribution
    cpd_util = TabularCPD(
        variable="CHANNEL UTIL",
        variable_card=len(BN_VARIABLE_DOMAIN),
        evidence=["N_CLIENTS"],
        values=get_the_distribution_vals(st.session_state.cpt_util),
        evidence_card=[len(BN_VARIABLE_DOMAIN)],
    )
    # Data rate distribution
    cpd_rate = TabularCPD(
        variable="DATA RATE",
        variable_card=len(BN_VARIABLE_DOMAIN),
        values=get_the_distribution_vals(st.session_state.cpt_rate),
        evidence=["SNR"],
        evidence_card=[len(BN_VARIABLE_DOMAIN)],
    )

    # PER distribution
    cpd_per = TabularCPD(
        variable="PER",
        variable_card=len(BN_VARIABLE_DOMAIN),
        values=get_the_distribution_vals(st.session_state.cpt_per),
        evidence=["SNR", "CHANNEL UTIL"],
        evidence_card=[len(BN_VARIABLE_DOMAIN), len(BN_VARIABLE_DOMAIN)],
    )

    # Goodput distribution
    cpd_gput = TabularCPD(
        variable="GOODPUT",
        variable_card=len(BN_VARIABLE_DOMAIN),
        values=get_the_distribution_vals(st.session_state.cpt_gput),
        evidence=["N_CLIENTS", "DATA RATE", "PER"],
        evidence_card=[
            len(BN_VARIABLE_DOMAIN),
            len(BN_VARIABLE_DOMAIN),
            len(BN_VARIABLE_DOMAIN),
        ],
    )

    # Add CPDs to the model
    model.add_cpds(cpd_snr, cpd_clients, cpd_util, cpd_rate, cpd_per, cpd_gput)

    return model


# Inference OR MAP (Maximum a Posteriori) / MAP (Most Probable Explanation)
# @st.cache_data
def inference(_model, vars_query, vars_observed, kind="inference"):
    inference = VariableElimination(_model)
    if kind == "inference":
        return (
            inference.query(variables=vars_query, evidence=vars_observed, joint=False)
            if vars_observed
            else inference.query(variables=vars_query, joint=False)
        )
    else:
        return (
            inference.map_query(variables=vars_query, evidence=vars_observed)
            if vars_observed
            else inference.map_query(variables=vars_query)
        )


# Prepare the results to show it in the dashboard
def get_printable_inference_result(factor):
    posterior_prob = [
        i
        for i in str(factor).split()
        if ("+" not in i) & ("|" not in i) & ("phi" not in i)
    ]

    column_names = {}
    column_values = {}
    j = 0
    for i, val in enumerate(posterior_prob[1:]):
        if "(0)" in val:
            column_names[j] = "POOR"
            column_values[j] = posterior_prob[i + 2]
            j += 1
        elif "(1)" in val:
            column_names[j] = "MEDIUM"
            column_values[j] = posterior_prob[i + 2]
            j += 1
        elif "(2)" in val:
            column_names[j] = "GOOD"
            column_values[j] = posterior_prob[i + 2]
            j += 1
    return (column_names, column_values)


def get_plots(BN_model, file):
    t1, t2, t3, t4, t5 = st.tabs(
        [
            "Network Graph",
            "Inference / MAP",
            "Probabilities",
            "Frequencies",
            "Variable Definitions",
        ]
    )

    with t1:
        network_html_file = open(file, "r", encoding="utf-8")
        source_code = network_html_file.read()
        components.html(source_code, height=500, width=1500)

    with t2:
        q_vars = st.session_state.query_variables
        o_vars = st.session_state.observed_variables

        inference_text = ""
        if q_vars:
            true_observed = {
                var: DOMAIN_VALUES[dom] for var, dom in o_vars.items() if dom
            }

            analysis_type = st.sidebar.selectbox(
                "Analysis type", options=["Inference", "MAP (or MPE)"], index=None
            )

            if analysis_type == "Inference":
                result = inference(BN_model, q_vars, vars_observed=true_observed)

                if not true_observed:
                    inference_text += (
                        f"No variable is observed then the posterior distribution(s)"
                    )
                else:
                    inference_text += f"When "
                    for i, (var, dom) in enumerate(true_observed.items()):
                        inference_text += f" {var} = {[k for k, v in DOMAIN_VALUES.items() if v == dom][0]}"
                        if i < len(true_observed) - 2:
                            inference_text += "; "
                        elif (i == len(true_observed) - 2) & (len(true_observed) > 1):
                            inference_text += " and "
                    if len(true_observed) > 1:
                        inference_text += " are "
                    else:
                        inference_text += " is "
                    inference_text += "observed then the posterior distribution(s)"

                st.write(inference_text)

                for ret, q_var in zip(result.values(), q_vars):
                    st.write(f"{q_var}")
                    var_name, var_val = get_printable_inference_result(ret)

                    for i, col_tab in enumerate(st.columns(len(var_val))):
                        col_tab.metric(var_name[i], var_val[i], "")

            elif analysis_type == "MAP (or MPE)":
                result = inference(
                    BN_model, q_vars, vars_observed=true_observed, kind="map"
                )

                if true_observed:
                    net = Network()
                    root_nodes = []
                    colors = [
                        rgb2hex(i)
                        for i in distcolors.get_colors(len(true_observed) + len(result))
                    ]
                    for i, (var, dom) in enumerate(true_observed.items()):
                        root_nodes.append(i)
                        label = f"{var} = {[k for k, v in DOMAIN_VALUES.items() if v==dom][0]}"
                        net.add_node(
                            i, label=label, shape="ellipse", level=1, color=colors[i],
                        )

                    leaf_nodes = []
                    for j, (var, dom) in enumerate(result.items(), max(root_nodes) + 1):
                        leaf_nodes.append(j)
                        label = f"{var} = {[k for k, v in DOMAIN_VALUES.items() if v==dom][0]}"
                        net.add_node(
                            j, label=label, shape="ellipse", level=2, color=colors[j],
                        )

                    for i in root_nodes:
                        for j in leaf_nodes:
                            net.add_edge(i, j)

                    map_graph = "graphs/map_query.html"
                    net.set_edge_smooth("dynamic")
                    net.repulsion(node_distance=100, spring_length=200)
                    net.save_graph(map_graph)

                    network_html_file = open(map_graph, "r", encoding="utf-8")
                    source_code = network_html_file.read()
                    components.html(source_code, height=500, width=1500)

    with t3:
        if isinstance(st.session_state.cpt_gput, pd.DataFrame):
            st.write(f"SNR conditional probability table (CPT)  ")
            st.table(st.session_state.cpt_snr.sort_index(ascending=False))
            st.write(f"Clients count conditional probability table (CPT)  ")
            st.table(st.session_state.cpt_clients.sort_index(ascending=False))
            st.write(f"Channel utilization CPT ")
            st.table(st.session_state.cpt_util.sort_index(ascending=False))
            st.write(f"Data Rate CPT ")
            st.table(st.session_state.cpt_rate.sort_index(ascending=False))
            st.write("PER CPT ")
            st.table(st.session_state.cpt_per.sort_index(ascending=False))
            st.write("Goodput CPT ")
            st.table(st.session_state.cpt_gput.sort_index(ascending=False))

    with t4:
        if isinstance(st.session_state.count_gput, pd.DataFrame):
            st.write(f"SNR counts table  ")
            st.table(st.session_state.count_snr.sort_index(ascending=False).T)
            st.write(f"Clients count conditional probability table (CPT)  ")
            st.table(st.session_state.count_clients.sort_index(ascending=False).T)
            st.write(f"Channel utilization counts table ")
            st.table(st.session_state.count_util.sort_index(ascending=False))
            st.write(f"Data Rate counts table ")
            st.table(st.session_state.count_rate.sort_index(ascending=False))
            st.write("PER counts table ")
            st.table(st.session_state.count_per.sort_index(ascending=False))
            st.write("Goodput counts table ")
            st.table(st.session_state.count_gput.sort_index(ascending=False))

    with t5:
        st.write("CHANNEL UTIL")
        st.write("N_CLIENTS")
        st.write("SNR")
        st.write("DATA RATE")
        st.write("PER")
        st.write("GOODPUT")


def get_variables_list(vars):
    return list(set(list(BN_VARIABLES.keys())) - set(vars)) if vars else []


def get_query_and_evidence_variables():
    st.sidebar.title("Inference")
    query_variables = st.sidebar.multiselect(
        "Pick Query Variable(s)",
        options=BN_VARIABLES.keys(),
    )
    evidence_variables = st.sidebar.multiselect(
        "Pick evidence variable(s)",
        options=get_variables_list(query_variables),
    )

    observed_variables = {}
    for e_var in evidence_variables:
        observed_variables[e_var] = st.sidebar.selectbox(
            f"{e_var} observed value", BN_VARIABLE_DOMAIN.keys(), index=None
        )

    return (query_variables, observed_variables)


# Get conditional probability tables from the CSV files
def get_cpts():
    # SNR counts and cpts
    st.session_state.count_snr = pd.read_csv("tables/count_snr.csv", index_col=0)
    st.session_state.cpt_snr = pd.read_csv("tables/cpt_snr.csv", index_col=0)

    # N_clients counts and cpts
    st.session_state.count_clients = pd.read_csv(
        "tables/count_clients.csv", index_col=0
    )
    st.session_state.cpt_clients = pd.read_csv("tables/cpt_clients.csv", index_col=0)

    # Channel utilization counts and cpts
    st.session_state.count_util = pd.read_csv("tables/count_util.csv", index_col=0)
    st.session_state.cpt_util = pd.read_csv("tables/cpt_util.csv", index_col=0)

    # Data rate counts and cpts
    st.session_state.count_rate = pd.read_csv("tables/count_rate.csv", index_col=0)
    st.session_state.cpt_rate = pd.read_csv("tables/cpt_rate.csv", index_col=0)

    # Packet error rate counts and cpts
    st.session_state.count_per = pd.read_csv("tables/count_per.csv", index_col=[0, 1])
    st.session_state.cpt_per = pd.read_csv("tables/cpt_per.csv", index_col=[0, 1])

    # Packet error rate counts and cpts
    st.session_state.count_gput = pd.read_csv(
        "tables/count_gput.csv", index_col=[0, 1, 2]
    )
    st.session_state.cpt_gput = pd.read_csv("tables/cpt_gput.csv", index_col=[0, 1, 2])


if __name__ == "__main__":
    load_streamlit_variables()

    # Draw the Bayesian Network (as an image)
    graph_file = "graphs/toy_model.html"
    draw_the_network_graph(graph_file)

    # Get conditional probability tables (CPTs) from CSVs
    get_cpts()

    # Get the Bayesian Network model
    model = None
    if st.session_state.cpt_gput.shape[0] > 0:
        model = get_the_bn_model()

    # Get variable inputs
    (
        st.session_state.query_variables,
        st.session_state.observed_variables,
    ) = get_query_and_evidence_variables()

    # Get the dashboard plots
    get_plots(model, graph_file)
