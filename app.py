#!/usr/bin/env python3

"""
Author: Daniel.Nicorici@gmail.com

Extract info about compounds from different plant datasets (coconut, laji, gbif, etc.).

"""

import os
import sys
import math
from typing import Tuple

import pandas as pd
import streamlit as st


DEFAULT_COMPOUND = "arctigenin"

DATA_DIR = "data"

LIST_COMPOUNDS_PATH = os.path.join(DATA_DIR, "coconut_compounds.txt")
#LIST_PLANTS_FI_NO_PATH = os.path.join(DATA_DIR, "organisms_laji_gbif_FI_NO_plants.tsv")
COCONUT_DB_PATH = os.path.join(DATA_DIR,"coconut_csv-09-2025_FI_NO_plants.csv")
LAJI_DB_PATH = os.path.join(DATA_DIR,"laji2_fi.txt")
GBIF_DB_PATH = os.path.join(DATA_DIR,"gbif_plants_FI_NO_merged.tsv")
LIST_PLANTS_GENERA_PATH = os.path.join(DATA_DIR, "plants_genera.txt")




st.set_page_config(page_title="AURORA Pilot", layout="wide")

# Cap the overall content width and shrink the first (#) column
st.markdown(
    """
    <style>
    .block-container { max-width: 1200px; }
    div[data-testid="stDataFrame"] { max-width: 1200px; }
    /* Shrink the first (#) column */
    div[data-testid="stDataFrame"] th:first-child,
    div[data-testid="stDataFrame"] td:first-child {
        width: 40px !important;
        max-width: 40px !important;
        text-align: right !important;
    }
    /* Make organism column wider in HTML tables */
    table th:nth-child(2),
    table td:nth-child(2) {
        min-width: 300px !important;
        width: 300px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Results pagination size
RESULTS_PAGE_SIZE = 30

st.title("AURORA Pilot (compounds in Nordic plants)")
st.markdown("_Daniel Nicorici, Juha Klefström — University of Helsinki_")


########################################################################################
def infer_genus(name: str) -> str:
    """Infer the genus from a species name."""
    if not isinstance(name, str):
        return None    
    s = name.lower().strip()
    r = s.partition(" ")[0]
    if s.find("×") != -1 and s.count(r) > 1:
        r = s.partition("×")[0].strip()
    return r

########################################################################################
def batch_infer_genus(names: str) -> str:
    if not isinstance(names, str):
        return None
    s = sorted(set([infer_genus(e.strip()) for e in names.lower().strip().split("|") if e.strip()]))
    s = [e for e in s if e]
    return "|".join(s)
    

########################################################################################
@st.cache_data
def load_data(): 

    print("Processing short plant list of plants genera...")
    plants_genera = set()
    with open(LIST_PLANTS_GENERA_PATH,"r") as f:
         plants_genera = set([e.lower().rstrip("\r\n") for e in f if e.rstrip("\r\n")])    

    print("Processing Laji.fi database information...")
    laji = pd.read_csv(LAJI_DB_PATH, sep = "\t", low_memory=False)
    laji["name"] = laji["Scientific name"].str.lower()
    laji = laji[["name","Identifier","Observation count from Finland","Genus, Scientific name"]].copy()
    laji.columns = ["name","identifier_laji","obs. in Finland (laji)", "genus_laji"]
    laji["genus_laji"] = laji["genus_laji"].str.lower()
    laji = laji.dropna(subset=["name"])
    laji = laji.drop_duplicates()
    

    print("Processing GBIF database information...")
    gbif = pd.read_csv(GBIF_DB_PATH, sep = "\t", low_memory=False)
    gbif = gbif[["canonicalName","genus","obs_FI","obs_NO","count_FI_60N","count_NO_60N","count_FI_66N","count_NO_66N","genusKey","speciesKey"]].copy()
    gbif["canonicalName"] = gbif["canonicalName"].str.lower()
    gbif["genus"] = gbif["genus"].str.lower()
    gbif = gbif.drop_duplicates()
    gbif.columns = ["name",
                    "genus_gbif",
                    "obs. in Finland (gbif)", 
                    "obs. in Norway (gbif)",
                    "obs. in Finland >60N(gbif)", 
                    "obs. in Norway >60N (gbif)",
                    "obs. in Finland >66N (gbif)", 
                    "obs. in Norway >66N (gbif)",
                    "genusKey_gbif",
                    "speciesKey_gbif"]


    laji_gbif = pd.merge(laji,gbif,how="left",left_on="name",right_on="name")
    laji_gbif = laji_gbif.drop_duplicates()

    # build the genus
    laji_gbif["genus_guess"] = laji_gbif["name"].apply(infer_genus)
    laji_gbif["genus"] = laji_gbif["genus_gbif"].fillna(laji_gbif["genus_laji"])
    laji_gbif["genus"] = laji_gbif["genus"].fillna(laji_gbif["genus_guess"])
    laji_gbif = laji_gbif.drop(columns=["genus_guess","genus_gbif","genus_laji"])
    laji_gbif = laji_gbif.drop_duplicates()


    cols = ["obs. in Finland (gbif)", "obs. in Norway (gbif)", "obs. in Finland (laji)"]
    for c in cols:
        laji_gbif[c] = laji_gbif[c].fillna(0)

    laji_gbif["obs"] = laji_gbif[cols].sum(axis=1)
    laji_gbif = laji_gbif[laji_gbif["obs"] > 2]  # keep only species with more than 2 observations in Finland or Norway
    laji_gbif = laji_gbif.drop(columns=["obs"])
    laji_gbif = laji_gbif.drop_duplicates()
    laji_gbif = laji_gbif[laji_gbif["genus"].notna()]
    laji_gbif["url_laji"] = "https://laji.fi/taxon/" + laji_gbif["identifier_laji"] + "/occurrence"
    laji_gbif["url_gbif"] = laji_gbif["speciesKey_gbif"].apply( lambda x: f"https://www.gbif.org/species/{x}" if pd.notna(x) else pd.NA)
    laji_gbif["url"] = laji_gbif["url_laji"].fillna(laji_gbif["url_gbif"])
    laji_gbif = laji_gbif.drop(columns=["identifier_laji","genusKey_gbif","speciesKey_gbif","url_laji","url_gbif"])


    print("Processing Coconut database information...")   
    coconut = pd.read_csv(COCONUT_DB_PATH, sep = "\t",low_memory=False)
    coconut = coconut.dropna(subset=["name","identifier"])
    coconut = coconut.drop(columns=["identifier"])
    coconut["name"] = coconut["name"].str.lower().str.strip()    
    coconut["organisms"] = coconut["organisms"].str.lower().str.strip()        
    coconut["genus_coconut"] = coconut["organisms"].apply(batch_infer_genus)
    coconut = coconut.drop_duplicates()

    compounds = sorted(set(coconut["name"].unique().tolist()))
    smiles = sorted(set(coconut["canonical_smiles"].unique().tolist()))

    return (plants_genera, coconut, laji_gbif, compounds, smiles)



########################################################################################
def analyse(compound: str="arctigenin", genus: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyse a compound and compute results summary.
    """

    print(f"Analysing compound '{compound}' (genus={genus})...")

    # Filter coconut for the compound
    res = coco[coco["name"] == compound].copy()
    if res.empty:
        st.warning(f"No data found for compound '{compound}' in coconut database.")
        return None, None

    # take the first one
    org = res["organisms"].iloc[0].split("|")
    smiles = res["canonical_smiles"].iloc[0]
    
    print("Compound:", compound)
    print("Smiles:",smiles)

    org = sorted(set([e.lower().strip() for e in org if e]))
    # keep only plants
    org = [e for e in org if infer_genus(e) in plants]
    if not org:
        print("WARNING: No organisms found!")
        return None, None
    print("Organisms:", len(org))

    if genus:
        # use genus to make the search wider
        genera = [infer_genus(e) for e in org]
        genera = set([e for e in genera if e])
        if not org:
            print("WARNING: No organisms found!")
            return None, None
        genera = pd.DataFrame({"genus":sorted(genera)})
        genera = pd.merge(genera,db,how="left",left_on="genus",right_on="genus")
        genera = genera.dropna(subset=["name"])
        org = sorted(set(genera["name"].tolist()))
        


    res = pd.DataFrame({"organism":org})
    
    print("Processing Laji & GBIF database information...")
   
    res = pd.merge(res,db,how="left",left_on="organism",right_on="name")
    res = res.drop(columns=["name"])
    res = res.dropna(subset=["genus"])

    cols = ["obs. in Finland (laji)", 
            "obs. in Finland (gbif)", 
            "obs. in Norway (gbif)",
            "obs. in Finland >60N(gbif)", 
            "obs. in Norway >60N (gbif)",
            "obs. in Finland >66N (gbif)", 
            "obs. in Norway >66N (gbif)"]
    for c in cols:
        res[c] = res[c].fillna(0).astype("Int64")
            
    res = res.sort_values(by=["obs. in Finland (laji)", "obs. in Finland (gbif)","obs. in Norway (gbif)"], ascending=[False, False,False])

    results = res.copy()

    # Compute counts and percentages
    counts = res["genus"].value_counts(normalize=False, dropna=False)
    percents = res["genus"].value_counts(normalize=True, dropna=False) * 100
    summary = (
        counts.rename("count")
        .reset_index()
        .rename(columns={"index": "genus"})
        .assign(percent=percents.values.round(2))
    )
    summary = summary.sort_values(by=["count", "genus"], ascending=[False, True])

    return (results, summary)


# Load data once at startup
(plants, coco, db, compounds, smiles) = load_data()


########################################################################################
# --- Helpers ---
def add_row_id(df: pd.DataFrame, colname: str = "#") -> pd.DataFrame:
    df = df.copy()
    df.insert(0, colname, range(1, len(df) + 1))
    return df

def add_row_id_offset(df: pd.DataFrame, start: int, colname: str = "#") -> pd.DataFrame:
    df = df.copy()
    df.insert(0, colname, range(start + 1, start + 1 + len(df)))
    return df

def paginate_df(df: pd.DataFrame, page_size: int = RESULTS_PAGE_SIZE):
    total = len(df)
    pages = max(1, math.ceil(total / page_size))
    left, right = st.columns([1, 3])
    with left:
        page = st.number_input("Page", min_value=1, max_value=pages, value=1, step=1)
    with right:
        st.caption(f"{total} rows • {pages} pages • {page_size} rows/page")
    start = (page - 1) * page_size
    end = start + page_size
    return df.iloc[start:end].copy(), start, total

# --- Renderers ---
def render_results_table(df: pd.DataFrame):
    """Render Results with organism names as clickable links (no header sorting), 30 rows/page."""
    if "organism" not in df.columns:
        st.warning("Expected column 'organism' is missing from results")
        return
    df = df.copy()

    # Hide 'genus' and build clickable organism from 'url' if available
    if "genus" in df.columns:
        df = df.drop(columns=["genus"])
    if "url" in df.columns:
        def mk_link(row):
            name = str(row.get("organism", ""))
            href = str(row.get("url", ""))
            if href and href.lower().startswith(("http://", "https://")):
                return f'<a href="{href}" target="_blank" rel="noopener noreferrer">{name}</a>'
            return name
        df["organism"] = df.apply(mk_link, axis=1)
        df = df.drop(columns=["url"])

    # Paginate BEFORE rendering (keeps HTML links fast)
    page_df, start, total = paginate_df(df, page_size=RESULTS_PAGE_SIZE)

    # Add row id with offset
    page_df = add_row_id_offset(page_df, start)

    # Render as HTML to preserve links (no sorting)
    st.markdown(page_df.to_html(escape=False, index=False), unsafe_allow_html=True)
    st.caption(f"Showing rows {start+1}–{start+len(page_df)} of {total}")


# --- Controls ---
c1, c2, c3 = st.columns([3, 2, 1])
with c1:
    compound = st.selectbox(
        "Compound",
        compounds,
        index=(compounds.index("arctigenin") if "arctigenin" in compounds else 0),
        help="Choose a compound from the coconut database",
    )
with c2:
    association = st.radio(
        "Association",
        ["species", "genus"],
        index=0,
        horizontal=True,
        help="Choose the aggregation level (for display/reporting).",
    )
with c3:
    run_btn = st.button("Find", type="primary")

# --- Action ---
if run_btn:
    if association == "genus":
        results, summary = analyse(compound, genus=True)
    else:
        results, summary = analyse(compound, genus=False)

    if results is not None and summary is not None:
        st.divider()
        
        # Display results
        st.subheader("Results")
        render_results_table(results)
        
        # Display summary
        st.subheader("Summary per genus")
        s_show = add_row_id(summary)
        st.dataframe(
            s_show,
            width="content",
            hide_index=True,
            column_config={
                "#": st.column_config.NumberColumn("#", width="small", disabled=True)
            },
        )
        st.caption(f"{len(s_show)} rows")
    else:
        st.warning("No plant organisms found for this compound!")

st.caption("Tip: Start typing in the selector to filter a long compound list.")
st.caption("**Data sources:** [COCONUT](https://coconut.naturalproducts.net/) (Collection of Open Natural Products database), [Laji.fi](https://laji.fi/) (Finnish Biodiversity Information Facility) and [GBIF](https://www.gbif.org/) (Global Biodiversity Information Facility).")


