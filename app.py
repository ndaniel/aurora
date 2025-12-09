#!/usr/bin/env python3
"""
Author: Daniel.Nicorici@gmail.com

Extract info about compounds from different plant datasets (coconut, laji, gbif, etc.).
"""

import os
import sys
import math
import io
from html import escape
from typing import Tuple
from urllib.parse import quote_plus, unquote_plus
from collections.abc import Mapping

import pandas as pd
import streamlit as st

import logging

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    from rdkit.Chem import rdFingerprintGenerator as rfg
    RDKit_AVAILABLE = True
except Exception:  # pragma: no cover - RDKit required for similarity features
    Chem = None
    DataStructs = None
    AllChem = None
    rfg = None
    RDKit_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("aurora")
log.info("App started")


DEFAULT_COMPOUND = "arctigenin"

DATA_DIR = "data"

# LIST_COMPOUNDS_PATH = os.path.join(DATA_DIR, "coconut_compounds.txt")
# LIST_PLANTS_FI_NO_PATH = os.path.join(DATA_DIR, "organisms_laji_gbif_FI_NO_plants.tsv")
#COCONUT_DB_PATH = os.path.join(DATA_DIR, "coconut_csv-09-2025_FI_NO_plants.csv")
COCONUT_DB_PATH = os.path.join(DATA_DIR, "coconut_csv-09-2025_FI_NO_selected.parquet")
LAJI_DB_PATH = os.path.join(DATA_DIR, "laji2_fi.txt")
GBIF_DB_PATH = os.path.join(DATA_DIR, "gbif_selected_FI_NO_merged.tsv")
LIST_PLANTS_GENERA_PATH = os.path.join(DATA_DIR, "selected_genera.txt")

# Hard-coded app password; change the value to update access
APP_PASSWORD = "metformin"

st.set_page_config(page_title="AURORA Pilot", layout="wide")

# ---- Session state init (so tables persist after any rerun, e.g., download clicks) ----
for k, v in {
    "results": None,
    "summary": None,
    "last_compound": None,
    "last_association": "species",
    "last_smiles": None,
    "suggestions": [],
    "compound_input": DEFAULT_COMPOUND,
    "auto_run": False,
    "searched_via_smiles": False,
    "similarity_table": None,
    "active_smiles_query": None,
    "similarity_error": None,
    "auth_ok": False,
    "auth_error": False,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


def require_password():
    """Optional password gate that does not disturb the rest of the UI."""
    if not APP_PASSWORD:
        return
    if st.session_state.get("auth_ok"):
        return

    if st.session_state.get("auth_error"):
        st.error("Incorrect password, please try again.")

    with st.form("auth_form"):
        st.subheader("Enter password to continue")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Unlock")

    if submitted:
        if password == APP_PASSWORD:
            st.session_state["auth_ok"] = True
            st.session_state["auth_error"] = False
            st.rerun()
        else:
            st.session_state["auth_error"] = True

    if not st.session_state.get("auth_ok"):
        st.stop()

QUERY_PARAM_KEY = "select_compound"
query_params: dict[str, list[str] | str] = {}
raw_query_params = getattr(st, "query_params", None)
if raw_query_params is not None:
    if isinstance(raw_query_params, Mapping):
        query_params = dict(raw_query_params)
    else:
        to_dict = getattr(raw_query_params, "to_dict", None)
        if callable(to_dict):
            query_params = to_dict()
else:
    experimental_get = getattr(st, "experimental_get_query_params", None)
    if callable(experimental_get):
        query_params = experimental_get()
selected_values = query_params.get(QUERY_PARAM_KEY, [])
if not isinstance(selected_values, list):
    selected_values = [selected_values]
if selected_values:
    selected_name = unquote_plus(selected_values[0])
    if selected_name:
        st.session_state["pending_compound"] = selected_name
        st.session_state["pending_association"] = st.session_state.get("association_sel", "species")
        st.session_state["auto_run"] = True
    # Clear the query param so we don't loop on rerun
    if hasattr(st, "query_params"):
        st.query_params = {}

if st.session_state.get("pending_association"):
    st.session_state["association_sel"] = st.session_state.pop("pending_association")
if st.session_state.get("pending_compound"):
    st.session_state["compound_input"] = st.session_state.pop("pending_compound")
    st.session_state["auto_run"] = True

require_password()

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

SIM_FP_RADIUS = 2
SIM_FP_BITS = 2048
SIMILARITY_MAX_ROWS = 500

def _make_morgan_count_gen(radius: int = 2):
    if not RDKit_AVAILABLE or rfg is None:
        return None
    try:
        return rfg.GetMorganGenerator(radius=radius)
    except Exception:
        return None

SIM_COUNT_GEN = _make_morgan_count_gen(SIM_FP_RADIUS)


def _make_morgan_bit_gen(radius: int = 2, nbits: int = 2048):
    if not RDKit_AVAILABLE or rfg is None:
        return None
    try:
        return rfg.GetMorganGenerator(
            radius=radius,
            includeChirality=True,
            useBondTypes=True,
            fpSize=nbits,
        )
    except TypeError:
        try:
            return rfg.GetMorganGenerator(radius, False, True, True, False, True, None, nbits)
        except Exception:
            return None


SIM_BIT_GEN = _make_morgan_bit_gen(SIM_FP_RADIUS, SIM_FP_BITS)

st.title("AURORA Pilot (compounds in selected Nordic species)")
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

def is_smiles(smiles_string: str) -> bool:
    """
    Checks very fast if a given string is a valid SMILES string without using regular expressions.
    """
    if not isinstance(smiles_string, str) or not smiles_string:
        return False

    smiles_string = smiles_string.strip()
    if not smiles_string:
        return False

    allowed = {'@', '[', 'i', 'N', '/', '+', '6', 'F', '3', 'H', 'r', 'u', ']', 'L', 'C', ')', 's', '0', 'K', 'P', 'O', '.', '=', '9', 'a', 'e', '4', '2', '5', '%', '1', '#', '7', 'g', '-', '(', '8', 'b', 'S', 'o', 'M', 'I', 'A', 'l', 'B', '\\'}
    s = set(smiles_string)
    if s.difference(allowed):
        return False

    return True

########################################################################################
@st.cache_data(show_spinner=True, ttl=3600)
def load_data():

    required = {
        "COCONUT_DB_PATH": COCONUT_DB_PATH,
        "LAJI_DB_PATH": LAJI_DB_PATH,
        "GBIF_DB_PATH": GBIF_DB_PATH,
        "LIST_PLANTS_GENERA_PATH": LIST_PLANTS_GENERA_PATH,
    }
    missing = []
    for label, p in required.items():
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            missing.append(f"{label} → {p}")

    if missing:
        st.error("Required data files are missing or empty:\n- " + "\n- ".join(missing))
        st.stop()

    log.info("Processing short plant list of plants genera...")

    plants_genera = set()
    with open(LIST_PLANTS_GENERA_PATH, "r") as f:
        plants_genera = set([e.lower().rstrip("\r\n") for e in f if e.rstrip("\r\n")])

    log.info("Processing Laji.fi database information...")
    laji = pd.read_csv(LAJI_DB_PATH, sep="\t", low_memory=False)
    laji["name"] = laji["Scientific name"].str.lower()
    laji = laji[["name", "Identifier", "Observation count from Finland", "Genus, Scientific name"]].copy()
    laji.columns = ["name", "identifier_laji", "obs. in Finland (laji)", "genus_laji"]
    laji["genus_laji"] = laji["genus_laji"].str.lower()
    laji = laji.dropna(subset=["name"]).drop_duplicates()

    log.info("Processing GBIF database information...")
    gbif = pd.read_csv(GBIF_DB_PATH, sep="\t", low_memory=False)
    gbif = gbif[
        ["canonicalName", "genus", "obs_FI", "obs_NO", "count_FI_60N", "count_NO_60N", "count_FI_66N", "count_NO_66N", "genusKey", "speciesKey"]
    ].copy()
    gbif["canonicalName"] = gbif["canonicalName"].str.lower()
    gbif["genus"] = gbif["genus"].str.lower()
    gbif = gbif.drop_duplicates()
    gbif.columns = [
        "name",
        "genus_gbif",
        "obs. in Finland (gbif)",
        "obs. in Norway (gbif)",
        "obs. in Finland >60N(gbif)",
        "obs. in Norway >60N (gbif)",
        "obs. in Finland >66N (gbif)",
        "obs. in Norway >66N (gbif)",
        "genusKey_gbif",
        "speciesKey_gbif",
    ]

    laji_gbif = pd.merge(laji, gbif, how="left", left_on="name", right_on="name").drop_duplicates()

    # build the genus
    laji_gbif["genus_guess"] = laji_gbif["name"].apply(infer_genus)
    laji_gbif["genus"] = laji_gbif["genus_gbif"].fillna(laji_gbif["genus_laji"])
    laji_gbif["genus"] = laji_gbif["genus"].fillna(laji_gbif["genus_guess"])
    laji_gbif = laji_gbif.drop(columns=["genus_guess", "genus_gbif", "genus_laji"]).drop_duplicates()

    cols = ["obs. in Finland (gbif)", "obs. in Norway (gbif)", "obs. in Finland (laji)"]
    for c in cols:
        laji_gbif[c] = laji_gbif[c].fillna(0)

    laji_gbif["obs"] = laji_gbif[cols].sum(axis=1)
    laji_gbif = laji_gbif[laji_gbif["obs"] > 2]  # keep only species with more than 2 observations in Finland or Norway
    laji_gbif = laji_gbif.drop(columns=["obs"]).drop_duplicates()
    laji_gbif = laji_gbif[laji_gbif["genus"].notna()]
    laji_gbif["url_laji"] = "https://laji.fi/taxon/" + laji_gbif["identifier_laji"] + "/occurrence"
    laji_gbif["url_gbif"] = laji_gbif["speciesKey_gbif"].apply(lambda x: f"https://www.gbif.org/species/{x}" if pd.notna(x) else pd.NA)
    laji_gbif["url"] = laji_gbif["url_laji"].fillna(laji_gbif["url_gbif"])
    laji_gbif = laji_gbif.drop(columns=["identifier_laji", "genusKey_gbif", "speciesKey_gbif", "url_laji", "url_gbif"])

    log.info("Processing Coconut database information...")
    #coconut = pd.read_csv(COCONUT_DB_PATH, sep="\t", low_memory=False)
    coconut = pd.read_parquet(COCONUT_DB_PATH)
    coconut = coconut.dropna(subset=["name", "identifier"])
    coconut = coconut.drop(columns=["identifier"])
    coconut["name"] = coconut["name"].str.lower().str.strip()
    coconut["organisms"] = coconut["organisms"].str.lower().str.strip()
    coconut["genus_coconut"] = coconut["organisms"].apply(batch_infer_genus)
    coconut = coconut.drop_duplicates()

    compounds = sorted(set(coconut["name"].unique().tolist()))
    smiles = sorted(set(coconut["canonical_smiles"].unique().tolist()))

    log.info("Processing databases finished.")
    return (plants_genera, coconut, laji_gbif, compounds, smiles)

########################################################################################
def analyse(compound: str = "arctigenin", smile: str = "",genus: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, str, str]:
    """
    Analyse a compound and compute results summary.
    """
    # Handle empty input gracefully
    if not compound and not smile:
        #st.warning("Please enter a compound name or SMILES string.")
        return None, None, None, None

    # Determine search mode and get initial data
    log.info("Analyse -> %s",compound)
    log.info("Analyse -> %s",smile)
    log.info("Analyse -> %s",genus)
    smiles_value = None
    flag = False
    if smile:
        log.info(f"Analysing SMILES '{smile}' (genus={genus})...")
        # Filter coconut for the SMILES
        res = coco[coco["canonical_smiles"] == smile].copy()
        if res.empty:
             #st.warning(f"No data found for SMILES '{smile}' in coconut database.")
             #return None, None, None, None
             # maybe is a compound name?
            res = coco[coco["name"] == smile.lower()].copy()
            if res.empty:
                #st.warning(f"No data found for compound '{smile}' in coconut database.")
                return None, None, None, None
            compound = smile.lower()
        else:
            # Take the first result
            org = res["organisms"].iloc[0].split("|")
            compound = res["name"].iloc[0] # Set compound name from SMILES lookup
            smiles_value = res["canonical_smiles"].iloc[0]
            flag = True
    
    if compound and not flag:
        log.info(f"Analysing compound '{compound}' (genus={genus})...")
        # Filter coconut for the compound
        res = coco[coco["name"] == compound.lower()].copy()
        if res.empty:
            #st.warning(f"No data found for compound '{compound}' in coconut database.")
            return None, None, None, None
        # take the first one
        org = res["organisms"].iloc[0].split("|")
        compound = res["name"].iloc[0]
        smiles_value = res["canonical_smiles"].iloc[0]

    log.info("Found -> Compound: %s", compound)
    log.info("Found -> SMILES: %s", smiles_value)
    log.info("Found -> ORG: %s", "|".join(org))

    org = sorted(set([e.lower().strip() for e in org if e]))
    # keep only plants
    org = [e for e in org if infer_genus(e) in plants]
    if not org:
        log.info("WARNING: No organisms found!")
        return pd.DataFrame(), pd.DataFrame(), compound, smiles_value # Return empty DFs but the found compound name
    log.info("Organisms: %d", len(org))

    if genus:
        # use genus to make the search wider
        genera = [infer_genus(e) for e in org]
        genera = set([e for e in genera if e])
        if not genera:
            log.info("WARNING: No genera found for organisms!")
            return pd.DataFrame(), pd.DataFrame(), compound, smiles_value # Return empty DFs
        genera = pd.DataFrame({"genus": sorted(genera)})
        genera = pd.merge(genera, db, how="left", left_on="genus", right_on="genus")
        genera = genera.dropna(subset=["name"])
        #org = sorted(set(genera["name"].tolist()))
        org = [" ".join(e.split(" ")[:2]) for e in genera["name"].tolist() if e]
        org = sorted(set([e.replace("-ryhmä","").replace("×","") for e in org]))

    res = pd.DataFrame({"organism": org})

    log.info("Processing Laji & GBIF database information...")
    res = pd.merge(res, db, how="left", left_on="organism", right_on="name")
    res = res.drop(columns=["name"])
    res = res.dropna(subset=["genus"])

    cols = [
        "obs. in Finland (laji)",
        "obs. in Finland (gbif)",
        "obs. in Norway (gbif)",
        "obs. in Finland >60N(gbif)",
        "obs. in Norway >60N (gbif)",
        "obs. in Finland >66N (gbif)",
        "obs. in Norway >66N (gbif)",
    ]
    for c in cols:
        res[c] = res[c].fillna(0).astype("Int64")

    res = res.sort_values(by=["obs. in Finland (laji)", "obs. in Finland (gbif)", "obs. in Norway (gbif)"], ascending=[False, False, False])

    results = res.copy()
    # actually here are only plants
    results.rename(columns={"organism": "plant"}, inplace=True)

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

    return (results, summary, compound, smiles_value)

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

def paginate_df(df: pd.DataFrame, page_size: int = RESULTS_PAGE_SIZE, widget_key: str | None = None):
    total = len(df)
    pages = max(1, math.ceil(total / page_size))
    left, right = st.columns([1, 3])
    with left:
        number_input_kwargs = {
            "min_value": 1,
            "max_value": pages,
            "value": 1,
            "step": 1,
        }
        if widget_key:
            number_input_kwargs["key"] = f"{widget_key}_page"
        page = st.number_input("Page", **number_input_kwargs)
    with right:
        st.caption(f"{total} rows • {pages} pages • {page_size} rows/page")
    start = (page - 1) * page_size
    end = start + page_size
    return df.iloc[start:end].copy(), start, total

def df_to_xlsx_bytes(df: pd.DataFrame, sheet_name: str = "Sheet1") -> bytes:
    """Return an .xlsx file (as bytes) for the given DataFrame."""
    try:
        import openpyxl  # ensure dependency exists at runtime
    except Exception as e:
        st.warning("Excel export not available (openpyxl missing).")
        log.warning("openpyxl import failed: %s", e)
        return b""
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    bio.seek(0)
    return bio.getvalue()

def header_with_download(title: str, df: pd.DataFrame, filename: str):
    """Render a subheader with a right-aligned download button for df."""
    left, right = st.columns([6, 2])
    with left:
        st.subheader(title)
    with right:
        st.download_button(
            label="⬇️ Download as Excel file",
            data=df_to_xlsx_bytes(df),
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

# --- Renderers ---
def render_results_table(df: pd.DataFrame):
    """Render Results with organism names as clickable links (no header sorting), 30 rows/page."""
    if "plant" not in df.columns:
        st.warning("Expected column 'plant' is missing from results")
        return
    df = df.copy()

    # Hide 'genus' and build clickable organism from 'url' if available
    if "genus" in df.columns:
        df = df.drop(columns=["genus"])
    if "url" in df.columns:
        def mk_link(row):
            name = str(row.get("plant", ""))
            href = str(row.get("url", ""))
            if href and href.lower().startswith(("http://", "https://")):
                return f'<a href="{href}" target="_blank" rel="noopener noreferrer">{name}</a>'
            return name
        df["plant"] = df.apply(mk_link, axis=1)
        df = df.drop(columns=["url"])

    # Paginate BEFORE rendering (keeps HTML links fast)
    page_df, start, total = paginate_df(df, page_size=RESULTS_PAGE_SIZE, widget_key="results")

    # Add row id with offset
    page_df = add_row_id_offset(page_df, start)

    # Render as HTML to preserve links (no sorting)
    st.markdown(
        """
        <style>
        table.similarity-table td:nth-child(2),
        table.similarity-table th:nth-child(2) {
            min-width: 600px;
            width: 600px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    table_html = page_df.to_html(escape=False, index=False, classes="similarity-table")
    st.markdown(table_html, unsafe_allow_html=True)
    st.caption(f"Showing rows {start+1}–{start+len(page_df)} of {total}")


def _mol_from_smiles_safe(smiles: str):
    if not RDKit_AVAILABLE or Chem is None:
        return None
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        return Chem.MolFromSmiles(smiles, sanitize=True)
    except Exception:
        return None


def _fingerprint_from_mol(mol):
    if not RDKit_AVAILABLE or mol is None:
        return None
    if SIM_COUNT_GEN is not None:
        try:
            return SIM_COUNT_GEN.GetCountFingerprint(mol)
        except AttributeError:
            try:
                return SIM_COUNT_GEN.GetFingerprint(mol, countSimulation=True)
            except Exception:
                pass
    if AllChem is not None:
        try:
            return AllChem.GetMorganFingerprint(mol, radius=SIM_FP_RADIUS)
        except Exception:
            return None
    return None


def compute_tanimoto_scores(reference_smiles: str, smiles_list: list[str], fp_radius: int = SIM_FP_RADIUS, nbits: int = SIM_FP_BITS):
    if not RDKit_AVAILABLE or DataStructs is None or AllChem is None:
        return [None] * len(smiles_list)
    ref_smiles = reference_smiles.strip() if isinstance(reference_smiles, str) else reference_smiles
    ref_mol = _mol_from_smiles_safe(ref_smiles)
    if ref_mol is None:
        return [None] * len(smiles_list)
    ref_fp = None
    if SIM_BIT_GEN is not None:
        try:
            ref_fp = SIM_BIT_GEN.GetFingerprint(ref_mol)
        except Exception:
            ref_fp = None
    if ref_fp is None:
        try:
            ref_fp = AllChem.GetMorganFingerprintAsBitVect(ref_mol, radius=fp_radius, nBits=nbits)
        except Exception:
            return [None] * len(smiles_list)

    scores = []
    for smi in smiles_list:
        mol = _mol_from_smiles_safe(smi)
        if mol is None:
            scores.append(None)
            continue
        try:
            if SIM_BIT_GEN is not None:
                fp = SIM_BIT_GEN.GetFingerprint(mol)
            else:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=fp_radius, nBits=nbits)
            sim = DataStructs.TanimotoSimilarity(fp, ref_fp)
            scores.append(round(float(sim), 3))
        except Exception:
            scores.append(None)
    return scores


def compute_tversky_dice_scores(reference_smiles: str, smiles_list: list[str], alpha: float = 0.7, beta: float = 0.3):
    if not RDKit_AVAILABLE or DataStructs is None:
        return [None] * len(smiles_list), [None] * len(smiles_list)
    ref_mol = _mol_from_smiles_safe(reference_smiles)
    ref_fp = _fingerprint_from_mol(ref_mol)
    if ref_fp is None:
        return [None] * len(smiles_list), [None] * len(smiles_list)

    tversky_scores = []
    dice_scores = []

    def _round_score(val):
        try:
            f_val = float(val)
        except (TypeError, ValueError):
            return None
        if math.isnan(f_val):
            return None
        return round(f_val, 3)

    for smi in smiles_list:
        mol = _mol_from_smiles_safe(smi)
        fp = _fingerprint_from_mol(mol)
        if fp is None:
            tversky_scores.append(None)
            dice_scores.append(None)
            continue
        try:
            t_val = DataStructs.TverskySimilarity(fp, ref_fp, a=alpha, b=beta)
        except Exception:
            t_val = None
        try:
            d_val = DataStructs.DiceSimilarity(fp, ref_fp)
        except Exception:
            d_val = None
        tversky_scores.append(_round_score(t_val))
        dice_scores.append(_round_score(d_val))
    return tversky_scores, dice_scores


def build_similarity_table(query_smiles: str) -> pd.DataFrame:
    if not (RDKit_AVAILABLE and AllChem is not None and coco is not None):
        log.warning("Similarity disabled: RDKit=%s", RDKit_AVAILABLE)
        return pd.DataFrame()
    if not isinstance(query_smiles, str) or not query_smiles.strip():
        return pd.DataFrame()
    query_smiles = query_smiles.strip()

    df = coco[["name", "canonical_smiles"]].dropna().copy()
    df["canonical_smiles"] = df["canonical_smiles"].astype(str).str.strip()
    df = df[df["canonical_smiles"] != ""]
    df = df.drop_duplicates(subset=["name", "canonical_smiles"]).reset_index(drop=True)
    log.info("Similarity base set size: %d", len(df))
    if df.empty:
        return pd.DataFrame()

    smiles_list = df["canonical_smiles"].tolist()
    tanimoto_scores = compute_tanimoto_scores(query_smiles, smiles_list)
    log.info("Computed tanimoto scores: %d", sum(1 for x in tanimoto_scores if x is not None))
    tversky_scores, dice_scores = compute_tversky_dice_scores(query_smiles, smiles_list)
    log.info("Computed tversky scores: %d", sum(1 for x in tversky_scores if x is not None))

    df = df.assign(
        tanimoto=tanimoto_scores,
        tversky=tversky_scores,
        dice=dice_scores,
    )
    df = df.dropna(subset=["tanimoto"])
    log.info("Similarity rows after dropna: %d", len(df))
    if df.empty:
        return pd.DataFrame()

    df = df.rename(columns={"name": "compound", "canonical_smiles": "smiles"})
    df = df.sort_values(by="tanimoto", ascending=False).head(SIMILARITY_MAX_ROWS).reset_index(drop=True)
    return df


def render_similarity_table(df: pd.DataFrame):
    if df is None or df.empty:
        st.info("No similarity scores available for this SMILES input.")
        return

    display_df = df.copy()
    display_df = display_df[["compound", "tanimoto", "tversky", "dice"]]
    display_df.columns = ["Compound", "Tanimoto", "Tversky", "Dice"]

    for col in ["Tanimoto", "Tversky", "Dice"]:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else "")

    page_df, start, total = paginate_df(display_df, page_size=RESULTS_PAGE_SIZE, widget_key="similarity")
    page_df = add_row_id_offset(page_df, start)

    def _compound_link(name: str) -> str:
        safe_name = escape(name)
        return f'<a href="?{QUERY_PARAM_KEY}={quote_plus(name)}">{safe_name}</a>'

    page_df["Compound"] = page_df["Compound"].apply(_compound_link)

    st.markdown(page_df.to_html(escape=False, index=False), unsafe_allow_html=True)
    st.caption(f"Showing rows {start+1}–{start+len(page_df)} of {total}")


def display_similarity_section(similarity_df: pd.DataFrame, compound_name: str | None, show_compound_heading: bool):
    if similarity_df is None or similarity_df.empty:
        return

    download_similarity = similarity_df[["compound", "tanimoto", "tversky", "dice"]].copy()
    for col in ["tanimoto", "tversky", "dice"]:
        download_similarity[col] = download_similarity[col].apply(lambda x: round(x, 3) if isinstance(x, (int, float)) else x)
    download_similarity = download_similarity.rename(
        columns={
            "compound": "Compound",
            "tanimoto": "Tanimoto",
            "tversky": "Tversky",
            "dice": "Dice",
        }
    )

    filename_base = compound_name if compound_name else "smiles_search"
    safe_base = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in filename_base) or "smiles_search"
    safe_base = safe_base[:80]
    header_with_download(
        "SMILES similarity",
        download_similarity,
        filename=f"similarity_{safe_base}.xlsx",
    )
    render_similarity_table(similarity_df)

# --- Controls (with keys so values persist naturally) ---
c1, c2, c3 = st.columns([3, 1, 3], vertical_alignment="bottom")
with c1:
    # Text input allows free-form entry for names or SMILES strings
    compound = st.text_input(
        "Compound",
        value=st.session_state.get("compound_input", DEFAULT_COMPOUND),
        key="compound_input",
        help="Enter a compound name.",
    )
with c2:
    run_btn = st.button("Find", type="primary", use_container_width=True)
with c3:
    association = st.radio(
        "Association",
        ["species", "genus"],
        index=(0 if st.session_state["last_association"] == "species" else 1),
        key="association_sel",
        horizontal=True,
        help="Choose the aggregation level (for display/reporting).",
    )

# --- Action: compute on button, always render from session state ---
auto_run_trigger = st.session_state.pop("auto_run", False)
search_triggered = False
if run_btn or auto_run_trigger:
    search_triggered = True
    use_genus = association == "genus"
    search_term = st.session_state.compound_input
    input_is_smiles = is_smiles(search_term)
    st.session_state["suggestions"] = []

    normalized_smiles = search_term.strip() if isinstance(search_term, str) else search_term

    if run_btn and input_is_smiles:
        with st.spinner("Computing similarity scores..."):
            similarity_df = build_similarity_table(normalized_smiles)
        if similarity_df is not None and not similarity_df.empty:
            st.session_state["similarity_table"] = similarity_df.copy()
            st.session_state["active_smiles_query"] = normalized_smiles
            st.session_state["similarity_error"] = None
        else:
            st.session_state["similarity_table"] = None
            st.session_state["active_smiles_query"] = None
            st.session_state["similarity_error"] = "No valid similarity scores were computed for the provided SMILES."
    elif run_btn and not input_is_smiles:
        st.session_state["similarity_table"] = None
        st.session_state["active_smiles_query"] = None
        st.session_state["similarity_error"] = None

    found_compound_name = None
    found_smile = None
    results = None
    summary = None

    log.info("Search -> %s", search_term)
    log.info("Search -> %s", input_is_smiles)

    if input_is_smiles:
        # For SMILES inputs we only display similarity scores; clear any previous results
        st.session_state["results"] = None
        st.session_state["summary"] = None
        st.session_state["last_smiles"] = normalized_smiles
    else:
        results, summary, found_compound_name, found_smile = analyse(compound=search_term, smile="", genus=use_genus)
        if results is None:
            query = (search_term or "").strip().lower()
            if query:
                matches = [c for c in compounds if query in c]
                st.session_state["suggestions"] = matches[:30]
            st.session_state["results"] = None
            st.session_state["summary"] = None
            st.session_state["last_compound"] = search_term
            st.session_state["last_association"] = association
            st.session_state["last_smiles"] = None
            st.session_state["active_smiles_query"] = None
            st.session_state["similarity_table"] = None
            st.session_state["similarity_error"] = None
        else:
            st.session_state["results"] = results
            st.session_state["summary"] = summary
            if found_compound_name:
                search_term = found_compound_name
            st.session_state["last_smiles"] = found_smile
            st.session_state["similarity_error"] = None

    st.session_state["last_compound"] = normalized_smiles if input_is_smiles else search_term
    st.session_state["last_association"] = association
    st.session_state["searched_via_smiles"] = bool(st.session_state.get("active_smiles_query"))

# pull from state for rendering (survives reruns like download clicks)
results = st.session_state.get("results")
summary = st.session_state.get("summary")
compound = st.session_state.get("last_compound", DEFAULT_COMPOUND)
association = st.session_state.get("last_association", "species")
suggestions = st.session_state.get("suggestions", [])
compound_smiles = st.session_state.get("last_smiles")
similarity_df = st.session_state.get("similarity_table")
if isinstance(similarity_df, pd.DataFrame) and similarity_df.empty:
    similarity_df = None
active_smiles_query = st.session_state.get("active_smiles_query")
similarity_error = st.session_state.get("similarity_error")

if results is not None and not results.empty:
    st.divider()

    if similarity_df is not None:
        display_similarity_section(similarity_df, compound, show_compound_heading=True)

    if compound_smiles:
        st.subheader("Canonical SMILES")
        st.code(compound_smiles, language="text")

    # ---- Results table + download ----
    results_download = results.copy()
    int_cols = [
        "obs. in Finland (laji)",
        "obs. in Finland (gbif)",
        "obs. in Norway (gbif)",
        "obs. in Finland >60N(gbif)",
        "obs. in Norway >60N (gbif)",
        "obs. in Finland >66N (gbif)",
        "obs. in Norway >66N (gbif)",
    ]
    for c in int_cols:
        if c in results_download.columns:
            results_download[c] = pd.to_numeric(results_download[c], errors="coerce").astype("Int64")

    header_with_download(
        "Results",
        results_download,
        filename=f"results_{compound}_{association}.xlsx",
    )
    render_results_table(results)

    # ---- Summary table + download ----
    s_show = add_row_id(summary)
    header_with_download(
        "Summary per genus",
        summary.copy(),  # download the raw summary (without the added row-id)
        filename=f"summary_per_genus_{compound}_{association}.xlsx",
    )
    st.dataframe(
        s_show,
        width="content",
        hide_index=True,
        column_config={"#": st.column_config.NumberColumn("#", width="small", disabled=True)},
    )
    st.caption(f"{len(s_show)} rows")
elif similarity_df is not None:
    st.divider()
    display_similarity_section(similarity_df, active_smiles_query, show_compound_heading=True)
elif similarity_error:
    st.divider()
    st.info(similarity_error)
elif suggestions:
    st.divider()
    st.info("No exact match found. Choose one of the suggested compounds below:")
    cols = st.columns(3)
    for idx, name in enumerate(suggestions):
        col = cols[idx % len(cols)]
        with col:
            if st.button(name, key=f"suggestion_{idx}"):
                pending_association = st.session_state.get("association_sel", "species")
                st.session_state["pending_compound"] = name
                st.session_state["pending_association"] = pending_association
                st.session_state["suggestions"] = []
                st.session_state["searched_via_smiles"] = False
                st.session_state["similarity_table"] = None
                st.session_state["active_smiles_query"] = None
                st.session_state["similarity_error"] = None
                st.session_state["auto_run"] = True
                st.rerun()
elif search_triggered: # Only show "No results" if a search was just performed
    st.divider()
    st.info("No results found for the given criteria.")
else:
    st.caption("Enter a compound name and click 'Find'.")

st.caption(
    "**Data sources:** [COCONUT](https://coconut.naturalproducts.net/) (Collection of Open Natural Products database), "
    "[Laji.fi](https://laji.fi/) (Finnish Biodiversity Information Facility) and "
    "[GBIF](https://www.gbif.org/) (Global Biodiversity Information Facility)."
)


st.caption('<span style="font-size: x-small;">App version: 3.2;  © 2025 Daniel Nicorici, Juha Klefström — University of Helsinki</span>', unsafe_allow_html=True)
