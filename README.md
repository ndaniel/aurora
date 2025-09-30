# Aurora

[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-ff4b4b.svg)](https://streamlit.io/)

[AURORA](aurorapilot.streamlit.app) (compounds in Nordic plants) is a Streamlit app, that integrates several plants
databases like:
- [COCONUT](https://coconut.naturalproducts.net/) (Collection of Open Natural Products database)
- [Laji.fi](https://laji.fi/) (Finnish Biodiversity Information Facility), and
- [GBIF](https://www.gbif.org/) (Global Biodiversity Information Facility).



## Quickstart (local)

```bash
# create & activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # on Windows use: .venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# run the app
streamlit run aurora/app.py


