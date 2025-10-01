# Aurora Pilot app

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.49-brightgreen)
![CI](https://github.com/ndaniel/aurora/actions/workflows/ci.yml/badge.svg)
![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-lightgrey)


[AURORA](https://aurorapilot.streamlit.app) (compounds in Nordic plants) is a Streamlit app, that integrates several plants
databases like:
- [COCONUT](https://coconut.naturalproducts.net/) (Collection of Open Natural Products database),
- [Laji.fi](https://laji.fi/) (Finnish Biodiversity Information Facility), and
- [GBIF](https://www.gbif.org/) (Global Biodiversity Information Facility).


## Screenshots

### Main search interface
![Search interface](docs/screenshot1.png)

### Results view
![Results table](docs/screenshot2.png)

## Quickstart (local)

```bash
# create & activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # on Windows use: .venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# run the app
streamlit run app.py
```

## Docker

You can also run the app inside Docker:

```bash
docker build -t aurora-app .
docker run -it --rm -p 8501:8501 aurora-app
```

## Data Sources & Attribution

- [COCONUT](https://coconut.naturalproducts.net/) (Collection of Open Natural Products database) - CC0 license
- [Laji.fi](https://laji.fi/) (Finnish Biodiversity Information Facility) - CC-BY license
- [GBIF](https://www.gbif.org/) (Global Biodiversity Information Facility) - CC0/CC-BY/CC-BY-NC licenses (depending on the dataset)

All rights and data terms respected according to source guidelines.

## Roadmap

- [ ] Add ETL scripts for automated retrieval and cleaning of GBIF, Laji.fi, and COCONUT datasets  
- [ ] Provide a small reproducible test dataset for CI and demo purposes  
- [ ] Expand CI workflow (unit tests, schema validation with Pandera)  
- [ ] Optimize data loading and performance for larger datasets  
- [ ] Add more interactive visualizations in the Streamlit app  



## ⚠️ Note

Processed data files are expected under `data/`. Some data 
sources (Laji.fi, COCONUT) may require manual export. ETL scripts will 
be added in a future update.





