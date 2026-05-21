

## Installing Dependencies

The package manager used for this project is `uv`

`uv` package manager is recommended but the dependencies can also be installed using `pip` 
```bash
pip install .
```

### Dependencies Installation using uv

Install the `uv` pacakge manager first, using the following

For Windows
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
For MacOs and Linux
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
or Using pip package manager
```bash
pipx install uv
```
After installing run 
```bash
uv sync
```
to install all the required pacakges for this project

---

## Data Extraction

The branch `data` contains the jupyter notebook and different virtual environment configuration satisfying the python version and dependencies of the `wrds` package. The reason for using a python script instead of wrds website to query and download data is becasuse it is not reliable to download large datasets. 

Run `uv sync` again after checking out to the wrds/data branch to install dependencies required for the data extraction.