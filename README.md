# Sentinel2 Crop Trait Retrieval

This repository contains the Python code and data required to re-run the analysis and results presented in

> **_PAPER:_**  Graf, L.V, Merz, Q.M, Walter, A., Aasen, H. (2023) "Insights from field phenotyping improve satellite remote sensing based in-season estimation of winter wheat growth and phenology". *Under review*".

The data used in the study and made available in this repository is the result of multi-year, labor-intense work in the field and laboratory conducted by teams at [ETH Zurich Crop Science](https://kp.ethz.ch/), the [School of Agircultural, Forest and Food Sciences, HAFL](https://www.bfh.ch/hafl/en/) and the [Division of Agroecology and Environment at Agroscope Reckenholz](https://www.agroscope.admin.ch/agroscope/en/home/about-us/organization/competence-divisions-strategic-research-divisions/agroecology-environment.html). A [list of contributors](data/AUTHORS.txt) is provided.

We therefore kindly ask you to **acknowledge our work** by

* **citing** our research properly whenever you use the data and/or methods presented here
* leave a **star on GitHub** and/or fork our repository

This helps us to continue the labor and cost-intensive process of data acquisition, preparation and, ultimately, publication to benefit science and society.

If your work relies substantially on our data please also [get in touch with us](https://www.eoa-team.net/) and consider offering co-authorship.

## Content


### Code
The Python source code can be found in [src](src). It extracts the Sentinel-2 data, runs the PROSAIL simulations, performs the inversion for trait retrieval, implements the phenology model and generates the figures shown in the paper.
For re-running the entire workflow (all results are provided, see below) you have the execute the Python scripts listed in the order below:

* [extract_s2_scenes.py](src/extract_s2_scenes.py): Extracts the Sentinel-2 surface reflectance data and runs PROSAIL in forward mode for the field parcel geometries provided
* [invert_s2_scenes.py](src/invert_s2_scenes.py): Carries out the inversion of PROSAIL to derive the functional traits
* [combine_models.py](src/combine_models.py): Implements the phenological model and combines PROSAIL outputs
* [validate_traits.py](src/validate_traits.py): Carries out the trait validation against in-situ data

Moreover, all but two figures (the overview map that was created in QGIS, and the workflow figure that was handcrafted in LibreOffice) can be recreated using [these Python scripts](src/figures_paper).

### Data
In [data](data) the in-situ trait values, field parcel geometries, location of sampling points where the traits were measured, field calendars and spectral response functions of Sentinel-2 can be found.

Please given **proper credit** of our work. See [our guidelines](data/README.md) and [list of contributors](data/AUTHORS.txt) for more details.

### Results
In [results](results) we deliver the extracted Sentinel-2 data and results of the PROSAIL inversion (including lookup-tables). This is mainly due to computational demands of running PROSAIL in forward mode to allow users with limited computing resources to check our methodology and reproduce our main findings.

In results, also all [figures](results/Figures) of the paper can be found.
 
## OS and Software Requirements

The code was tested and run completely on Fedora 35 using Python 3.10. In theory, it should also run on other operating systems and Python versions but we never verified it.

All requirements to execute the scripts can be found in the [requirements.txt](requirements.txt) file. To install we recommend to create a clean Python virtual environment. The steps below show how to do the installation using `git`, `pip` and `venv` on Linux:

```bash
git clone https://github.com/EOA-team/sentinel2_crop_traits.git
cd sentinel2_crop_traits
python -m venv my_venv
source my_venv/bin/activate
pip install -r requirements.txt
```
