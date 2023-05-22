# BIDPP

## Pre-requisites
- Python 3.6
- pip
- Git or source code

## Installation
- Clone the repository: `git clone https://github.com/DasithEdirisinghe/BIDPP.git`
- Create a virtual environment: `python3 -m venv venv`
- Activate the virtual environment: `source venv/bin/activate`
- Install the requirements: `pip3 install -r requirements.txt`

## Running the application
- Place raw data files in the `Data/raw_data` folder
- cd into the `src` folder: `cd src`
- Run the application `python3 run.py <training positive file> <training negative file> <testing positive file> <testing negative file>` 
- Ex: `python3 run.py TR_pos_SPIDER.txt TR_neg_SPIDER.txt TS_pos_SPIDER.txt TS_neg_SPIDER.txt`
- The results will be saved in the `src/out` folder
