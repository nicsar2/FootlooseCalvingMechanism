import dataCDRPMSIC4 as dSIC
import dataERA5 as dERA
import dataGHRSST as dSST

import figICESAT
import figWaveMelt
import figICESAT2ATL06

"""
HOW TO USE:
	- Install python version 3.8 or above: See doc for specific OS
	- Install miniconda: See doc for specific OS
	- Install dependencies: conda env create -f environment.yml
	- Add Copernicus API keys for cdsapi package: see "Install the CDS API key" at https://cds.climate.copernicus.eu/api-how-to
	- Add Earthdata Login keys for podaac-data-subscriber package: see step 1 and 2 at https://github.com/podaac/data-subscriber
	- Activate conda environment: conda activate sartore24FigureEnv
	- Run: python3 main.py

This code aims to recreate the figures in 'Sartore et al. (2024)' paper. 
It automatically downloads the necessary data. WARNING, this requires around 4 TB of internet data and 50 GB of disk space. Given the amount of data to download, this can take a long time.

To use it, you need to install the python libraries in the environment.yml file using the following miniconda command: conda env create -f environment.yml
I strongly recommend using conda, as some libraries are much easier to install with conda than with pip. 

To download the ERA5 dataset, the Copernicus API is used. The key must be installed as follows: https://cds.climate.copernicus.eu/api-how-to
To download the GHRSST dataset, you need the Earthdata connection key. The key must be installed using https://github.com/podaac/data-subscriber.

The directory containing the data and the figure can be changed in the params.py file. If you change the directory containing the data, don't forget to move the initial data with it. 
"""

def main():
	print("START DOWNLOADING DATASETS")
	print("Start downloading NSIDC SIC")
	dSIC.download()
	print("\n\n\nStart downloading ERA5")
	dERA.download()
	print("\n\n\nStart downloading GHRSST")
	print("!!!! CAUTION: This downloads 4 TB of data. The size is reduced during the process to 15 GB, but be sure to use an unlimited connection !!!!")
	dSST.download()

	print("\n\n\n\n\nSTART PLOTING PAPER FIGURES")
	print("FIGURE 2")
	figICESAT.TimeSeriesRM()
	print("\nFIGURE 3")
	figWaveMelt.mapsWinterMean()
	print("\nFIGURE 4")
	figICESAT2ATL06.Overimpose()
	print("\nFIGURE 5 and 6 and S3")
	figICESAT2ATL06.RMInfo()
	print("\nFIGURE 7")
	figWaveMelt.combinedClimatology()
	print("\nFIGURE S1")
	figICESAT2ATL06.FrontTypeMap()


if __name__ == '__main__':
	main()