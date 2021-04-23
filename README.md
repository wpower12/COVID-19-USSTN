# COVID-19 Spatio-Temporal Geometric Data

## Overview
The scripts in the Data file will generate a pytorch.geometric graph that represents the daily, spatio-temporal network of US counties over (currently) 8 months of 2020. Each node represents a particular county on a particular date. These county-date nodes have spatial links to the counties they are geographically adjacent to, that share the same date. Nodes also have temporal nodes, which represent links from past dates over a fixed horizon. For example, this data set currently uses a temporal horizon of 5. Therefore, each node has a link from (up to) the 5 sequentially prior county-date nodes. 

Each node has as its features 6 values representing the mobility values for that county. These values track a relative measure (to national averages) of the influx and outflux of people to the county, according to 6 different classes based on the goal or activity related to that transit of people. These classes are:

* retail_and_recreation_percent_change_from_baseline
* grocery_and_pharmacy_percent_change_from_baseline
* parks_percent_change_from_baseline
* transit_stations_percent_change_from_baseline
* workplaces_percent_change_from_baseline
* residential_percent_change_from_baseline

Each county-date node also contains a pair of target values, the absolute number of observed COVID-19 cases in the county at that date, and the absolute number of deaths due to COVID-19 within the county at that date. 

This data set covers the dates 04/12/2020 to 12/31/2020, contains 272,448 county-date nodes, with a total of 892,584 adjacency based links and 1,363,635 temporal links (based on a horizon size of 5). 

## Source Data
TODO - List sources for the raw files.

This repo also contains the source data for COVID rates, and county level mobility data. These are large files and makes the repo a little unwieldy on the first pull. However, those should not change much, if at all, and will most likely not need to be synced more than once. 

## Use
The example.py script contains an example of the methods used to create a derived (date and time window specific) graph from the source data files. Additionally, there is an example dataset in the derived data directory. 

## Example Method Applications
File 01 contains a draft implementation of a naieve approximation of the Temporal Skip Connections model used in:

* Kapoor, Amol, Xue Ben, Luyang Liu, Bryan Perozzi, Matt Barnes, Martin Blais, and Shawn O’Banion. 2020. “Examining COVID-19 Forecasting Using Spatio-Temporal Graph Neural Networks.” ArXiv:2007.03113 [Cs], July. http://arxiv.org/abs/2007.03113.
