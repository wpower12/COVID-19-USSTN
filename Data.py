import pathlib
import pandas as pd
import torch
from torch_geometric.data import Data

ADJ_FN      = "data/source_data/county_adjacency2010.csv"
MOBILITY_FN = "data/source_data/2020_US_Region_Mobility_Report.csv"
CSV_DIR     = "data/source_data/csse_covid_19_daily_reports"

X_FN       = "data/derived_data/{}/X_features.csv"
Y_FN       = "data/derived_data/{}/Y_targets.csv"
COO_FN     = "data/derived_data/{}/coo_list.csv" 
FD_MAP_FN  = "data/derived_data/{}/fips_date_nid_map.csv"
CF_FN      = "data/derived_data/{}/counties_fips.csv"
TRAIN_M_FN = "data/derived_data/{}/train_mask.csv"
TEST_M_FN  = "data/derived_data/{}/test_mask.csv"

### Data Loading ###############################################################
# The following methods simply read in the relevant 
# derived CSV files and return approriate torch an
# pytorch geo objects. 

# Returns the pytorch geometric Data object representing
# the full Spatio-Temporal network of US Counties. 
def getPyTorchGeoData(label):
	with open(X_FN.format(label), "r") as x_f:
		x_df = pd.read_csv(x_f)

	with open(Y_FN.format(label), "r") as y_f:
		y_df = pd.read_csv(y_f)

	with open(COO_FN.format(label), "r") as coo_f:
		coo_df = pd.read_csv(coo_f, header=None)

	with open(TRAIN_M_FN.format(label), "r") as train_m_f:
		train_m_df = pd.read_csv(train_m_f)

	with open(TEST_M_FN.format(label), "r") as test_m_f:
		test_m_df = pd.read_csv(test_m_f)	

	# idk why I need this hack, Gotta figure it out.
	coo_df = filterOOBs(coo_df, len(x_df)-1)

	# We need to reshape the coo first. this	
	coo_t   = torch.tensor(coo_df.values, dtype=torch.long)
	coo_t   = coo_t.reshape((2, len(coo_df.values)))

	x_t     = torch.tensor(x_df.values, dtype=torch.float)
	y_t     = torch.tensor(y_df.values, dtype=torch.float)
	train_t = torch.tensor(train_m_df.values, dtype=torch.long)
	test_t  = torch.tensor(test_m_df.values,  dtype=torch.long)
	return Data(x=x_t, y=y_t, edge_index=coo_t, train_mask=train_t, test_mask=test_t)


def filterOOBs(df, max_val):
	df = df[df[0] < max_val]
	df = df[df[1] < max_val]
	return df


def countOOBs(df, max_index):
	count = 0

	for row in df.iterrows():
		i, j = row[1]

		if i > max_index:
			count += 1
		if j > max_index:
			count += 1

	print("{} oobs".format(count))

# Returns the map used to find a node index given a 
# key made from the FIPS value of a county, and a 
# 'standard' date string. 
def getFipsDateMap(label):
	with open(FD_MAP_FN.format(label), "r") as fdm_f:
		x_df = pd.read_csv(fdm_f)
	return x_df.to_dict()


# Returns a dataframe containing summary information about
# the US counties, with a column containing the FIPS values
# to enable easier searching. 
def getCountyDF(label):
	with open(CF_FN.format(label)) as cf_f:
		c_df = pd.read_csv(cf_f)
	return c_df


### Data Generation ############################################################
# These methods operate on the source data CSV's to 
# create specfic derived data files given a date range. 
# The created files are all meant to be simple text files, hoping
# to skirt any system specific errors, while making it easy to 
# recreate specific data objects quickly. 

def generateFullDataset(start, 
			end, 
			window_size, 
			train_split, 
			dir_label, 
			target_type='both'):
	days = getDateRange(start, end)
	fips_values       = generateFIPSList()
	fdi_map, ifd_list = generateFIPSDateMaps(fips_values, days)

	fdi_save_fn = FD_MAP_FN.format(dir_label)
	writeMapToCSV(fdi_save_fn, fdi_map, ['fdkey', 'node_id'])
	print("fipsdate -> node index map saved to {}".format(fdi_save_fn))

	mob_features = generateMobilityFeatures(fdi_map)
	print("{} entries in mob_features list".format(len(mob_features)))
	x_fn = X_FN.format(dir_label)
	writeListToCSV(x_fn, mob_features)
	print("X features saved to {}".format(x_fn))

	coo_list     = generateCOOList(days, fdi_map, fips_values, window_size)
	print("{} edges in coo list".format(len(coo_list)))
	coo_fn = COO_FN.format(dir_label)
	writeListToCSV(coo_fn, coo_list)
	print("COO Edge List saved to {}".format(coo_fn))

	target_list  = generateTargetList(days, fdi_map, data=target_type)
	print("{} entries in target_list".format(len(target_list)))
	y_fn = Y_FN.format(dir_label)
	writeListToCSV(y_fn, target_list)
	print("Y targets saved to {}".format(y_fn))

	train_m, test_m = generateTrainTestMasks(days, train_split, fdi_map, fips_values)
	print("{} entries in train/test masks".format(len(train_m)))
	train_fn = TRAIN_M_FN.format(dir_label)
	writeListToCSV(train_fn, train_m)
	test_fn  = TEST_M_FN.format(dir_label)
	writeListToCSV(test_fn, test_m)
	print("train and test masks saved")

def getDateRange(start, end):
	START_DATE  = pd.to_datetime(start)
	END_DATE    = pd.to_datetime(end)
	return pd.date_range(start=START_DATE, end=END_DATE, freq='D')


def generateFIPSList():
	DTYPES = {'fipscounty': "str", 'fipsneighbor': 'str'}
	adj_df = pd.read_csv(ADJ_FN, dtype=DTYPES)
	u_fips = set()
	for row in adj_df.iterrows():
		raw = row[1]
		u_fips = u_fips.union([raw['fipscounty'], raw['fipsneighbor']])
	return u_fips


def generateFIPSDateMaps(fips_list, date_range):
	fdi_dict = dict() # The FIPS-Date -> idx Dictionary (fdi)
	ifd_list = []     # Index for dict, so idx -> FIPS-Date Key
	curr_idx = 0

	for fips in fips_list:
		for day in date_range:
			key_str = "{}-{}".format(fips, day.date())

			fdi_dict[key_str] = curr_idx
			ifd_list.append(key_str)
			curr_idx += 1

	return fdi_dict, ifd_list


def generateMobilityFeatures(fdi_map):
	target_cols = ["census_fips_code",
		"date",
		"retail_and_recreation_percent_change_from_baseline",
		"grocery_and_pharmacy_percent_change_from_baseline",
		"parks_percent_change_from_baseline",
		"transit_stations_percent_change_from_baseline",
		"workplaces_percent_change_from_baseline",
		"residential_percent_change_from_baseline"]

	DTYPES = {'census_fips_code': 'str',
		"date": 'str'}

	mobility_df = pd.read_csv(MOBILITY_FN, dtype=DTYPES)
	mobility_df = mobility_df[target_cols]
	m_df = mobility_df[mobility_df['census_fips_code'].notna()].copy()
	m_df.fillna(0, inplace=True)

	x = [[0.0 for o in range(6)] for i in range(len(fdi_map))]
	rows_touched = 0
	for m_row in m_df.iterrows():
		m_raw = m_row[1]
		fips  = m_raw[0]
		date  = m_raw[1]
		data  = [m_raw[i+2]/100.0 for i in range(6)]
		node_key = "{}-{}".format(fips, date)

		# I have mismatching date ranges, so there will 
		# be key errors if i iterate over the much larger
		# date range of the full mobility dataframe. This
		# hack catches that. 
		try:
			idx = fdi_map[node_key]
		except Exception as e:
			idx = -1
		finally:
			if idx != -1:
				x[idx] = data
				rows_touched += 1
	return x


def generateCOOList(date_range, fdi_map, fips_list, hist_window_size):
	# ## Adjacency Information
	# # Need to make the COO list. 
	ADJ_DTYPES = {'fipscounty': 'str', 'fipsneighbor': 'str'}
	adj_df = pd.read_csv(ADJ_FN)

	coo_list   = []

	# First we add all 'geographic adjacency' link
	adj_count  = 0
	key_errors = 0
	for link in adj_df.iterrows():
		# Adding links u -> v and v -> u over all days.
		_, u, _, v = link[1]

		for day in date_range:
			day_str = day.date()
			u_key = "{}-{}".format(u, day_str)
			v_key = "{}-{}".format(v, day_str)
			try:
				u_idx = fdi_map[u_key]
				v_idx = fdi_map[v_key]
				# We add both for a symetric link.
				coo_list.append([u_idx, v_idx])
				coo_list.append([v_idx, u_idx])
				adj_count += 1
			except KeyError:
				key_errors += 1
				pass
			except Exception as e:
				print(e)

	# Then we add all Temporal Links
	temp_count = 0
	for base_day_idx in range(0, len(date_range)-hist_window_size):
		base_day = date_range[base_day_idx]
		bd_str = base_day.date()
		for future_day in date_range[base_day_idx+1 : base_day_idx+hist_window_size+1]:
			fd_str = future_day.date()

			# iterate over each county fips
			for fips in fips_list:

				# Need a link from base_day to future_day
				u_key = "{}-{}".format(fips, bd_str)
				v_key = "{}-{}".format(fips, fd_str)
				try:
					u_idx = fdi_map[u_key]
					v_idx = fdi_map[v_key]
					# Only add past->future link. 
					coo_list.append([u_idx, v_idx])
					temp_count += 1
				except KeyError:
					key_errors += 1
				except Exception as e:
					print(e)
	print("{} adj links, {} temp links in cool list".format(adj_count, temp_count))
	print("{} key errors while generating coo list".format(key_errors))
	return coo_list


def generateTargetList(date_range, fdi_map, data='both'):
	if data == 'both':
		y_raw = [[0, 0] for i in range(len(fdi_map))]
	else:
		y_raw = [0 for i in range(len(fdi_map))]

	ke_count = 0

	for day in date_range:
		csv_fn = "{:0>2}-{:0>2}-{}.csv".format(day.month, day.day, day.year)
		day_df = pd.read_csv("{}/{}".format(CSV_DIR, csv_fn), dtype={'FIPS': 'str'})
		day_df = day_df[day_df['FIPS'].notna()]
		day_df = day_df[day_df['FIPS'].str.len() == 5]

		# Should now have a county-only dataframe.
		for c_row in day_df.iterrows():
			c_raw = c_row[1]
			fips  = c_raw[0]

			n_confirmed = c_raw[7] # TODO - Double check these!!!
			n_dead      = c_raw[8]
			key_str  = "{}-{}".format(fips, day.date())

			try:
				node_idx = fdi_map[key_str]

				if data == 'confirmed':
					y_raw[node_idx] = n_confirmed
				elif data == 'deaths':
					y_raw[node_idx] = n_dead
				else:
					y_raw[node_idx] = [n_confirmed, n_dead]

			except KeyError:
				ke_count += 1

	print("{} key errors building target list".format(ke_count))
	return y_raw


def generateTrainTestMasks(date_range, split_index, fdi_map, fips_list):
	train_mask = [0.0 for i in range(len(fdi_map))]
	test_mask  = [0.0 for i in range(len(fdi_map))]
	key_errors = 0 # TODO -  God i need to figure this out.
	for i in range(len(date_range)):
		for fips in fips_list:
			date_str = date_range[i].date()
			key_str = "{}-{}".format(fips, date_str)

			try:
				idx = fdi_map[key_str]
				if i < split_index:
					# Training!
					train_mask[idx] = 1.0
				else:
					# Testing!
					test_mask[idx] = 1.0
			except KeyError:
				key_errors += 1
			except Exception as e:
				# Don't think ill hit this but w.e.
				print(e)

	return train_mask, test_mask


def writeMapToCSV(fn, src_map, headers):
	# First we handle the creation/existence of the label dir.
	pathlib.Path(fn).parent.mkdir(exist_ok=True)

	# doing this manually bc i cant get pandas to do it right?
	# i mean its def me but lets just say its the panda.
	with open(fn, 'w') as f:
		f.write("{}\n".format(", ".join(headers)))
		for key in src_map:
			val = src_map[key]
			f.write("{}, {}\n".format(key, val))


def writeListToCSV(fn, src_list):
	pathlib.Path(fn).parent.mkdir(exist_ok=True)
	save_df = pd.DataFrame(src_list)
	save_df.to_csv(fn, header=False, index=False)
