import pathlib
import pandas as pd
import torch
import Utils as U
from torch_geometric.data import Data

ADJ_FN       = "data/source_data/county_adjacency2010.csv"
MOBILITY_FN  = "data/source_data/2020_US_Region_Mobility_Report.csv"
Y_CONF_FN    = "data/source_data/targets_confirmed.csv"
DY_CONF_FN   = "data/source_data/targets_deltas_confirmed.csv"
Y_DEATHS_FN  = "data/source_data/targets_deaths.csv"
DY_DEATHS_FN = "data/source_data/targets_deltas_deaths.csv"
RACT_S_FN    = "data/source_data/full_activity_data.csv"

STATIC_FEATURES_FN = "data/source_data/static_features_by_fips.csv"

X_FN       = "data/derived_data/{}/X_features.csv"
Y_FN       = "data/derived_data/{}/Y_targets.csv"
Y_P_FN     = "data/derived_data/{}/Y_prior_targets.csv"
COO_FN     = "data/derived_data/{}/coo_list.csv" 
FD_MAP_FN  = "data/derived_data/{}/fips_date_nid_map.csv"
CF_FN      = "data/derived_data/{}/counties_fips.csv"
TRAIN_M_FN = "data/derived_data/{}/train_mask.csv"
TEST_M_FN  = "data/derived_data/{}/test_mask.csv"
SUB_MAP_FN = "data/derived_data/{}/subreddit_id_map.csv"
RACT_D_FN  = "data/derived_data/{}/reddit_activity.csv"

### Data Loading ###############################################################
# The following methods simply read in the relevant 
# derived CSV files and return approriate torch an
# pytorch geo objects. 


# Returns the pytorch geometric Data object representing
# the full Spatio-Temporal network of US Counties. 
def getPyTorchGeoData(label):
	with open(X_FN.format(label), "r") as x_f:
		x_df = pd.read_csv(x_f, header=None)

	with open(Y_FN.format(label), "r") as y_f:
		y_df = pd.read_csv(y_f, header=None)

	with open(Y_P_FN.format(label), "r") as y_p_f:
		y_p_df = pd.read_csv(y_p_f, header=None)
		y_p_df.fillna(0, inplace=True)

	with open(COO_FN.format(label), "r") as coo_f:
		coo_df = pd.read_csv(coo_f, header=None)

	with open(TRAIN_M_FN.format(label), "r") as train_m_f:
		train_m_df = pd.read_csv(train_m_f, header=None)

	with open(TEST_M_FN.format(label), "r") as test_m_f:
		test_m_df = pd.read_csv(test_m_f, header=None)	

	# COO Edge List Tensor. Reshaped. 
	coo_t   = torch.tensor(coo_df.values, dtype=torch.long)
	coo_t   = coo_t.reshape((2, len(coo_df.values)))

	# Features and Target Tensors.
	x_t     = torch.tensor(x_df.values, dtype=torch.float)
	y_t     = torch.tensor(y_df.values, dtype=torch.float)
	y_p_t   = torch.tensor(y_p_df.values, dtype=torch.float)

	# Test/Train Mask Tensors.
	train_t = torch.tensor(train_m_df.values, dtype=torch.long)
	test_t  = torch.tensor(test_m_df.values,  dtype=torch.long)
	return Data(x=x_t, y=y_t, edge_index=coo_t, train_mask=train_t, test_mask=test_t, priors=y_p_t)


def getRedditData(label, num_cds):
	with open(SUB_MAP_FN.format(label), "r") as sm_f:
		sub_map_df = pd.read_csv(sm_f)

	with open(RACT_D_FN.format(label), "r") as ract_f:
		ract_df = pd.read_csv(ract_f)

	cd_nids, sub_nids, vals = [], [], []
	for row in ract_df.iterrows():
		cd_nids.append(row[1][0])
		sub_nids.append(row[1][1])
		vals.append(row[1][2])

	num_subs = len(sub_map_df)
	shape = (num_cds, num_subs)

	activity_tensor = torch.sparse_coo_tensor([cd_nids, sub_nids], vals, shape, dtype=torch.float)

	sub_map = dict(zip(sub_map_df['sub_reddit_id'], sub_map_df['sub_idx']))

	return sub_map, activity_tensor


# Returns the map used to find a node index given a 
# key made from the FIPS value of a county, and a 
# 'standard' date string. 
def getFipsDateMap(label):
	with open(FD_MAP_FN.format(label), "r") as fdm_f:
		fd_df = pd.read_csv(fdm_f)

	print(fd_df.columns)
	return dict(zip(fd_df['fdkey'], fd_df['node_id']))


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
def generateFullDataset(start, end, window_size, train_split, dir_label, target_type='both'):
	days = U.getDateRange(start, end)
	fips_values       = generateFIPSList()
	fdi_map, ifd_list = generateFIPSDateMaps(fips_values, days)

	fdi_save_fn = FD_MAP_FN.format(dir_label)
	U.writeMapToCSV(fdi_save_fn, fdi_map, ['fdkey', 'node_id'])
	print("fipsdate -> node index map saved to {}".format(fdi_save_fn))

	target_list, prior_list = generateTargetList(days, fdi_map, data=target_type)	
	print("{} entries in target_list".format(len(target_list)))
	y_fn = Y_FN.format(dir_label)
	U.writeListToCSV(y_fn, target_list)
	print("Y targets saved to {}".format(y_fn))
	y_prior_fn = Y_P_FN.format(dir_label)
	U.writeListToCSV(y_prior_fn, prior_list)
	print("Y priors saved to {}".format(y_prior_fn))

	features = generateFullFeatures(fdi_map, days, window_size, target_list)
	# mob_features = generateMobilityFeatures(fdi_map)
	print("{} entries in features list".format(len(features)))
	x_fn = X_FN.format(dir_label)
	U.writeListToCSV(x_fn, features)
	print("X features saved to {}".format(x_fn))

	coo_list = generateCOOList(days, fdi_map, fips_values, window_size)
	print("{} edges in coo list".format(len(coo_list)))
	coo_fn = COO_FN.format(dir_label)
	U.writeListToCSV(coo_fn, coo_list)
	print("COO Edge List saved to {}".format(coo_fn))

	train_m, test_m = generateTrainTestMasks(days, train_split, fdi_map, fips_values)
	print("{} entries in train/test masks".format(len(train_m)))
	train_fn = TRAIN_M_FN.format(dir_label)
	U.writeListToCSV(train_fn, train_m)
	test_fn  = TEST_M_FN.format(dir_label)
	U.writeListToCSV(test_fn, test_m)
	print("train and test masks saved")

	subid2idx, idx2subid = generateSubIdMaps(days)
	print("generated subreddit maps, {} subreddits in dataset".format(len(subid2idx)))
	submap_fn = SUB_MAP_FN.format(dir_label)
	U.writeMapToCSV(submap_fn, subid2idx, ['sub_reddit_id', 'sub_idx'])
	print("saved subreddit map to {}".format(submap_fn))

	activity_edges = generateActivityData(days, subid2idx, fdi_map)
	print("generated activity edges, {} total".format(len(activity_edges)))
	ract_fn = RACT_D_FN.format(dir_label)
	U.writeListToCSV(ract_fn, activity_edges)
	print("activity edges saved to {}".format(ract_fn))


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


def generateFullFeatures(fdi_map, date_range, horizon, targets):
	a_day = pd.Timedelta(value=1, unit='D')
	static_df = pd.read_csv(STATIC_FEATURES_FN, dtype={'fips': 'str'})
	NUM_FEATURES = len(static_df.columns)-1+horizon	

	# Initial empty feature 'tensor'
	raw_features = [[0 for i in range(NUM_FEATURES)] for cd in range(len(fdi_map))]
	fips_list = generateFIPSList()
	for fips in fips_list:
		static_features = static_df[static_df['fips'] == fips]
		# TODO - Remove this hardcoded logic. 
		
		if len(static_features) == 0:
			continue

		static_features = list(static_features.values[0][1:35])
		# print(static_features)
		for day in date_range:
			idx_key = "{}-{}".format(fips, day.date())
			cd_idx = fdi_map[idx_key]
			
			raw_row = [0 for i in range(NUM_FEATURES)]
			i = 0
			for s in static_features:
				raw_row[i] = s
				i += 1

			# Now we append the prior days values.
			prior_day = day
			while i < horizon:
				prior_day = prior_day-a_day
				idx_key = "{}-{}".format(fips, prior_day.date())

				if idx_key in fdi_map:
					prior_idx = fdi_map[idx_key]
					if targets[prior_idx] != None:
						raw_row[i] = targets[prior_idx]

				i += 1
					
			raw_features[cd_idx] = raw_row

	return raw_features


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


def generateTargetList(date_range, fdi_map, data='confirmed'):
	y_raw = [0 for i in range(len(fdi_map))]
	y_prior_raw = [0 for i in range(len(fdi_map))]
	conf_df = pd.read_csv(Y_CONF_FN, dtype={'fips': 'str'})
	ke_count = 0
	a_day = pd.Timedelta(value=1, unit='D')

	for row in conf_df.iterrows():
		record = row[1]
		fips = record['fips']
		for d in date_range:
			key_str = "{}".format(d.date())
			
			conf_count = record[key_str]
			if conf_count == None:
				continue

			key_str = "{}-{}".format(fips, d.date())
			if key_str in fdi_map:
				n_idx   = fdi_map[key_str]
				y_raw[n_idx] = conf_count

				# Fill in 'prior day y'
				key_str_prior = "{}".format((d-a_day).date())
				if key_str_prior in record:
					prior_count = record[key_str_prior]
					y_prior_raw[n_idx] = prior_count

			else:
				ke_count += 1

	print("{} ke_counts in target gen.".format(ke_count))
	return y_raw, y_prior_raw


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


def generateSubIdMaps(date_range):
	# Make maps SubId -> Idx and Idx -> SubId
	# Only consider subs with activity in the date range?
	day_nums = [d.dayofyear for d in date_range] # Awesome. 
	ract_df  = pd.read_csv(RACT_S_FN)
	ract_df = ract_df[ract_df['day'].isin(day_nums)]


	sub_set = set([])
	for row in ract_df.iterrows():
		sub_id = row[1]['subreddit_id']
		sub_set = sub_set.union([sub_id])

	subid2idx = dict()
	idx2subid = []

	for idx, sub_id in enumerate(sub_set):
		subid2idx[sub_id] = idx
		idx2subid.append(sub_id)

	return subid2idx, idx2subid


def generateActivityData(date_range, subid2idx, fdi_map):
	day_num_map = {d.dayofyear: d.date() for d in date_range}
	ract_df = pd.read_csv(RACT_S_FN)
	raw_data = []

	for row in ract_df.iterrows():
		# We're adding edges to a list. so we have a (i, j, v)
		# where 
		#   i is a CountyDateNode Index
		#   j is a Subreddit Index?
		#   v is the activity value (count)
		fips = row[1]['fips']
		day_num = int(row[1]['day']) 
		sub_id  = row[1]['subreddit_id']
		act_val = row[1]['activeusers']
		
		if day_num in day_num_map:
			cd_key = "{}-{}".format(fips, day_num_map[day_num])

			if cd_key in fdi_map and sub_id in subid2idx:
				cd_idx = fdi_map[cd_key]
				sub_idx = subid2idx[sub_id]
				raw_data.append([cd_idx, sub_idx, act_val])

	return raw_data

