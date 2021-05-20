import censusdata as cd 
import pandas as pd 

TARGET_TABLES_FN = "data/source_data/acs_table_list.csv"
RAW_SAVE_FN = "data/source_data/raw_acs5_data.csv" # Just for now.
OUTPUT_FN   = "data/source_data/static_features_by_fips.csv"

target_tables = pd.read_csv(TARGET_TABLES_FN)
target_tables = [t[0] for t in target_tables.values]

all_counties = cd.censusgeo([('county', '*')])
raw_df = cd.download('acs5', 2019, all_counties, target_tables)

# This is so weird.  Not a fan of 'multi level indexes' that should just
# be additional columns. just use the fips. or just use sequential. blarg. 
raw_df['state']  = [i.params()[0][1] for i in raw_df.index]
raw_df['county'] = [i.params()[1][1] for i in raw_df.index]
raw_df.to_csv(RAW_SAVE_FN)

age_groups = [
	'age_u_5',  
	'age_5_9',  
	'age_10_14',   
	'age_15_17',   
	'age_18_19',  
	'age_20',       
	'age_21',       
	'age_22_24',   
	'age_25_29',   
	'age_30_34',   
	'age_35_39',   
	'age_40_44',   
	'age_45_49',   
	'age_50_54',   
	'age_55_59',   
	'age_60_61',  
	'age_62_64',   
	'age_65_66',  
	'age_67_69',   
	'age_70_74',   
	'age_75_79',   
	'age_80_84',   
	'age_85_u']

male_age_cols   = [
	'B01001_003E',
	'B01001_004E',
	'B01001_005E',
	'B01001_006E',
	'B01001_007E',
	'B01001_008E',
	'B01001_009E',
	'B01001_010E',
	'B01001_011E',
	'B01001_012E',
	'B01001_013E',
	'B01001_014E',
	'B01001_015E',
	'B01001_016E',
	'B01001_017E',
	'B01001_018E',
	'B01001_019E',
	'B01001_020E',
	'B01001_021E',
	'B01001_022E',
	'B01001_023E',
	'B01001_024E',
	'B01001_025E']

female_age_cols = [
	'B01001_027E',
	'B01001_028E',
	'B01001_029E',
	'B01001_030E',
	'B01001_031E',
	'B01001_032E',
	'B01001_033E',
	'B01001_034E',
	'B01001_035E',
	'B01001_036E',
	'B01001_037E',
	'B01001_038E',
	'B01001_039E',
	'B01001_040E',
	'B01001_041E',
	'B01001_042E',
	'B01001_043E',
	'B01001_044E',
	'B01001_045E',
	'B01001_046E',
	'B01001_047E',
	'B01001_048E',
	'B01001_049E']

## Population, Age Ratios
TOTAL_POP_COL = "B01001_001E"
keep_columns = [TOTAL_POP_COL]
for i in range(len(age_groups)):
	age_group = age_groups[i]
	raw_df[age_group] = raw_df[male_age_cols[i]]+raw_df[female_age_cols[i]]
	raw_df["ratio_{}".format(age_group)] = raw_df[age_group]/raw_df[TOTAL_POP_COL]
	keep_columns.append("ratio_{}".format(age_group))

## Race
race_columns = [
	'B01001A_001E',
	'B01001B_001E',
	'B01001C_001E',
	'B01001D_001E',
	'B01001E_001E',
	'B01001F_001E',
	'B01001G_001E',
	'B01001H_001E',
	'B01001I_001E']

for i in range(len(race_columns)):
	race_column = race_columns[i]
	new_col_name = "ratio_{}".format(race_column)
	raw_df[new_col_name] = raw_df[race_column]/raw_df[TOTAL_POP_COL]
	keep_columns.append(new_col_name)

## Normalized Population
size_col = "norm_population"
keep_columns.append(size_col)
raw_df[size_col] = (raw_df[TOTAL_POP_COL]-raw_df[TOTAL_POP_COL].mean()) / raw_df[TOTAL_POP_COL].std()


## FIPS value
keep_columns.append('fips')
raw_df['fips'] = (raw_df['state']+raw_df['county']).copy()

## Save it!
raw_df[keep_columns].to_csv(OUTPUT_FN)
