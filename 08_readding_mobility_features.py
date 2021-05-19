import Data

START_DATE  = "04/12/2020"
END_DATE    = "12/31/2020"
TRAIN_SPLIT_IDX  = 200 # Leaves 64 to test
WINDOW_SIZE = 7 
DS_LABEL = 'w7_readded_mob'

Data.generateFullDataset(START_DATE, 
	END_DATE, 
	WINDOW_SIZE, 
	TRAIN_SPLIT_IDX, 
	DS_LABEL,
	target_type='confirmed')

