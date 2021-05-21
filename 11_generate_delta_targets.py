import Data

START_DATE  = "04/12/2020"
END_DATE    = "12/31/2020"
TRAIN_SPLIT_IDX  = 200 # Leaves 64 to test
WINDOW_SIZE = 7 

DS_LABEL = 'w7_wDeltas'

# Works on a 'label' based system. You create a derived
# data set by calling the following method. This creates a 
# data set in a new directory, matching the date and window
# size settings passed into the generation method.

# Uncomment to regenerate data set. 
Data.generateFullDataset(START_DATE, 
	END_DATE, 
	WINDOW_SIZE, 
	TRAIN_SPLIT_IDX, 
	DS_LABEL)
