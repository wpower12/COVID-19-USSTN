import Data

START_DATE  = "04/12/2020"
END_DATE    = "12/31/2020"
WINDOW_SIZE = 7 

DS_LABEL = "test_gen"

# Now works on a 'label' based system. You create a derived
# data set by calling the following method. This creates a 
# data set in a new directory, matching the date and window
# size settings passed into the generation method.
Data.generateFullDataset(START_DATE, END_DATE, WINDOW_SIZE, DS_LABEL)

# Then, the derived data set can be quickly loaded using:
full_graph = Data.getPyTorchGeoData(DS_LABEL)
print(full_graph)
