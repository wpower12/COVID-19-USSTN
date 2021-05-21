import Data
import Utils

START_DATE  = "04/12/2020"
END_DATE    = "12/31/2020"

date_range = Utils.getDateRange(START_DATE, END_DATE)
fip_2_idx, idx_2_fip = Data.generateFipsMaps()

x = Data.generateFullSequences(date_range, fip_2_idx)
y = Data.generateFullTargets(date_range, fip_2_idx)