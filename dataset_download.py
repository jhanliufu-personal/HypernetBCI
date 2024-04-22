from braindecode.datasets import MOABBDataset

# This will download data to default path, ~mne_data.
# Need to figure out how to specify download path
# BNCI2014_001 has 9 subjects
# BNCI2014_001_dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=list(range(1, 10)))
# Schirrmeister2017 has 14 subjects
Schirrmeister2017_dataset = MOABBDataset(dataset_name="Schirrmeister2017", subject_ids=list(range(1, 15)))