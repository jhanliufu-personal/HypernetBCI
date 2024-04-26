from braindecode.datasets import MOABBDataset, SleepPhysionet

# BNCI2014_001 has 9 subjects
# BNCI2014_001_dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=list(range(1, 10)))

# Schirrmeister2017 has 14 subjects
# Schirrmeister2017_dataset = MOABBDataset(dataset_name="Schirrmeister2017", subject_ids=list(range(1, 15)))

# SleepPhysionet has 78 subjects and 2 recordings per subject
sleepphysionet_dataset = SleepPhysionet(subject_ids=range(79), recording_ids=[1, 2,], crop_wake_mins=30)