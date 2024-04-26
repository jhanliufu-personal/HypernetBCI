from braindecode.datasets import BaseConcatDataset, BaseDataset
from braindecode.datasets.moabb import _find_dataset_in_moabb, _annotations_from_moabb_stim_channel
import pandas as pd
# from moabb.datasets.base import CacheConfig


class my_MOABBDataset(BaseConcatDataset):
    """A class for moabb datasets.

    Parameters
    ----------
    dataset_name: str
        name of dataset included in moabb to be fetched
    subject_ids: list(int) | int | None
        (list of) int of subject(s) to be fetched. If None, data of all
        subjects is fetched.
    dataset_kwargs: dict, optional
        optional dictionary containing keyword arguments
        to pass to the moabb dataset when instantiating it.
    """
    def __init__(self, dataset_name, subject_ids, dataset_kwargs=None, download_path=None):
        raws, description = my_fetch_data_with_moabb(dataset_name, subject_ids, dataset_kwargs, download_path)
        all_base_ds = [BaseDataset(raw, row)
                       for raw, (_, row) in zip(raws, description.iterrows())]
        super().__init__(all_base_ds)


def my_fetch_data_with_moabb(dataset_name, subject_ids, dataset_kwargs=None, download_path=None):
    # ToDo: update path to where moabb downloads / looks for the data
    """Fetch data using moabb.

    Parameters
    ----------
    dataset_name: str
        the name of a dataset included in moabb
    subject_ids: list(int) | int
        (list of) int of subject(s) to be fetched
    dataset_kwargs: dict, optional
        optional dictionary containing keyword arguments
        to pass to the moabb dataset when instantiating it.

    Returns
    -------
    raws: mne.Raw
    info: pandas.DataFrame
    """
    dataset = _find_dataset_in_moabb(dataset_name, dataset_kwargs)
    subject_id = [subject_ids] if isinstance(subject_ids, int) else subject_ids
    return my_fetch_and_unpack_moabb_data(dataset, subject_id, download_path)


def my_fetch_and_unpack_moabb_data(dataset, subject_ids, download_path):
    ################ This is the reason why I need my_moabb! ################
    if download_path is None:
        data = dataset.get_data(subject_ids)
    else:
        cache_config = {'path': download_path, 'use': True}
        print(f'In my_fetch_and_unpack_moabb_data(): {download_path}')
        data = dataset.get_data(subject_ids, cache_config=cache_config)
    #########################################################################
    raws, subject_ids, session_ids, run_ids = [], [], [], []
    for subj_id, subj_data in data.items():
        for sess_id, sess_data in subj_data.items():
            for run_id, raw in sess_data.items():
                # set annotation if empty
                if len(raw.annotations) == 0:
                    annots = _annotations_from_moabb_stim_channel(raw, dataset)
                    raw.set_annotations(annots)
                raws.append(raw)
                subject_ids.append(subj_id)
                session_ids.append(sess_id)
                run_ids.append(run_id)
    description = pd.DataFrame({
        'subject': subject_ids,
        'session': session_ids,
        'run': run_ids
    })
    return raws, description