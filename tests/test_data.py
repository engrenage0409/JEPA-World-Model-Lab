import numpy as np
import pytest
from jepa.data import split_episodes,ClipDataset

def test_episodes_and_windows(tmp_path):
    for i in range(3):
        np.savez_compressed(tmp_path/f"{i}.npz",
          observations=np.zeros((10,96,96,3),dtype=np.uint8),
          actions=np.zeros((9,3),dtype=np.float32),
          rewards=np.zeros(9,dtype=np.float32),
          dones=np.array([False]*8+[True]))
    tr,va=split_episodes(tmp_path)
    assert len(tr)==2 and len(va)==1
    dataset=ClipDataset(tr,steps=4,stride=4)
    assert len(dataset)>0
    row=dataset[0]
    assert row["frames"].shape==(5,3,96,96)
    assert row["actions"].shape==(4,3)

def test_invalid_split(tmp_path):
    with pytest.raises(ValueError): split_episodes(tmp_path)
