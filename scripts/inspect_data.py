import argparse
from pathlib import Path
from jepa.data import load_episode,split_episodes

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",default="data/episodes")
    a=p.parse_args()
    paths=sorted(Path(a.data).glob("*.npz"))
    for path in paths:
        images,actions,_,dones=load_episode(path)
        print(path.name,"frames",images.shape,"actions",actions.shape,"done_count",int(dones.sum()))
    if len(paths)>=2:
        train,val=split_episodes(a.data)
        print("train episodes:",[x.name for x in train])
        print("validation episodes:",[x.name for x in val])

if __name__=="__main__":main()
