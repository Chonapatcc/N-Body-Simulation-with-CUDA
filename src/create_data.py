import pandas as pd
import random
import tqdm

ns = [10**i for i in range(1,7)]

POS_RANGE = 1
VEL_RANGE = 0.5
MASS_RANGE1 = 1e-12
MASS_RANGE2 = 1e-11

for n in tqdm.tqdm(ns):
    df = pd.DataFrame(index=range(n), columns=["id", "x", "y", "z","vx", "vy", "vz", "mass"])
    for i in range(n):
        idx = i
        x = random.uniform(-POS_RANGE, POS_RANGE)
        y = random.uniform(-POS_RANGE, POS_RANGE)
        z = random.uniform(-POS_RANGE, POS_RANGE)
        vx = random.uniform(-VEL_RANGE, VEL_RANGE)
        vy = random.uniform(-VEL_RANGE, VEL_RANGE)
        vz = random.uniform(-VEL_RANGE, VEL_RANGE)
        mass = random.uniform(MASS_RANGE1,MASS_RANGE2)
        df.loc[i] = [idx, x, y, z, vx, vy, vz, mass]

    df.to_csv("../data/data_{}.csv".format(n),index = False)