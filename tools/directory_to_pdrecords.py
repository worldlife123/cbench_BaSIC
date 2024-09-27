import pandas as pd
import numpy as np
import os
import sys

def main(args):

    dir = args[-2]

    data = []
    for i, fname in enumerate(os.listdir(dir)):
        print("Adding {}".format(fname))
        with open(os.path.join(dir, fname), 'rb') as f:
            data.append(f.read())
            # data[fname] = [f.read()]
    # DataFrame is column major
    pd_table = pd.DataFrame(data=data)
    print(pd_table)

    outfile= args[-1]
    np.save(outfile, pd_table.to_records(index=False))

if __name__ == "__main__":
    main(sys.argv)
