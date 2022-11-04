from concurrent.futures import ProcessPoolExecutor
import os
import csv
from functools import partial
import pandas as pd
import time
from tqdm import tqdm


def make_vecs(output_path, df_dict, user):
    vecs = []
    vid = user[0]

    # if the _graph file exists
    if os.path.exists(f"{output_path}/subgraph/{vid}_graph.csv"):
        # load it to a df and group by the v_type
        sub_graph = pd.read_csv(f"{output_path}/subgraph/{vid}_graph.csv").groupby(
            "v_type"
        )
        # for every vertex type
        for v_type, df in sub_graph:
            verts_of_type = df_dict[v_type]  # get the df for that vertex type

            # get all the vectors in verts_of_type that have matching ids in df
            verts = verts_of_type.loc[
                verts_of_type["id"].isin(df["id"].tolist())
            ].drop_duplicates()

            #  assert that verts were found
            assert verts.shape[0] > 0, (vid, v_type, df.head(), verts.head())

            #  add the verts to the vectors of this user's graph
            vecs.extend(verts.to_numpy())

        # write the vectors to a csv
        cols = ["id", "label"]
        cols.extend([f"feat_{i}" for i in range(len(vecs[0]) - 2)])
        pd.DataFrame(vecs, columns=cols).to_csv(
            f"{output_path}/subgraph/{vid}_vectors.csv", index=False
        )


def assemble_data(output_path:str, parallel:bool):
    users = csv.reader(open(f"{output_path}/Users.csv"))
    next(users, None)  # skip header
    users = list(users)
    df_dict = {
        "User": pd.read_csv(f"{output_path}/Users.csv"),
        "Friend": pd.read_csv(f"{output_path}/Friends.csv"),
    }

    # make one file per subgraph with all its vectors
    if parallel:
        make_vecs_part = partial(make_vecs, output_path, df_dict)
        print(time.ctime())
        with ProcessPoolExecutor() as executor:
            executor.map(make_vecs_part, users, chunksize=20)
    else:
        for user in tqdm(users):
            make_vecs(output_path, df_dict, user)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("output_path", default="tgresponse")
    parser.add_argument("-p",action='store_true')
    print(parser.parse_args())
    output_path = parser.parse_args().output_path
    parallel = bool(parser.parse_args().p)

    print(f'{output_path= }')
    print(f'{parallel= }')
    assemble_data(output_path, parallel)
