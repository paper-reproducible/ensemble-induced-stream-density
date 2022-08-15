import os
import sys
import pandas as pd
from Common import query_pandas

sql_avg = """
SELECT 
  dataset, 
  isolation_method, 
  psi, 
  eps, 
  minPts,
  avg(mi) as avg_mi,
  avg(ami) as avg_ami,
  avg(nmi) as avg_nmi,
  avg(ri) as avg_ri,
  avg(ari) as avg_ari
FROM raw_data
GROUP BY
  dataset, 
  isolation_method, 
  psi, 
  eps, 
  minPts
"""

sql_max = """
SELECT 
    dataset,
    isolation_method,
    max(avg_mi) as max_avg_mi,
    max(avg_ami) as max_avg_ami,
    max(avg_nmi) as max_avg_nmi,
    max(avg_ri) as max_avg_ri,
    max(avg_ari) as max_avg_ari
FROM avg_data
GROUP BY 
    dataset,
    isolation_method
"""


def main(argv):
    if len(argv) <= 0:
        print("input folder?")
        return
    folder_name = argv[0]

    save_to = folder_name + "/benchmark_dbscan_aggregated.csv"
    if os.path.exists(save_to):
        os.remove(save_to)

    avg_results = None
    for f in os.listdir(folder_name):
        if not f.endswith(".csv"):
            continue
        file_name = folder_name + "/" + f
        print("Loading...", file_name)
        df = query_pandas(sql_avg, raw_data=file_name)
        if avg_results is None:
            avg_results = df
        else:
            avg_results = pd.concat([avg_results, df])

    print("Calculating best...")
    best_avg_results = query_pandas(sql_max, avg_data=avg_results)
    print("Saving...", save_to)
    best_avg_results.to_csv(save_to)


if __name__ == "__main__":
    main(sys.argv[1:])

# python Scripts\benchmark_cluster_aggregate.py Data/2022081511_dbscan
