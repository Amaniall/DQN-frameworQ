import pickle
import numpy as np
import pandas as pd
if __name__ == '__main__':
    filename = 'evaluation.pkl'
    l_step = 0.1
    result_dict = {}
    with open(filename, 'rb') as f:
        x = pickle.load(f)

    for k, v in x.items():
        avg_time_processus = np.sum(v[3]) / v[-1][-1] * l_step
        avg_length_platoon = (np.array(v[0][1]) / np.array(v[2]))[-1]
        cav_rate_lane0 = (np.array(v[1][0]) / np.array(v[0][0]))[-1]
        # ttd_veh = (np.sum(v[0][0]) + np.sum(v[0][1])) * l_step
        # ttd_cav = (np.sum(v[1][0]) + np.sum(v[1][1])) * l_step
        # ttd_veh_lane0 = np.sum(v[0][0]) * l_step
        rate_faillure = (v[-3][-1] + v[-2][-1]) / v[-1][-1]
        # result_dict[k] = [avg_time_processus, avg_length_platoon, cav_rate_lane0, ttd_veh, ttd_cav, ttd_veh_lane0,
        # rate_faillure]
        result_dict[k] = [avg_time_processus, avg_length_platoon, cav_rate_lane0, rate_faillure]

    df = pd.DataFrame(result_dict).sort_index(axis=1)
    df.to_excel('eval.xlsx')
