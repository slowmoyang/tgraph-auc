import json
import uproot
from sklearn import metrics

def get_auc(graph):
    graph = graph.tojson()
    return metrics.auc(graph['fX'], graph['fY'])

path = '/store/scratch/wjang/TEST/20230221_jetDiscriminator_moremore_vars_and_had_vars_udsc_b_g_hadcut_noQG_noCvL_noRecoTopM.root'
bdt_dir_path = 'UL17_BaseLine_ts_tb_udsc_b_g_moremore_vars_and_had_vars_hadcut_noQG_noCvL_noRecoTopM/Method_BDT/BDT'
graph_prefix_tuple = (
    'MVA_BDT_Test_rejBvsS_',
    'MVA_BDT_Test_1v1rejBvsS_ts_vs_'
)

input_file = uproot.open(path)
bdt_dir = input_file[bdt_dir_path]

graph_path_list = [key for key in bdt_dir.keys()
                   if key.startswith(graph_prefix_tuple)]

data = {each: get_auc(bdt_dir[each]) for each in graph_path_list}
with open('auc.json', 'w') as stream:
    json.dump(data, stream, indent=4)
