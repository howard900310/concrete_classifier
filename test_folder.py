import os
from read_yaml import parse_yaml

yaml_path = './configs.yaml'
cfg = parse_yaml(yaml_path)
output_path = cfg['test_data_save_path']

os.mkdir(output_path)
os.mkdir(output_path + '/0_normal')
os.mkdir(output_path + '/1_spalling')
os.mkdir(output_path + '/2')
os.mkdir(output_path + '/3_rebar_exposed')
os.mkdir(output_path + '/4')
os.mkdir(output_path + '/5_unknow')