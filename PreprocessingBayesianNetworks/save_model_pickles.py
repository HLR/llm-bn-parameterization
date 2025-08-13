import pickle, os, time
from tqdm import tqdm
from pgmpy.readwrite import BIFReader
from pgmpy import __version__ as pgmpy_version
from EstimationofPriorProbabilitiesbyLLMs.utils.helpers import read_names_from_csv

def safe_save(model_dict, filename):
    with open(filename, 'wb') as f:
        pickle.dump({
            'pgmpy_version': pgmpy_version,
            'models': model_dict
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

def safe_load(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        if data['pgmpy_version'] != pgmpy_version:
            print(f"Warning: Saved with pgmpy {data['pgmpy_version']}, current is {pgmpy_version}")
        return data['models']
if __name__ == '__main__':
    names, info_raw = read_names_from_csv('bn_node_explanations.csv')
    print("Pickling models...")
    models = {}
    for model_name in tqdm(names):
        if os.path.exists(f"BNs/Pickles/{model_name}.pkl"):
            continue
        safe_save(BIFReader("BNs/BIFs/" + model_name + ".bif").get_model(), f"BNs/Pickles/{model_name}.pkl")
    print("Models pickled.\n")

    print("Loading models...")
    t0=time.time()
    models = {}
    for model_name in tqdm(names):
        models[model_name] = safe_load(f"BNs/Pickles/{model_name}.pkl")
    print(f"Models loaded in {time.time()-t0:.2f}s.\n")