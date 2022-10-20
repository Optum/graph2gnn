## Subclassing (Under Construction)

It's best practice to subclass Tiger2GNN and implement the methods in a way that's more custom to your use case 

Here's an example (todo needs commenting)
``` python
from graph2gnn import Tiger2GNN
import numpy as np
from tqdm import tqdm
import json

class MyImplementation(Tiger2GNN):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.response_set_keys = []
		self.nlp = None
		self.num_labels = None
		self.labels = None
		self.output_path='subgraph'

def compute_vertex_vectors(self, key='VertexType'):
	vectors = []
	path = f'{self.output_path}/{key}.json'
	vertex_list = json.load(open(path))
	# embed using word vectorization
	for vert in tqdm(vertex_list, desc=key):
			vector = make_vector(vert) # turn the vertex json into a vector
			vectors.append([vert['v_id'], vector])
      
	# save computational progress
	save_arr = np.array(vectors, dtype=object)
	np.save(f'{self.output_path}/vectors/{key}', save_arr)  
			
def determine_labels(self):
	indvs = json.load(open(f'{self.output_path}/Indv.json'))
	labels = {}
	for i in tqdm(indvs):
		label = i['attributes']['@label']
		labels[i['v_id']] = label
	self.labels = labels

''' ------------------------------------------------ '''

# create the config object
tg = MyImplementation(
	host='https://your-graph-host',
	graph_name='MyGraph',
	token='<bearer token>',
	query='my_graph2gnn_query',
	cert_path='path/to/cert.crt'
	)

resp = tg.call_singlegraph_query()
tg.compute_adjacency_matxs()
tg.compute_vertex_vectors()
tg.determine_labels()
```