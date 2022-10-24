# Node Classification
Node classification on the <a href="https://paperswithcode.com/dataset/cora" target="_blank">Cora benchmark</a> dataset from a TigerGraph DB using g2gnn and DGL.

### Overview
Node classification, especially with the Cora dataset, is a common "Hello, world" for GNNs. The dataset is a single, homogenious graph of 2708 academic papers on 7 different topics. The edges between the papers represent one paper citing the works of the other. The task is to use the contents of each paper (one-hot encoded word-vectors) and the graph to determine which topic the paper is about.

The example code is [HERE](https://github.com/Optum/graph2gnn/tree/main/examples/node-level/cora). First, [install g2gnn] and navigate to the node-level/cora directory under examples. Details are below.

### Load the Data
Once you have stood up your tigergraph instance[^1], you can generate the files to load the data to graph by running [prepare_graph.ipynb]. When it's done, copy the files to your env. You should be able to run [setup.gsql] in the gsql shell to generate the schema, CoraGraph, load the data and finally install the query.

### The Query

The query, found at the bottom of [setup.gsql], is responsible for gathering the data and the graphy info that the model will train on.

```gsql linenums="1"
CREATE QUERY CoraData() FOR GRAPH Cora{
	SetAccum<Edge> @@edges;
	PaperSrc = {Paper.*};

	PaperResult = SELECT s FROM PaperSrc:s -(CITES:e)-> Paper:tgt
				  ACCUM @@edges += e;

	PRINT @@edges as _edges;
	PRINT PaperSrc;
}
```

Line 2 instantiates a global set accumulator of type `Edge` and line 6 begins to populate the accumulator. The edges that are in this accumulator will be used to reconstruct the graph's adjacency matrix. Keep in mind that any filtering (`WHERE` or `IF/ELSE`) can be used to more precicely choose which parts of the graph will be used. Since this is not a large graph, it can safely be extracted in one call (no need for partitioning).

If the graph is heterogenious and the query requires multiple hops, you can simply add `@@edges += e` to the `ACCUM` of any select statement to use that part of the graph in the model.

NOTE: It's critical that the edges accumulator is printed as `_edges` (line 8). Graph2GNN looks for `_edges` to reconstruct the adjecency information.

FOR LARGE GRAPHS: Graph2GNN is fully compatible with partioned queries. Replace the beginning of the query with the following lines to return smaller chunks that g2gnn will piece back together.

```gsql linenums="1"
CREATE QUERY CoraData(INT total_partitions=0, INT current_partition) FOR GRAPH Cora{
	SetAccum<Edge> @@edges;
	PaperSrc = {Paper.*};
    PaperSrc = SELECT s FROM PaperSrc:s
                  WHERE vertex_to_int(s) % total_partitions == current_partition;
...
```

## Write a subclass for Tiger2GNN
At this point, you should have a the Cora citation graph loaded to your TG instance. Now we'll begin to walk through the python code in [node-classification.ipynb] to get the data and train a 2-hop GCN. Feel free to write to write your own code and use the docs and example code as a guide.

The first cell initializes Cora2GNN, which is subclassed from Tiger2GNN. Subclassing is a convenient way to organize all the code around gathering data from the graph and making it ready for training.

When you initialize the object from your subclass (or just the Tiger2GNN class), you must pass in the host, graph_name, the query you will call and any credential info (e.g. the restpp token).
```python
class Cora2GNN(Tiger2GNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        ...
    
	def compute_vertex_vectors(self):
		...

	def assemble_data(self, samples_per_class):
		...


tg = Cora2GNN(
    host=f"https://{host}",
    graph_name="Cora",
    query="CoraData",
    token=tkn,
)
```
### Singlegraph Query Method
The call_singlegraph_query calls a query written in the form described above. If there are any parameters, such as partitioning values, pass them in to the params argument. Call the method to get the data.
```python
tg.call_singlegraph_query()
```

#### Note
At this point, g2gnn has done its job. You can use the resulting data in any way you wish! What follows is a guide to build the output from call_singlegraph_query into a GCN.

### Exploratory Data Analysis
Once the data has been stored locally, if the data is still raw, now is when you would do any necessary exploratory data analysis (EDA), feature selection, etc. No EDA is needed for this example, since the graph stores the features directly. When the method is finished, the tg response directory will contain the following structure.
```
project-root
└── tgresponse/
    ├── subgraph/               (empty dir)
    ├── parts/
    │   └── partition_0.json  (whole query responses, saved by partition)
    ├── PaperSrc.json         (vertex set response... create vectors from these files)
    ├── edges.csv               *(adjacency info... will become a model input)
└── node-classification.ipynb
```

### compute_vertex_vectors
The first of the custom methods in the Cora2GNN subclass is used to compute, or in this case organize, the vectors that will represent each vertex.
```python
tg.compute_vertex_vectors()
```

### assemble/split
The next cell shows the label split and some other dataset metrics. Here's a good place to have a sanity check -- there should be 2708 total labels since there are 2708 nodes in the graph.

Within this cell, the second custom method of the Cora2GNN subclass is called. In this notebook, assemble_data organizes the datset into train, eval and test datasets.
```python
tg.assemble_data(samples_per_class=samples_per_class)
```

## DGL
Now the graph data is ready. From this point on, feel free to continue in whichever graph-ML library is your preference. this example continues in pytorch-flavored [DGL].

### DGL Dataset
Following the dgl doc's for [making your own dataset], we subclass DGLDataset class and override the `__getitem__`, `__len__`, and `process` methods. When constructing the graph, keep in mind that node IDs have to be 0-indexed. Any node IDs used previously need to be tranlated into the ID-space for DGL. This is handled in [prepare_graph.ipynb], but you will need to do this if you use DGL for your own solutions. An example can be seen in the [graph-level](ego-graph.md#dgl-dataset) docs ([graph-classification.ipynb])

```python
class CoraDataset(DGLDataset):
    def __init__(self, num_nodes, **kwargs):
        super().__init__(**kwargs)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.graph

    def process(self):
		...
```
### Model building and training
Again, following [DGL's docs] and [Kipf's GCN paper], the model is constructed and trained in the following cells.
```python
class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, drop=0.5):
        super().__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)
        self.d1 = nn.Dropout(drop)

    def forward(self, g, in_feat):
        h = F.dropout(in_feat)
        h = self.conv1(g, h)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h
```

### Conclusion
If all went well, your model should have a test accuracy of ~83%.

Congrats on getting to the end! A lot of problems can be modeled as a single graph where node classification or link prediction is the task. If your problem is better suited as graph-level predictions, check out the [graph-level g2gnn] example module.

[install g2gnn]: ../index.md#installation
[setup.gsql]: https://github.com/Optum/graph2gnn/tree/main/examples/node-level/cora/setup.gsql
[node-classification.ipynb]: https://github.com/Optum/graph2gnn/tree/main/examples/node-level/cora/node-classification.ipynb
[prepare_graph.ipynb]: https://github.com/Optum/graph2gnn/tree/main/examples/node-level/cora/prepare_graph.ipynb
[DGL]: https://www.dgl.ai
[graph-level g2gnn]: ego-graph.md
[making your own dataset]: https://docs.dgl.ai/tutorials/blitz/6_load_data.html
[^1]: https://www.tigergraph.com/get-tigergraph/
[DGL's docs]: https://docs.dgl.ai/tutorials/blitz/1_introduction.html#defining-a-graph-convolutional-network-gcn
[Kipf's GCN paper]: https://arxiv.org/pdf/1609.02907.pdf
[graph-classification.ipynb]: https://github.com/Optum/graph2gnn/tree/main/examples/graph-level/ego/twitch/graph-classification.ipynb
