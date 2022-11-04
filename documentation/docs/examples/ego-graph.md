# Graph Classification

Graph-level classification on the <a href="http://snap.stanford.edu/data/twitch_ego_nets.html" target="_blank">Twitch Ego Nets</a> dataset from a TigerGraph DB using g2gnn and DGL.

<!-- ##[Check it out](https://github.com/Optum/graph2gnn/tree/main/examples/graph-level/ego/twitch) -->

### Overview

Another option for learning graph structures is at the graph or subgraph level. While node classification takes the region around a node to classify that node, graph-level classification aims to classify the region itself (in the case of subgraph classification), or the graph as a whole (e.g., molecule classification).

The example code is [HERE](https://github.com/Optum/graph2gnn/tree/main/examples/graph-level/ego/twitch). First, [install g2gnn] and navigate to the graph-level/twitch directory under examples. Details are below.

### Load the Data

Once you have stood up your tigergraph instance[^1], you can generate the files to load the data to graph by running [prepare_graph.ipynb]. When it's done, copy the files to your env. You should be able to run [setup.gsql] in the gsql shell to generate the schema, TwitchEgos, load the data and finally install the query.

### The Query

```gsql linenums="1"
CREATE QUERY TwitchEgosData(INT current_partition=0, INT total_partitions=10) FOR GRAPH TwitchEgos SYNTAX V2{
  TYPEDEF TUPLE <INT from_id, INT to_id> Edges; # GNN train type
  TYPEDEF TUPLE <INT id, STRING v_type> Vert; # GNN train type
  
  # Accumulators for tracking edges and involved vertices 
  MaxAccum<INT> @id, @deg;
  MapAccum<INT, SetAccum<Edges>> @@edges;
  MapAccum<INT, SetAccum<Vert>> @@graph;
  
  Users = {User.*};
  Users = SELECT s FROM Users:s 
          WHERE vertex_to_int(s) % total_partitions == current_partition
          ACCUM s.@id += getvid(s),
                @@graph += (getvid(s) -> Vert(getvid(s), s.type)),
                s.@deg += s.outdegree();

  # max 2-hop graph (this should be enough to encompas all subgraphs anyway)
  Friends1 = SELECT f FROM Users:s -(FRIENDSHIP:e)- Friend:f
             WHERE s.label >= 0
             ACCUM f.@id += getvid(f),
                   @@edges += (getvid(s) -> Edges(getvid(s), getvid(f))),
                   @@graph += (getvid(s) -> Vert(getvid(f), f.type)),
                   f.@deg += f.outdegree();
                    
  Friends2 = SELECT tgt FROM Users:s -(FRIENDSHIP:e)- _:f -(FRIENDSHIP:e)- Friend:tgt
             WHERE s.label >= 0 
             ACCUM tgt.@id += getvid(tgt),
                   @@edges += (getvid(s) -> Edges(getvid(f), getvid(tgt))),
                   @@graph += (getvid(s) -> Vert(getvid(tgt), tgt.type)),
                   f.@deg += f.outdegree(),
                   tgt.@deg += tgt.outdegree();

  Friends = Friends1 UNION Friends2;
	
  PRINT @@edges AS _edges; 
  PRINT @@graph AS _graph;
  PRINT Users;
  PRINT Friends;
}
```

Lines 2 and 3 define types that graph2gnn will use to extract the graph structures from the query's response. These types allow for the minimal amount of data to be sent in the response. If you there are edge features in your dataset, add them as attributes to th `Edges` type like this: `TYPEDEF TUPLE <INT from_id, INT to_id, INT feat_name1, String INT feat_name2...> Edges;`

Lines 6 through 8 instantiate ID, feature (`@deg`) and global map accumulators to keep track of the adjacency information and graph membership. `@@edges` is a key-value store that holds the adjacency information of each subgraph where the ID of the subgraph's ego is the key and the values are the edges of that subgraph. `@@graph` has the same keys as edges, but its values simply store which vertices belong to which graph. This allows you to more quickly select features that go into the individual graphs when you're assembling the dataset.

Lines 10 through 27 are the main body of the query. The actions to take note of are additions to the `@@edges` and `@@graph` map accums. Same as in the node-level query, you have access to all of the features of GSQL here to help you refine the subgraphs that you want.

NOTE: It's critical that the edges accumulator is printed as `_edges` and the graph accumulator as `_graph` (lines 33, 34). Graph2GNN looks for `_edges` and `_graph` to reconstruct the adjecency information.

In the case of subgraph classification, it's not uncommon that a vertex is a part of multiple graphs.

## Write a subclass for Tiger2GNN

At this point, you should have a the TwitchEgos graph loaded to your TG instance. Now we'll begin to walk through the python code in [graph-classification.ipynb] to get the data and train a GCN. Feel free to write to write your own code and use the docs and example code as a guide.

The first cell initializes Twitch2GNN, which is subclassed from Tiger2GNN. Subclassing is a convenient way to organize all the code around gathering data from the graph and making it ready for training.

When you initialize the object from your subclass (or just the Tiger2GNN class), you must pass in the host, graph_name, the query you will call and any credential info (e.g. the restpp token).

```python
class Twitch2GNN(Tiger2GNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        ...

	def compute_vertex_vectors(self):
		...


tg = Twitch2GNN(
    host=f"https://{host}",
    graph_name="Cora",
    query="CoraData",
    token=tkn,
)
```

### Subgraph Query Method

The call_subgraph_query calls a query written in the form described above. If there are any parameters, such as partitioning values, pass them in to the params argument. Call the method to get the data.

```python
tg.call_subgraph_query()
```

#### Note

At this point, g2gnn has done its job. You can use the resulting data in any way you wish! What follows is a guide to build the output from call_singlegraph_query into a GCN.

### Exploratory Data Analysis

Once the data has been stored locally, if the data is still raw, now is when you would do any necessary exploratory data analysis (EDA), feature selection, etc. No EDA is needed for this example, since the graph stores the features directly. When the method is finished, the tg response directory will contain the following structure.

```
project-root
├── tgresponse/
│   ├── subgraph/
│   │   ├── <graph ID>_edges.csv    (adjacency info... will become a model input)
│   │   └── <graph ID>_graph.csv    (membership info... use to help select vertex vectors for the graph when assembling the datset)
│   ├── parts/
│   │   └── partition_0.json         (whole query responses, saved by partition)
│   ├── Friends.json                 (vertex set response... create vectors from these files)
│   ├── Users.json
└── graph-classification.ipynb
```

### compute_vertex_vectors

The first of the custom methods in the Twitch2GNN subclass is used to compute, or in this case organize, the vectors that will represent each vertex.

```python
tg.compute_vertex_vectors()
```

### assemble/split

The next cell shows the label split and some other dataset metrics. Here's a good place to have a sanity check.

Unlike the node-level example, the assembly code is moved into a separate script in order to take advantage of concurrent processing.

When the method is finished, the tg response directory will contain the following structure.

```
project-root
├── tgresponse/
│   ├── subgraph/
│   │   ├── <graph ID>_edges.csv       (adjacency info... will become a model input)
│   │   ├── <graph ID>_graph.csv       (membership info... use to help select vertex vectors for the graph when assembling the datset)
│   │   └── <graph ID>_vectors.csv  ***(vectorized vertices... will become a model input)
│   ├── parts/
│   │   └── partition_0.json           
│   ├── Friends.json                   
│   ├── Friends.csv                   
│   ├── Users.json
│   ├── Users.csv
└── graph-classification.ipynb
```

## DGL

Now the graph data is ready. From this point on, feel free to continue in whichever graph-ML library is your preference. this example continues in pytorch-flavored [DGL].

### DGL Dataset

Following the dgl doc's for [making your own dataset], we subclass DGLDataset class and override the `__getitem__`, `__len__`, and `process` methods. When constructing the graphs, keep in mind that node IDs have to be 0-indexed. Any node IDs used previously need to be tranlated into the ID-space for DGL. This is handled on lines 38-47 in the dataset instantiating cell.

```python
class TwitchDataset(DGLDataset):
    def __init__(self, num_nodes, **kwargs):
        super().__init__(**kwargs)

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)

    def process(self):
        # lines 38-47 have the ID transfer code
		...
```

### Model building and training

Again, following [DGL's docs], the model is constructed and trained in the following cells.

```python
class TwitchModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super().__init__()
        self.conv1 = dglnn.GraphConv(in_dim, hidden_dim)
        self.conv2 = dglnn.GraphConv(hidden_dim, n_classes)

    def forward(self, g:dgl.DGLGraph, h):
        h = self.conv1(g, h)
        h = F.relu(h)
        h = self.conv2(g, h)
        g.ndata['h'] = h
        return dgl.mean_nodes(g, 'h')
```

### Conclusion

Congrats on getting to the end! If you run into any issues or questions, please reach out on [Github].

[install g2gnn]: ../index.md#installation
[graph-classification.ipynb]: https://github.com/Optum/graph2gnn/tree/main/examples/graph-level/ego/twitch/graph-classification.ipynb
[DGL]: https://www.dgl.ai
[making your own dataset]: https://docs.dgl.ai/tutorials/blitz/6_load_data.html#creating-a-dataset-for-graph-classification-from-csv
[github]: https://github.com/Optum/graph2gnn/issues
[DGL's docs]: https://docs.dgl.ai/tutorials/blitz/5_graph_classification.html