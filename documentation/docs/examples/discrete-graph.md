# Discrete Graph - Regression

### Overview

Graph-level regression on the <a href="https://paperswithcode.com/paper/alchemy-a-quantum-chemistry-dataset-for/review/" target="_blank">QM7b</a> dataset from TigerGraph using g2gnn.

QM7b is a dataset for multitask learning. Since this is just an example, only one of those tasks is trained. The dataset is a homogeneous graph of 7,211 separate molucules where each vertex is one atom and each edge is a bond.

The example code is [HERE](https://github.com/Optum/graph2gnn/tree/main/examples/graph-level/discrete). Details are below.

### Write the query

The query, found at the bottom of [setup.gsql](https://github.com/Optum/graph2gnn/tree/main/examples/graph-level/discrete/setup.gsql). It's responsible for gathering the graphy info that the model will train on.

```sql linenums="1"
CREATE QUERY QM7bData(INT total_partitions=0, INT current_partition) FOR GRAPH QM7b{
  SetAccum<Edge> @edges;
  MaxAccum<VERTEX> @lead_node;

  AtomSrc = {Atom.*};

  AtomResult = SELECT s FROM AtomSrc:s -(_:e)-> Atom:t
               ACCUM s.@edges += e,
                 IF s.label.size() > 0 THEN // Then it is the leader node
                   s.@lead_node += s,
                   t.@lead_node += s
                 END;

  AtomResult = SELECT s FROM AtomResult:s
               WHERE vertex_to_int(s) % total_partitions == current_partition;

  PRINT AtomResult;
}
```

Lines 2 and 3 instantiate local set accumulators of types `Edge` and `VERTEX`. They will be used to store each vertex's adjacency info as well as which node in their graph acts as the leader node (the one that the graph will be named after). In the case of QM7b, the leader node holds the label, as well. 

The select statement on line 7 populates the result set and accumulators. The edges that in the edges accumulator will be used to reconstruct the adjacency matricies. Keep in mind that any filtering (`WHERE` or `IF/ELSE`) can be used to more precicely choose which part(s) of the graph are returned.

If the graph is heterogenious and the query requires multiple hops, you can simply add `s.@edges += e` to the `ACCUM` of any select statement to use that part of the graph in the model.

The partitioning is done at the end to be sure that every vertex that's returned in the partitioned result set has its graph membership

## Write a subclass for Tiger2GNN

### subclass

### compute_vertex_vectors

### assemble/split

## DGL

### DGL Dataset

### DGL NN
