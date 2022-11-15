import json
import logging
import os
from collections import defaultdict
from time import perf_counter
from typing import Any, Dict, List, Text

import pandas as pd
import requests as req
from tqdm import trange

from graph2gnn.exceptions import QueryException


class Tiger2GNN:
    """
    Class when the DB is Tigergraph
    """

    def __init__(
        self,
        host,
        graph_name,
        query,
        token=None,
        cert_path=None,
        output_path="tgresponse",
        log_level=logging.WARNING,
    ):
        """
        Args:
            host: http or https followed by the DB's host (e.g. https://localhost)
            graph_name: The subgraph that data will be queried from
            query: The query that will be called
            token: The Bearer token to query the graph via RESTPP
            cert_path: The path to the cert for ssl validation
        """
        logging.basicConfig(level=log_level)
        self.query = query
        self.host = host
        self.graph_name = graph_name
        self.token = token
        self.cert_path = cert_path
        self.headers = {"Authorization": f"Bearer {self.token}"}
        self.output_path = output_path
        self.response_set_keys = None

    @staticmethod
    def check_response(response: req.Response, start: float):
        """
        Check the response from TG. Ensure the status code is 200 and there is no error.
        Args:
            response: response to be verified
            start: time when the query was started. Used for tracking query time for optimization
        Returns:
            0 for OK, 1 for Error

        """
        # handle bad http responses
        if response.status_code != 200:
            logging.error([response, response.text])
            end = perf_counter()
            time = end - start
            logging.info(f"query response time: {time}")
            return (1, response.text)

        # handle errors from TG
        elif response.json()["error"]:
            logging.error(response.json()["message"])
            end = perf_counter()
            time = end - start
            logging.info(f"query response time: {time}")
            return (1, response.json()["message"])
        return (0, None)

    def _make_output_dir(self):
        """
        Make the directory to save the output if it does not exist. If it does exist, delete and recreate the path.

        """
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
            os.mkdir(self.output_path + "/subgraph")
            os.mkdir(self.output_path + "/parts")
        else:
            if not os.path.exists(self.output_path + "/subgraph"):
                os.mkdir(self.output_path + "/subgraph")
            if not os.path.exists(self.output_path + "/parts"):
                os.mkdir(self.output_path + "/parts")

    def call_or_read_query(
        self,
        params: Dict[Text, Any],
        partition_path: Text,
        query_url: Text,
        start_time: float,
    ) -> dict:
        """Call the query or read the current partition from a file if the current partion has been previously called and saved to a file

        Args:
            params (Dict[Text, Any]): Parameters passed to the query
            partition_path (Text): Path to save the partion or read the partition from
            query_url (Text): The URL to call the query
            start_time (float): the start time of the previous function. Used to track how long the call_x_query function takes

        Raises:
            Exception: If the graph's response is incorrect for some reason.

        Returns:
            req.Response: The query's response
        """
        if os.path.exists(partition_path):
            logging.info(f"reading partition {partition_path}")
            with open(partition_path) as fin:
                response = json.load(fin)
        else:
            response = req.get(
                query_url, params=params, headers=self.headers, verify=self.cert_path
            )
            check = self.check_response(response, start_time)
            if check[0] == 1:
                raise QueryException(check[1])

            with open(partition_path, "w") as f:
                json.dump(response.json(), f, indent=1)
            response = response.json()
        return response

    def call_singlegraph_query(
        self,
        params: dict = None,
        edge_features: List[Text] = None,
        num_calls: int = None,
    ) -> None:
        """
        Call a query that is written to return single subgraph.
        It stores each query response as its raw json response that can be used during development. If the query doesn't need to be called multiple times, feed the partition json's in to be parsed.
        Edge pairs are stored in self.output_path/edges.csv.
        Vertices are stored in self.output_path/{VertexType}.json.

        Args:
            params: The query params.
            edge_features: A list of the names of edge features that will be extracted.

        Returns:
            None
        """
        logging.info("call_singlegraph_query...")
        # make the directory to save the output if it does not exist
        self._make_output_dir()

        if params is None:
            params = {"total_partitions": 1}
        if num_calls is not None:
            tr = trange(num_calls)
        else:
            tr = trange(params["total_partitions"])

        query_url = f"{self.host}:9000/query/{self.graph_name}/{self.query}"
        whole_response = defaultdict(list)
        edges_list = []
        logging.info(f"calling query: {query_url}")

        start = perf_counter()
        for i in tr:
            params["current_partition"] = i
            partition_path = f"{self.output_path}/parts/partition_{i}.json"
            response = self.call_or_read_query(params, partition_path, query_url, start)
            for v_set in response["results"]:
                key = list(v_set.keys())[0]
                # if the  current vertex set/query response object is not the adj list or lists of vertices
                if key not in ["_edges"]:
                    whole_response[key].extend(v_set[key])
                else:
                    # assemble and save the adjacency list
                    for edge in v_set[key]:
                        if edge_features is None:
                            edges_list.append([edge["from_id"], edge["to_id"]])
                        else:
                            e = [edge["from_id"], edge["to_id"]]
                            e.extend([edge[f] for f in edge_features])
                            edges_list.append(e)

        end = perf_counter()
        time = end - start
        logging.info(f"Run time: {time}")
        logging.info("saving vertex sets as separate files...")
        self.response_set_keys = list(whole_response.keys())
        for k in whole_response.keys():
            json.dump(
                whole_response[k], open(f"{self.output_path}/{k}.json", "w"), indent=1
            )

        # TODO optimize storage space... don't write everything to a csv? scipy coo npz?
        if edge_features is None:
            edge_features = []
        if len(edges_list) == 0:
            raise ValueError("There is no adjacency information for the graph")
        pd.DataFrame(edges_list, columns=["src", "tgt"] + edge_features).to_csv(
            f"{self.output_path}/edges.csv", index=False
        )

    def call_multi_graph_query(
        self,
        params: Dict[Text, Any] = None,
        edge_features: List[Text] = None,
        num_calls: int = None,
    ) -> None:
        """
        Call a query that is written to return multiple, descrete graphs.

        It stores each query response as its raw json response that can be used during development. If the query doesn't need to be called multiple times, feed the partition json's in to be parsed.
        Edge pairs are stored in self.output_path/edges.csv.
        Vertices are stored in self.output_path/{VertexType}.json.

        Args:
            params: The query params.
            edge_features: A list of the names of edge features that will be extracted.

        Returns:
            None
        """
        logging.info("call_multi_graph_query...")
        # make the directory to save the output if it does not exist
        self._make_output_dir()

        if params is None:
            params = {"total_partitions": 1}
        if num_calls is not None:
            tr = trange(num_calls)
        else:
            tr = trange(params["total_partitions"])

        query_url = f"{self.host}:9000/query/{self.graph_name}/{self.query}"
        whole_response = defaultdict(list)
        edges_list = []
        logging.info(f"calling query: {query_url}")

        start = perf_counter()
        # call the query total_partitions times
        for i in tr:
            params["current_partition"] = i
            partition_path = f"{self.output_path}/parts/partition_{i}.json"
            response = self.call_or_read_query(params, partition_path, query_url, start)

            # parse the results
            for v_set in response["results"]:
                key = list(v_set.keys())[0]
                for v in v_set[key]:
                    attrs = v["attributes"]
                    v_id = attrs["id"]
                    graph_id = attrs["@lead_node"]
                    labels = attrs["label"]
                    edges: list = attrs["@edges"]
                    whole_response[key].append(
                        {"id": v_id, "graph_id": graph_id, "label": labels}
                    )
                    for edge in edges:
                        if edge_features is None:
                            edges_list.append(
                                [edge["from_id"], edge["to_id"], graph_id]
                            )
                        else:
                            e = [edge["from_id"], edge["to_id"], graph_id]
                            e.extend([edge["attributes"][f] for f in edge_features])
                            edges_list.append(e)

        end = perf_counter()
        time = end - start
        logging.info(f"query response time: {time}")
        logging.info("saving vertex sets as separate files...")
        if edge_features is None:
            edge_features = []
        pd.DataFrame(
            edges_list, columns=["src", "tgt", "graph_id"] + edge_features
        ).to_csv(f"{self.output_path}/edges.csv", index=False)

        self.response_set_keys = list(whole_response.keys())
        for k in whole_response.keys():
            json.dump(
                whole_response[k], open(f"{self.output_path}/{k}.json", "w"), indent=1
            )

    def call_subgraph_query(
        self,
        params: Dict[Text, Any] = None,
        edge_features: List[Text] = None,
        num_calls: int = None,
    ) -> None:
        """
        Call a query that is written to return multiple, descrete graphs that are a part of one larger, potentially connected graph.

        It stores each query response as its raw json response that can be used during development. If the query doesn't need to be called multiple times, feed the partition json's in to be parsed.
        Edge pairs are stored in self.output_path/{EGO-ID}_edges.csv.
        Vertices are stored in self.output_path/{VertexType}.json.

        Args:
            params: The query params.
            edge_features: A list of the names of edge features that will be extracted.

        Returns:
            None
        """
        logging.info("call_subgraph_query...")
        if params is None:
            params = {"total_partitions": 1}
        if num_calls is not None:
            tr = trange(num_calls)
        else:
            tr = trange(params["total_partitions"])

        if num_calls is not None:
            tr = trange(num_calls)
        else:
            tr = trange(params["total_partitions"])

        # make the directory to save the output if it does not exist
        self._make_output_dir()

        query_url = f"{self.host}:9000/query/{self.graph_name}/{self.query}"
        whole_response = defaultdict(list)
        logging.info(f"calling query: {query_url}")

        start = perf_counter()
        # call the query total_partitions times
        for i in tr:
            params["current_partition"] = i
            partition_path = f"{self.output_path}/parts/partition_{i}.json"
            response = self.call_or_read_query(params, partition_path, query_url, start)
            keys = [list(x.keys())[0] for x in response["results"]]
            if "_edges" not in keys:
                raise QueryException("_edges not in query response")
            elif "_graph" not in keys:
                raise QueryException("_graph not in query response")

            # parse the response
            for v_set in response["results"]:
                key = list(v_set.keys())[0]
                # if the  current vertex set/query response object is not the adj list or lists of vertices
                if key[0] != "_":
                    for v in v_set[key]:
                        attrs = v["attributes"]
                        v_id = attrs["@id"]
                        whole_response[key].append(
                            {"id": v_id, "attrs": attrs}
                        )  # TODO dataclass?
                else:
                    arr = []
                    # assemble and save the adjacency list for each recorded vert v
                    if key == "_edges":
                        for v in v_set[key].keys():
                            for edge in v_set[key][v]:
                                if edge_features is None:
                                    arr.append([edge["from_id"], edge["to_id"]])
                                else:
                                    e = [edge["from_id"], edge["to_id"]]
                                    e.extend([edge[f] for f in edge_features])
                                    arr.append(e)
                            if edge_features is None:
                                edge_features = []
                            pd.DataFrame(
                                arr, columns=["src", "tgt"] + edge_features
                            ).to_csv(
                                f"{self.output_path}/subgraph/{v}_edges.csv",
                                index=False,
                            )
                            arr.clear()
                    elif key == "_graph":
                        for v in v_set[key].keys():
                            for vert in v_set[key][v]:
                                arr.append((vert["id"], vert["v_type"]))
                            pd.DataFrame(arr, columns=["id", "v_type"]).to_csv(
                                f"{self.output_path}/subgraph/{v}_graph.csv",
                                index=False,
                            )
                            arr.clear()

        end = perf_counter()
        time = end - start
        logging.info(f"run time: {time}")
        logging.info("saving vertex sets as separate files...")
        self.response_set_keys = list(whole_response.keys())
        for k in whole_response.keys():
            json.dump(
                whole_response[k], open(f"{self.output_path}/{k}.json", "w"), indent=1
            )
