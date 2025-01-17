import pytest
from tg_gnn.tg_gsql import create_gsql_query, install_and_run_query

# Fixtures for metadata and expected output
@pytest.fixture
def basic_metadata():
    return {
        "nodes": [
            {
                "vertex_name": "product",
                "features_list": {"feature": "LIST"},
                "label": "label",
                "split": "split",
            }
        ],
        "edges": [
            {
                "rel_name": "rel",
                "src": "product",
                "dst": "product"
            }
        ],
        "data_dir": "/test/data"
    }

@pytest.fixture
def complex_metadata():
    return {
        "nodes": [
            {
                "vertex_name": "product",
                "features_list": {"feature1": "INT", "feature2": "FLOAT"},
                "label": "label",
                "split": "split",
            },
            {
                "vertex_name": "category",
                "features_list": {},
                "label": "",
                "split": "",
            }
        ],
        "edges": [
            {
                "rel_name": "connects",
                "src": "product",
                "dst": "category",
                "features_list": {"weight": "FLOAT"},
                "label": "connection_type",
                "split": "validation",
            },
            {
                "rel_name": "belongs_to",
                "src": "category",
                "dst": "product",
                "features_list": {},
                "label": "",
                "split": "",
            }
        ],
        "data_dir": "/test/data"
    }

@pytest.fixture
def metadata_with_no_split_label():
    return {
        "nodes": [
            {
                "vertex_name": "product",
                "features_list": {"feature1": "INT", "feature2": "FLOAT"},
                "label": "",
                "split": "",
            },
            {
                "vertex_name": "category",
                "features_list": {},
                "label": "",
                "split": "",
            }
        ],
        "edges": [
            {
                "rel_name": "connects",
                "src": "product",
                "dst": "category",
                "features_list": {"weight": "FLOAT"},
                "label": "",
                "split": "",
            },
            {
                "rel_name": "belongs_to",
                "src": "category",
                "dst": "product",
                "features_list": {},
                "label": "",
                "split": "",
            }
        ],
        "data_dir": "/test/data"
    }

# Unit Tests
@pytest.mark.parametrize("num_partitions", [1, 2, 4])
def test_process_nodes_with_partitions(basic_metadata, num_partitions):
    """
    Verify that the node processing section of the query works with different numbers of partitions.
    """
    query = create_gsql_query(basic_metadata, num_partitions=num_partitions)
    for i in range(num_partitions):
        assert f"node_file0_p{i}" in query

def test_empty_feature_list(complex_metadata):
    """
    Verify that nodes and edges with empty feature lists are handled correctly.
    """
    query = create_gsql_query(complex_metadata, num_partitions=2)
    # For node with empty features
    assert "// Process node: category" in query
    assert "FOREACH" not in query  # No feature loop
    # For edge with empty features
    assert "// Process edge: belongs_to" in query
    assert "FOREACH" not in query  # No feature loop

def test_multiple_nodes_and_edges(complex_metadata):
    """
    Verify that multiple nodes and edges are processed correctly.
    """
    query = create_gsql_query(complex_metadata, num_partitions=2)
    # Check presence of all nodes
    for node in complex_metadata["nodes"]:
        assert f"// Process node: {node['vertex_name']}" in query
    # Check presence of all edges
    for edge in complex_metadata["edges"]:
        assert f"// Process edge: {edge['rel_name']}" in query

# test different partitions
@pytest.mark.parametrize("num_partitions", [1, 2, 3])
def test_create_gsql_query_partitions(basic_metadata, num_partitions):
    """
    Test the full GSQL query generation with multiple partition values.
    """
    query = create_gsql_query(basic_metadata, num_partitions=num_partitions)
    assert f"CREATE DISTRIBUTED QUERY data_load_gen_query_dist()" in query
    assert f"getvid(s) % {num_partitions}" in query

# Edge Case Tests
@pytest.mark.parametrize("num_partitions", [2])
def test_edge_case_no_label_no_split(metadata_with_no_split_label, num_partitions):
    """
    Test edge case where nodes and edges have no labels or splits.
    """
    query = create_gsql_query(metadata_with_no_split_label, num_partitions=num_partitions)
    assert "to_string(s.label)" not in query
    assert "to_string(s.split)" not in query
    assert "to_string(e.connection_type)" not in query
    assert "to_string(e.validation)" not in query
