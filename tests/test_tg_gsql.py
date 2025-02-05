import pytest
from tg_gnn.tg_gsql import create_gsql_query, install_and_run_query

# Fixtures for metadata and expected output
@pytest.fixture
def basic_metadata():
    return {
        "nodes": {
            "product": {
                "features_list": {"feature": "LIST"},
                "label": "label",
                "split": "split",
            }
        },
        "edges": {
            "rel": {
                "src": "product",
                "dst": "product"
            }
        },
        "data_dir": "/test/data"
    }

@pytest.fixture
def complex_metadata():
    return {
        "nodes": {
            "product": {
                "features_list": {"feature1": "INT", "feature2": "FLOAT"},
                "label": "label",
                "split": "split",
            },
           "category": {
                "vertex_name": "category",
                "features_list": {},
                "label": "",
                "split": "",
            }
        },
        "edges": {
            "connects": {
                "rel_name": "connects",
                "src": "product",
                "dst": "category",
                "features_list": {"weight": "FLOAT"},
                "label": "connection_type",
                "split": "validation",
            },
            "belongs_to": {
                "src": "category",
                "dst": "product",
                "features_list": {},
                "label": "",
                "split": "",
            }
        },
        "data_dir": "/test/data"
    }

@pytest.fixture
def metadata_with_no_split_label():
    return {
        "nodes": {
            "product": {
                "features_list": {"feature1": "INT", "feature2": "FLOAT"},
                "label": "",
                "split": "",
            },
            "category": {
                "vertex_name": "category",
                "features_list": {},
                "label": "",
                "split": "",
            }
        },
        "edges": {
            "connects": {
                "src": "product",
                "dst": "category",
                "features_list": {"weight": "FLOAT"},
                "label": "",
                "split": "",
            },
            "belongs_to": {
                "src": "category",
                "dst": "product",
                "features_list": {},
                "label": "",
                "split": "",
            }
        },
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
    for vertex_name, _ in complex_metadata["nodes"].items():
        assert f"// Process node: {vertex_name}" in query
    # Check presence of all edges
    for rel_name, _ in complex_metadata["edges"].items():
        assert f"// Process edge: {rel_name}" in query

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

def test_minimal_nodes_and_edges():
    metadata = {
        "data_dir": "/data",
        "nodes": {
            "User": {}
        },
        "edges": {
            "FRIENDS": {"src": "User", "dst": "User"}
        }
    }
    num_partitions = 1
    query = create_gsql_query(metadata, num_partitions)
    
    # Test basic structure
    assert "CREATE DISTRIBUTED QUERY data_load_gen_query_dist() {" in query
    assert "node_file0_p0" in query
    assert "edge_file0_p0" in query
    
    # Test node generation
    assert 'FROM User:s' in query
    assert 'node_data_list = to_string(getvid(s))' in query
    assert 'node_file0_p0.println(node_data_list)' in query
    
    # Test edge generation
    assert 'User:s - (FRIENDS:e) -> User:t' in query
    assert 'edge_data_list = to_string(getvid(s)) + "," + to_string(getvid(t))' in query
    assert 'edge_file0_p0.println(edge_data_list)' in query

def test_nodes_with_all_attributes():
    metadata = {
        "data_dir": "/data",
        "nodes": {
            "User": {
                "label": "is_active",
                "split": "train_val_split",
                "features_list": {
                    "numerical": "FLOAT",
                    "tags": "LIST"
                }
            }
        },
        "edges": {}
    }
    num_partitions = 2
    query = create_gsql_query(metadata, num_partitions)
    
    # Test file partitions
    assert "FILE node_file0_p0" in query
    assert "FILE node_file0_p1" in query
    
    # Test label and split
    assert 'to_string(s.is_active)' in query
    assert 'to_string(s.train_val_split)' in query
    
    # Test features
    assert 'to_string(s.numerical)' in query
    assert 'FOREACH feature IN s.tags DO' in query
    
    # Test partition logic
    assert 'IF getvid(s) % 2 == 0' in query
    assert 'IF getvid(s) % 2 == 1' in query

def test_edges_with_all_attributes():
    metadata = {
        "data_dir": "/data",
        "nodes": {},
        "edges": {
            "TRANSACTION": {
                "src": "User",
                "dst": "Product",
                "label": "fraudulent",
                "split": "split_ratio",
                "features_list": {
                    "amount": "FLOAT",
                    "categories": "LIST"
                }
            }
        }
    }
    num_partitions = 1
    query = create_gsql_query(metadata, num_partitions)
    
    # Test edge metadata
    assert 'User:s - (TRANSACTION:e) -> Product:t' in query
    assert 'to_string(e.fraudulent)' in query
    assert 'to_string(e.split_ratio)' in query
    assert 'to_string(e.amount)' in query
    assert 'FOREACH feature IN e.categories DO' in query

def test_multiple_partitions():
    metadata = {
        "data_dir": "/data",
        "nodes": {
            "User": {}
        },
        "edges": {
            "FRIENDS": {"src": "User", "dst": "User"}
        }
    }
    num_partitions = 3
    query = create_gsql_query(metadata, num_partitions)
    
    # Test partition files
    assert "FILE node_file0_p0" in query
    assert "FILE node_file0_p1" in query
    assert "FILE node_file0_p2" in query
    assert "FILE edge_file0_p0" in query
    assert "FILE edge_file0_p1" in query
    assert "FILE edge_file0_p2" in query
    
    # Test partition logic
    assert 'IF getvid(s) % 3 == 0' in query
    assert 'IF getvid(s) % 3 == 1' in query
    assert 'IF getvid(s) % 3 == 2' in query

def test_list_feature_handling():
    metadata = {
        "data_dir": "/data",
        "nodes": {
            "Product": {
                "features_list": {
                    "prices": "LIST",
                    "name": "STRING"
                }
            }
        },
        "edges": {}
    }
    query = create_gsql_query(metadata, 1)
    
    # Test list feature expansion
    assert 'FOREACH feature IN s.prices DO' in query
    assert 'node_data_list = node_data_list + "," + to_string(feature)' in query
    # Test regular feature
    assert 'to_string(s.name)' in query

def test_edge_case_single_partition():
    metadata = {
        "data_dir": "/data",
        "nodes": {
            "User": {}
        },
        "edges": {
            "FRIENDS": {"src": "User", "dst": "User"}
        }
    }
    num_partitions = 1
    query = create_gsql_query(metadata, num_partitions)
    
    # Verify single partition syntax
    assert "END;" in query  # Last partition should end with semicolon
    assert "END,\n" not in query  # No comma for last partition
