"""Neo4j graph database implementation."""
from typing import Dict, List, Optional, Any
from neo4j import GraphDatabase as _neo4j_driver, Driver
import logging

from .interface import GraphDatabase

logger = logging.getLogger(__name__)

# Reduce Neo4j deprecation/notification log spam (id→elementId, property warnings)
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)


class Neo4jGraphDatabase(GraphDatabase):
    """Neo4j implementation of the GraphDatabase interface."""
    
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: Optional[str] = None
    ):
        """
        Initialize Neo4j connection.
        
        Args:
            uri: Neo4j connection URI (e.g., "bolt://localhost:7687")
            user: Neo4j username
            password: Neo4j password
            database: Optional database name (defaults to default database)
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver: Optional[Driver] = None
    
    def connect(self) -> None:
        """Establish connection to Neo4j."""
        try:
            self.driver = _neo4j_driver.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            # Verify connectivity
            self.driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {self.uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def disconnect(self) -> None:
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
            self.driver = None
            logger.info("Disconnected from Neo4j")
    
    def _get_session(self):
        """Get a Neo4j session."""
        if not self.driver:
            raise RuntimeError("Not connected to Neo4j. Call connect() first.")
        return self.driver.session(database=self.database)
    
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a node by its ID (elementId or legacy integer id)."""
        # Neo4j 5+: use elementId (stable). Fallback: legacy id() for numeric strings.
        if node_id and ":" in node_id:
            query = "MATCH (n) WHERE elementId(n) = $node_id RETURN n, labels(n) as labels, elementId(n) as eid LIMIT 1"
            param = node_id
        else:
            query = "MATCH (n) WHERE id(n) = toInteger($node_id) RETURN n, labels(n) as labels, elementId(n) as eid LIMIT 1"
            param = node_id
        
        with self._get_session() as session:
            result = session.run(query, node_id=param)
            record = result.single()
            
            if record:
                node = dict(record["n"])
                eid = record.get("eid")
                return {
                    "id": str(eid) if eid is not None else node_id,
                    "labels": record["labels"],
                    "properties": node
                }
            return None
    
    def get_neighbors(
        self,
        node_id: str,
        relationship_types: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get neighboring nodes of a given node."""
        if relationship_types:
            rel_filter = f":{':'.join(relationship_types)}"
        else:
            rel_filter = ""
        
        limit_clause = f"LIMIT {limit}" if limit else ""
        
        if node_id and ":" in node_id:
            where_clause = "elementId(n) = $node_id"
            param = node_id
        else:
            where_clause = "id(n) = toInteger($node_id)"
            param = node_id
        
        query = f"""
        MATCH (n)-[r{rel_filter}]-(neighbor)
        WHERE {where_clause}
        RETURN neighbor, labels(neighbor) as labels, elementId(neighbor) as neighbor_eid, id(neighbor) as neighbor_id,
               type(r) as relationship_type, properties(r) as rel_properties
        {limit_clause}
        """
        
        with self._get_session() as session:
            result = session.run(query, node_id=param)
            neighbors = []
            
            for record in result:
                neighbor = dict(record["neighbor"])
                eid = record.get("neighbor_eid")
                neighbors.append({
                    "id": str(eid) if eid is not None else str(record.get("neighbor_id", "")),
                    "labels": record["labels"],
                    "properties": neighbor,
                    "relationship_type": record["relationship_type"],
                    "relationship_properties": record["rel_properties"]
                })
            
            return neighbors
    
    def get_node_properties(self, node_id: str) -> Dict[str, Any]:
        """Get all properties of a node."""
        if node_id and ":" in node_id:
            query = "MATCH (n) WHERE elementId(n) = $node_id RETURN properties(n) as props LIMIT 1"
            param = node_id
        else:
            query = "MATCH (n) WHERE id(n) = toInteger($node_id) RETURN properties(n) as props LIMIT 1"
            param = node_id
        
        with self._get_session() as session:
            result = session.run(query, node_id=param)
            record = result.single()
            
            if record:
                return dict(record["props"])
            return {}
    
    def get_relationships(
        self,
        node_id: str,
        relationship_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get all relationships of a node."""
        if relationship_types:
            rel_filter = f":{':'.join(relationship_types)}"
        else:
            rel_filter = ""
        
        if node_id and ":" in node_id:
            where_clause = "elementId(n) = $node_id"
            param = node_id
        else:
            where_clause = "id(n) = toInteger($node_id)"
            param = node_id
        
        query = f"""
        MATCH (n)-[r{rel_filter}]-(other)
        WHERE {where_clause}
        RETURN type(r) as type, properties(r) as properties,
               elementId(startNode(r)) as start_id, elementId(endNode(r)) as end_id,
               labels(startNode(r)) as start_labels,
               labels(endNode(r)) as end_labels
        """
        
        with self._get_session() as session:
            result = session.run(query, node_id=param)
            relationships = []
            
            for record in result:
                relationships.append({
                    "type": record["type"],
                    "properties": dict(record["properties"]),
                    "start_id": str(record["start_id"]),
                    "end_id": str(record["end_id"]),
                    "start_labels": record["start_labels"],
                    "end_labels": record["end_labels"]
                })
            
            return relationships
    
    def query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a custom query on Neo4j."""
        params = parameters or {}
        
        with self._get_session() as session:
            result = session.run(query, **params)
            return [dict(record) for record in result]
    
    def get_subgraph(
        self,
        node_id: str,
        depth: int = 2,
        relationship_types: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get the subgraph around a node up to a given depth.
        Returns center node, all nodes/edges within depth, and all paths.
        """
        if relationship_types:
            rel_filter = f":{':'.join(relationship_types)}"
        else:
            rel_filter = ""

        if node_id and ":" in node_id:
            where_clause = "elementId(center) = $node_id"
            param = node_id
        else:
            where_clause = "id(center) = toInteger($node_id)"
            param = node_id

        limit_clause = f"LIMIT {limit}" if limit else "LIMIT 500"

        # Query all paths from the center node up to the given depth
        query = f"""
        MATCH path = (center)-[r{rel_filter}*1..{depth}]-(connected)
        WHERE {where_clause}
        WITH path, center, connected,
             [n IN nodes(path) | {{
                 id: elementId(n),
                 labels: labels(n),
                 properties: properties(n)
             }}] AS path_nodes,
             [rel IN relationships(path) | {{
                 source_id: elementId(startNode(rel)),
                 target_id: elementId(endNode(rel)),
                 type: type(rel),
                 properties: properties(rel)
             }}] AS path_rels
        RETURN path_nodes, path_rels,
               elementId(center) AS center_id,
               labels(center) AS center_labels,
               properties(center) AS center_props
        {limit_clause}
        """

        with self._get_session() as session:
            result = session.run(query, node_id=param)

            nodes_map: Dict[str, Dict[str, Any]] = {}
            edges_list: List[Dict[str, Any]] = []
            paths: List[List[Dict[str, Any]]] = []
            center_node = None
            seen_edges = set()

            for record in result:
                # Build center node on first record
                if center_node is None:
                    center_id = str(record["center_id"])
                    center_node = {
                        "id": center_id,
                        "labels": record["center_labels"],
                        "properties": dict(record["center_props"]),
                    }
                    nodes_map[center_id] = center_node

                # Collect nodes
                path_nodes = record["path_nodes"]
                for pn in path_nodes:
                    nid = str(pn["id"])
                    if nid not in nodes_map:
                        nodes_map[nid] = {
                            "id": nid,
                            "labels": list(pn["labels"]),
                            "properties": dict(pn["properties"]),
                        }

                # Collect edges (deduplicate)
                path_rels = record["path_rels"]
                for pr in path_rels:
                    edge_key = (str(pr["source_id"]), str(pr["target_id"]), pr["type"])
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        edge = {
                            "source_id": str(pr["source_id"]),
                            "target_id": str(pr["target_id"]),
                            "type": pr["type"],
                            "properties": dict(pr["properties"]) if pr["properties"] else {},
                        }
                        edges_list.append(edge)

                # Build the path as alternating nodes and edges
                path = []
                for i, pn in enumerate(path_nodes):
                    nid = str(pn["id"])
                    path.append(nodes_map[nid])
                    if i < len(path_rels):
                        pr = path_rels[i]
                        path.append({
                            "source_id": str(pr["source_id"]),
                            "target_id": str(pr["target_id"]),
                            "type": pr["type"],
                            "properties": dict(pr["properties"]) if pr["properties"] else {},
                        })
                paths.append(path)

            if center_node is None:
                # No paths found; fetch the center node directly
                center_node_data = self.get_node(node_id)
                if center_node_data:
                    center_node = center_node_data
                    nodes_map[center_node["id"]] = center_node
                else:
                    center_node = {"id": node_id, "labels": [], "properties": {}}

            return {
                "center": center_node,
                "nodes": nodes_map,
                "edges": edges_list,
                "paths": paths,
            }

    def get_node_count(self) -> int:
        """Get total number of nodes in the graph."""
        query = "MATCH (n) RETURN count(n) as count"
        
        with self._get_session() as session:
            result = session.run(query)
            record = result.single()
            return record["count"] if record else 0
