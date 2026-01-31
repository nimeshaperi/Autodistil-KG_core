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
    
    def get_node_count(self) -> int:
        """Get total number of nodes in the graph."""
        query = "MATCH (n) RETURN count(n) as count"
        
        with self._get_session() as session:
            result = session.run(query)
            record = result.single()
            return record["count"] if record else 0
