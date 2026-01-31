#!/usr/bin/env python3
"""
DBLP to Neo4j Loader Script

This script loads DBLP publication data into a Neo4j graph database.
"""
import gzip
import html.entities
import xml.etree.ElementTree as ET
from xml.parsers.expat import ExpatError
from html import unescape
from neo4j import GraphDatabase
from typing import Dict, List, Set
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DBLPNeo4jLoader:
    def __init__(self, uri: str, user: str, password: str):
        """
        Initialize connection to Neo4j database.
        
        Args:
            uri: Neo4j connection URI (e.g., "bolt://localhost:7687")
            user: Neo4j username
            password: Neo4j password
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info("Connected to Neo4j database")
    
    def close(self):
        """Close the database connection."""
        self.driver.close()
        logger.info("Closed Neo4j connection")
    
    def create_constraints(self):
        """Create constraints and indexes for better performance."""
        with self.driver.session() as session:
            constraints = [
                "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Publication) REQUIRE p.key IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (v:Venue) REQUIRE v.name IS UNIQUE",
                "CREATE INDEX IF NOT EXISTS FOR (p:Publication) ON (p.year)",
                "CREATE INDEX IF NOT EXISTS FOR (p:Publication) ON (p.title)"
            ]
            
            for constraint in constraints:
                try:
                    session.run(constraint)
                    logger.info(f"Created: {constraint}")
                except Exception as e:
                    logger.warning(f"Constraint/Index already exists or error: {e}")
    
    def parse_dblp_xml(self, xml_file: str, batch_size: int = 1000, max_records: int = None):
        """
        Parse DBLP XML file and load into Neo4j in batches.
        
        Args:
            xml_file: Path to DBLP XML file (can be .gz compressed)
            batch_size: Number of records to process in each batch
            max_records: Maximum number of records to process (None for all)
        """
        logger.info(f"Starting to parse DBLP XML file: {xml_file}")
        
        # Publication types to process
        pub_types = {'article', 'inproceedings', 'proceedings', 'book', 
                     'incollection', 'phdthesis', 'mastersthesis', 'www'}
        
        publications_batch = []
        count = 0
        
        # Preprocess XML to handle HTML entities
        # Read the file content and replace HTML entities before parsing
        logger.info("Reading and preprocessing XML file to handle HTML entities...")
        
        if xml_file.endswith('.gz'):
            with gzip.open(xml_file, 'rt', encoding='utf-8') as f:
                content = f.read()
        else:
            with open(xml_file, 'r', encoding='utf-8') as f:
                content = f.read()
        
        # Replace all HTML named entities before parsing - DBLP uses many entities
        # (e.g. &iacute;, &ntilde;) that aren't in the XML spec; the parser fails
        # without the DTD. Exclude amp, lt, gt, quot, apos - those must stay.
        _xml_critical = {'amp', 'lt', 'gt', 'quot', 'apos'}
        entity_replacements = {
            f'&{k};': v for k, v in html.entities.html5.items()
            if k not in _xml_critical and isinstance(v, str)
        }
        for entity, replacement in entity_replacements.items():
            content = content.replace(entity, replacement)
        
        # Create a file-like object from the processed content
        import io
        processed_file = io.StringIO(content)
        
        try:
            # Use iterparse for memory-efficient parsing
            context = ET.iterparse(processed_file, events=('start', 'end'))
            context = iter(context)
            
            for event, elem in context:
                if event == 'end' and elem.tag in pub_types:
                    pub_data = self._extract_publication_data(elem)
                    if pub_data:
                        publications_batch.append(pub_data)
                        count += 1
                        
                        if len(publications_batch) >= batch_size:
                            self._load_batch(publications_batch)
                            publications_batch = []
                            logger.info(f"Processed {count} publications")
                        
                        if max_records and count >= max_records:
                            break
                    
                    # Clear element to free memory
                    elem.clear()
            
            # Load remaining publications
            if publications_batch:
                self._load_batch(publications_batch)
                logger.info(f"Processed final batch. Total: {count} publications")
        
        finally:
            # processed_file is a StringIO, no need to close explicitly
            # File was already closed in the with statement above
            pass
        
        logger.info(f"Completed parsing. Total publications processed: {count}")
    
    def _extract_publication_data(self, elem) -> Dict:
        """Extract publication data from XML element."""
        try:
            pub_data = {
                'key': elem.get('key', ''),
                'type': elem.tag,
                'title': '',
                'authors': [],
                'year': None,
                'venue': '',
                'pages': '',
                'volume': '',
                'number': '',
                'url': '',
                'ee': '',
                'doi': ''
            }
            
            for child in elem:
                # Helper function to safely extract and unescape text
                def get_text(element):
                    if element.text:
                        # Unescape HTML entities
                        text = element.text.strip()
                        return unescape(text)
                    return None
                
                if child.tag == 'author':
                    text = get_text(child)
                    if text:
                        pub_data['authors'].append(text)
                elif child.tag == 'title':
                    text = get_text(child)
                    if text:
                        pub_data['title'] = text
                elif child.tag == 'year':
                    text = get_text(child)
                    if text:
                        try:
                            pub_data['year'] = int(text)
                        except ValueError:
                            pass
                elif child.tag in ['journal', 'booktitle']:
                    text = get_text(child)
                    if text:
                        pub_data['venue'] = text
                elif child.tag == 'pages':
                    text = get_text(child)
                    if text:
                        pub_data['pages'] = text
                elif child.tag == 'volume':
                    text = get_text(child)
                    if text:
                        pub_data['volume'] = text
                elif child.tag == 'number':
                    text = get_text(child)
                    if text:
                        pub_data['number'] = text
                elif child.tag == 'url':
                    text = get_text(child)
                    if text:
                        pub_data['url'] = text
                elif child.tag == 'ee':
                    text = get_text(child)
                    if text:
                        pub_data['ee'] = text
                elif child.tag == 'doi':
                    text = get_text(child)
                    if text:
                        pub_data['doi'] = text
            
            # Only return if we have at least a key and title
            if pub_data['key'] and pub_data['title']:
                return pub_data
            
            return None
        
        except Exception as e:
            logger.error(f"Error extracting publication data: {e}")
            return None
    
    def _load_batch(self, publications: List[Dict]):
        """Load a batch of publications into Neo4j."""
        with self.driver.session() as session:
            session.execute_write(self._create_publications_tx, publications)
    
    @staticmethod
    def _create_publications_tx(tx, publications: List[Dict]):
        """Transaction function to create publications and relationships."""
        
        # Create publications
        query_pub = """
        UNWIND $pubs AS pub
        MERGE (p:Publication {key: pub.key})
        SET p.title = pub.title,
            p.type = pub.type,
            p.year = pub.year,
            p.pages = pub.pages,
            p.volume = pub.volume,
            p.number = pub.number,
            p.url = pub.url,
            p.ee = pub.ee,
            p.doi = pub.doi
        """
        tx.run(query_pub, pubs=publications)
        
        # Create authors and relationships
        for pub in publications:
            if pub['authors']:
                query_authors = """
                MATCH (p:Publication {key: $key})
                UNWIND $authors AS author_name
                MERGE (a:Author {name: author_name})
                MERGE (a)-[:AUTHORED]->(p)
                """
                tx.run(query_authors, key=pub['key'], authors=pub['authors'])
            
            # Create venue and relationship
            if pub['venue']:
                query_venue = """
                MATCH (p:Publication {key: $key})
                MERGE (v:Venue {name: $venue})
                MERGE (p)-[:PUBLISHED_IN]->(v)
                """
                tx.run(query_venue, key=pub['key'], venue=pub['venue'])
    
    def create_coauthor_relationships(self):
        """Create CO_AUTHOR relationships between authors who published together."""
        logger.info("Creating co-author relationships...")
        
        with self.driver.session() as session:
            query = """
            MATCH (a1:Author)-[:AUTHORED]->(p:Publication)<-[:AUTHORED]-(a2:Author)
            WHERE id(a1) < id(a2)
            WITH a1, a2, COUNT(p) as collaborations
            MERGE (a1)-[r:CO_AUTHOR]-(a2)
            SET r.collaborations = collaborations
            """
            result = session.run(query)
            logger.info("Co-author relationships created")
    
    def get_statistics(self) -> Dict:
        """Get statistics about the loaded data."""
        with self.driver.session() as session:
            stats = {}
            
            # Count nodes
            stats['publications'] = session.run("MATCH (p:Publication) RETURN count(p) as count").single()['count']
            stats['authors'] = session.run("MATCH (a:Author) RETURN count(a) as count").single()['count']
            stats['venues'] = session.run("MATCH (v:Venue) RETURN count(v) as count").single()['count']
            
            # Count relationships
            stats['authored'] = session.run("MATCH ()-[r:AUTHORED]->() RETURN count(r) as count").single()['count']
            stats['published_in'] = session.run("MATCH ()-[r:PUBLISHED_IN]->() RETURN count(r) as count").single()['count']
            stats['co_authors'] = session.run("MATCH ()-[r:CO_AUTHOR]-() RETURN count(r) as count").single()['count']
            
            return stats


def main():
    """Main function to run the DBLP loader."""
    
    # Configuration
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "password"  # Change this!
    
    # Path to DBLP XML file
    # Download from: https://dblp.org/xml/
    DBLP_XML_FILE = "scripts/dblp.xml.gz"  # or "dblp.xml"
    
    # Initialize loader
    loader = DBLPNeo4jLoader(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    try:
        # Create constraints and indexes
        logger.info("Creating constraints and indexes...")
        loader.create_constraints()
        
        # Parse and load DBLP data
        # Set max_records to limit for testing (e.g., 10000), or None for all
        logger.info("Loading DBLP data into Neo4j...")
        loader.parse_dblp_xml(
            DBLP_XML_FILE, 
            batch_size=1000, 
            max_records=None  # Set to a number for testing, e.g., 10000
        )
        
        # Create co-author relationships
        loader.create_coauthor_relationships()
        
        # Print statistics
        stats = loader.get_statistics()
        logger.info("=" * 50)
        logger.info("Database Statistics:")
        logger.info(f"  Publications: {stats['publications']:,}")
        logger.info(f"  Authors: {stats['authors']:,}")
        logger.info(f"  Venues: {stats['venues']:,}")
        logger.info(f"  AUTHORED relationships: {stats['authored']:,}")
        logger.info(f"  PUBLISHED_IN relationships: {stats['published_in']:,}")
        logger.info(f"  CO_AUTHOR relationships: {stats['co_authors']:,}")
        logger.info("=" * 50)
        
    finally:
        loader.close()


if __name__ == "__main__":
    main()