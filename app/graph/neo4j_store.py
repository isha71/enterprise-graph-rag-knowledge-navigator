from neo4j import AsyncGraphDatabase
from app.models.domain import (
    EntityType, RelationshipType, ExtractedEntity,
    ExtractedRelationship, GraphNode
)
from app.graph.normalization import normalize_entity_name, compute_entity_id
from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.exceptions import GraphQueryError

logger = get_logger(__name__)

VALID_REL_TYPES = {rt.value for rt in RelationshipType}

class Neo4jStore:
    def __init__(self):
        settings = get_settings()
        self._driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
        )
        self._database = settings.neo4j_database
    
    async def close(self):
        await self._driver.close()
    
    async def initialize(self):
        """Create constraints."""
        async with self._driver.session(database=self._database) as session:
            await session.run(
                "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS "
                "FOR (n:Entity) REQUIRE n.id IS UNIQUE"
            )
            logger.info("Neo4j constraints initialized")
    
    async def upsert_entity(
        self, entity: ExtractedEntity, document_id: str, chunk_id: str,
        document_name: str = "",
    ) -> str:
        """Upsert an entity node. Returns entity ID."""
        normalized = normalize_entity_name(entity.name, entity.type)
        entity_id = compute_entity_id(entity.type, normalized)
        
        entity_type_label = entity.type.value
        
        query = (
            f"MERGE (n:Entity:{entity_type_label} {{id: $id}}) "
            "ON CREATE SET n.name = $name, n.normalized_name = $normalized_name, "
            "n.type = $type, n.source_chunk_ids = [$chunk_id], "
            "n.source_document_ids = [$document_id], "
            "n.source_document_names = [$document_name] "
            "ON MATCH SET "
            "n.source_chunk_ids = CASE WHEN NOT $chunk_id IN n.source_chunk_ids "
            "THEN n.source_chunk_ids + $chunk_id ELSE n.source_chunk_ids END, "
            "n.source_document_ids = CASE WHEN NOT $document_id IN n.source_document_ids "
            "THEN n.source_document_ids + $document_id ELSE n.source_document_ids END, "
            "n.source_document_names = CASE WHEN NOT $document_name IN n.source_document_names "
            "THEN n.source_document_names + $document_name ELSE n.source_document_names END "
            "RETURN n.id"
        )
        
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                query,
                id=entity_id,
                name=entity.name,
                normalized_name=normalized,
                type=entity.type.value,
                chunk_id=chunk_id,
                document_id=document_id,
                document_name=document_name,
            )
            record = await result.single()
        
        return entity_id
    
    async def upsert_relationship(
        self,
        source_entity: ExtractedEntity,
        target_entity: ExtractedEntity,
        rel_type: RelationshipType,
        document_id: str,
        chunk_id: str,
        document_name: str = "",
    ):
        """Upsert a relationship with provenance."""
        source_normalized = normalize_entity_name(source_entity.name, source_entity.type)
        target_normalized = normalize_entity_name(target_entity.name, target_entity.type)
        source_id = compute_entity_id(source_entity.type, source_normalized)
        target_id = compute_entity_id(target_entity.type, target_normalized)
        
        if rel_type.value not in VALID_REL_TYPES:
            raise GraphQueryError(f"Invalid relationship type: {rel_type}")
        
        query = (
            f"MATCH (a:Entity {{id: $source_id}}) "
            f"MATCH (b:Entity {{id: $target_id}}) "
            f"MERGE (a)-[r:{rel_type.value}]->(b) "
            "ON CREATE SET r.source_chunk_ids = [$chunk_id], "
            "r.source_document_ids = [$document_id], "
            "r.source_document_names = [$document_name] "
            "ON MATCH SET "
            "r.source_chunk_ids = CASE WHEN NOT $chunk_id IN r.source_chunk_ids "
            "THEN r.source_chunk_ids + $chunk_id ELSE r.source_chunk_ids END, "
            "r.source_document_ids = CASE WHEN NOT $document_id IN r.source_document_ids "
            "THEN r.source_document_ids + $document_id ELSE r.source_document_ids END, "
            "r.source_document_names = CASE WHEN NOT $document_name IN r.source_document_names "
            "THEN r.source_document_names + $document_name ELSE r.source_document_names END"
        )
        
        async with self._driver.session(database=self._database) as session:
            await session.run(
                query,
                source_id=source_id,
                target_id=target_id,
                chunk_id=chunk_id,
                document_id=document_id,
                document_name=document_name,
            )
    
    async def persist_extraction(
        self,
        extraction,  # GraphExtraction
        document_id: str,
        chunk_id: str,
        document_name: str = "",
    ) -> tuple[int, int]:
        """Persist all entities and relationships. Returns (entity_count, rel_count).
        
        Raises on infrastructure/database errors (connection, transaction, Cypher
        execution failures). Validation-level issues (e.g. invalid relationship
        type) are logged and skipped.
        """
        entity_count = 0
        for entity in extraction.entities:
            try:
                await self.upsert_entity(entity, document_id, chunk_id, document_name)
                entity_count += 1
            except GraphQueryError as e:
                logger.error(f"Validation error upserting entity {entity.name}: {e}")
            except Exception:
                raise
        
        entity_map = {e.name.lower(): e for e in extraction.entities}
        
        rel_count = 0
        for rel in extraction.relationships:
            source = entity_map.get(rel.source.lower())
            target = entity_map.get(rel.target.lower())
            if source and target:
                try:
                    await self.upsert_relationship(
                        source, target, rel.type, document_id, chunk_id, document_name
                    )
                    rel_count += 1
                except GraphQueryError as e:
                    logger.error(f"Validation error upserting relationship: {e}")
                except Exception:
                    raise
        
        return entity_count, rel_count
    
    async def find_entity(self, name: str) -> dict | None:
        """Find entity by normalized name (exact then partial match)."""
        from app.graph.normalization import normalize_entity_name
        
        # Try exact match across all entity types
        for etype in EntityType:
            normalized = normalize_entity_name(name, etype)
            query = (
                "MATCH (n:Entity {normalized_name: $normalized_name}) "
                "RETURN n"
            )
            async with self._driver.session(database=self._database) as session:
                result = await session.run(query, normalized_name=normalized)
                record = await result.single()
                if record:
                    node = record["n"]
                    return dict(node)
        
        # Fallback: case-insensitive partial match
        query = (
            "MATCH (n:Entity) "
            "WHERE toLower(n.name) CONTAINS toLower($name) "
            "RETURN n LIMIT 1"
        )
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, name=name)
            record = await result.single()
            if record:
                return dict(record["n"])
        
        return None
    
    async def find_entities_by_names(self, names: list[str]) -> list[dict]:
        """Find multiple entities by name."""
        entities = []
        for name in names:
            entity = await self.find_entity(name)
            if entity:
                entities.append(entity)
        return entities
    
    async def get_entity_neighbors(self, entity_id: str) -> dict:
        """Get entity and its immediate neighbors."""
        query = (
            "MATCH (n:Entity {id: $entity_id}) "
            "OPTIONAL MATCH (n)-[r]-(neighbor:Entity) "
            "RETURN n, collect(DISTINCT {neighbor: neighbor, rel_type: type(r), "
            "direction: CASE WHEN startNode(r) = n THEN 'outgoing' ELSE 'incoming' END}) as neighbors"
        )
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, entity_id=entity_id)
            record = await result.single()
            if not record:
                return {}
            
            node = dict(record["n"])
            neighbors = []
            for n in record["neighbors"]:
                if n["neighbor"] is not None:
                    neighbors.append({
                        "entity": dict(n["neighbor"]),
                        "relationship_type": n["rel_type"],
                        "direction": n["direction"],
                    })
            
            return {"entity": node, "neighbors": neighbors}
    
    async def multi_hop_traversal(
        self,
        seed_id: str,
        max_hops: int = 4,
        max_paths: int = 12,
        relationship_hints: list[str] | None = None,
        prefer_longer: bool = False,
    ) -> list[dict]:
        """
        Bounded multi-hop graph traversal from a seed entity.
        Returns list of paths ranked by relationship-hint coverage.
        """
        # Clamp max_hops for safety
        max_hops = min(max(int(max_hops), 1), 4)
        hints = relationship_hints or []

        if hints:
            query = (
                "MATCH (seed:Entity {id: $seed_id}) "
                f"MATCH p=(seed)-[*1..{max_hops}]-(connected:Entity) "
                "WITH p, "
                "size([hint IN $hints "
                "  WHERE any(r IN relationships(p) WHERE type(r) = hint)"
                "]) AS hint_matches "
                "ORDER BY hint_matches DESC, "
                "CASE WHEN $prefer_longer THEN length(p) ELSE -length(p) END DESC "
                "LIMIT $max_paths "
                "RETURN p"
            )
            params = {
                "seed_id": seed_id,
                "hints": hints,
                "prefer_longer": prefer_longer,
                "max_paths": max_paths,
            }
        else:
            query = (
                "MATCH (seed:Entity {id: $seed_id}) "
                f"MATCH p=(seed)-[*1..{max_hops}]-(connected:Entity) "
                "RETURN p LIMIT $max_paths"
            )
            params = {"seed_id": seed_id, "max_paths": max_paths}
        
        paths = []
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, **params)
            async for record in result:
                path = record["p"]
                path_nodes = []
                path_rels = []
                
                for node in path.nodes:
                    path_nodes.append({
                        "name": node.get("name", ""),
                        "type": node.get("type", ""),
                        "id": node.get("id", ""),
                        "source_chunk_ids": list(node.get("source_chunk_ids", [])),
                        "source_document_ids": list(node.get("source_document_ids", [])),
                        "source_document_names": list(node.get("source_document_names", [])),
                    })
                
                for idx, rel in enumerate(path.relationships):
                    start_node_id = rel.start_node.get("id", "")
                    traversal_from_id = path_nodes[idx]["id"] if idx < len(path_nodes) else ""
                    forward = (start_node_id == traversal_from_id)
                    path_rels.append({
                        "type": rel.type,
                        "forward": forward,
                        "source_chunk_ids": list(rel.get("source_chunk_ids", [])),
                        "source_document_ids": list(rel.get("source_document_ids", [])),
                        "source_document_names": list(rel.get("source_document_names", [])),
                    })
                
                paths.append({"nodes": path_nodes, "relationships": path_rels})
        
        return paths
    
    async def is_healthy(self) -> bool:
        """Check if Neo4j is reachable."""
        try:
            async with self._driver.session(database=self._database) as session:
                result = await session.run("RETURN 1")
                await result.single()
            return True
        except Exception:
            return False
