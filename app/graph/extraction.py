import json
from app.models.domain import (
    EntityType, RelationshipType, ExtractedEntity,
    ExtractedRelationship, GraphExtraction
)
from app.graph.schema import (
    is_valid_entity_type,
    is_valid_relationship_type,
    is_valid_relationship_signature,
)
from app.core.logging import get_logger
from app.core.exceptions import ExtractionError

logger = get_logger(__name__)

EXTRACTION_PROMPT = """You are an entity and relationship extraction system.

Extract entities and relationships from the text below.

Allowed entity types: {entity_types}
Allowed relationship types: {relationship_types}

Relationship definitions (use ONLY these source -> target patterns):

  WORKS_IN: Employee -> Team
    Use only when the text explicitly says an employee works in or is a member of a team.
  MANAGES: Employee -> Team
    Use only when the employee explicitly manages or leads the team.
  OWNS: Team -> Project
    Use only when the team explicitly owns the project.
  MAINTAINS: Employee -> Project | Team -> Service
    Use only when explicitly stated as maintaining or responsible for maintenance.
  USES: Project -> Technology | Service -> Technology
    Use when the project or service explicitly uses the technology.
  DEPENDS_ON: Project -> Service | Service -> Service
    Use only for explicit dependency statements.
  SUPPORTED_BY: Technology -> Vendor
    Use when text explicitly states that a technology is supported by a vendor.
  AFFECTS: Incident -> Service
    Use when an incident explicitly affects a service.
  GOVERNED_BY: Project -> Policy
    Use when a project is explicitly governed by a policy.
  REPORTS_TO: Employee -> Department
    Use only when an employee explicitly reports to the department.

Rules:
- Process the text sentence by sentence.
- Extract ALL explicitly stated relationships that match the allowed schema.
- Do not stop after finding one relationship of a given type. If several sentences independently state relationships, extract each one.
- Prefer literal relationships directly expressed by the wording.
- Do not convert contextual statements such as "primary consumer", "relies on this partnership", "through Project Atlas", or "benefits from" into relationships unless they exactly match an allowed relationship definition.
- Extract only facts explicitly stated in the text.
- Do not infer unstated information.
- Do not use external or world knowledge.
- Only use the allowed entity types and relationship types listed above.
- Every relationship source and target must correspond to extracted entities.
- Return valid JSON only, no additional text.

Positive examples:

  "Project Atlas uses Neo4j."
  => Entity: Project Atlas (Project), Entity: Neo4j (Technology)
  => Relationship: Project Atlas -[USES]-> Neo4j

  "Neo4j is supported by GraphSphere Technologies."
  => Entity: Neo4j (Technology), Entity: GraphSphere Technologies (Vendor)
  => Relationship: Neo4j -[SUPPORTED_BY]-> GraphSphere Technologies

  "Project Atlas depends on Identity Service."
  => Entity: Project Atlas (Project), Entity: Identity Service (Service)
  => Relationship: Project Atlas -[DEPENDS_ON]-> Identity Service

Negative examples (do NOT extract these):

  "The AI Platform Team is the primary consumer of Neo4j through Project Atlas."
  Do NOT extract: AI Platform Team -[USES]-> Neo4j (Team cannot USES Technology)
  Do NOT extract: AI Platform Team -[USES]-> Project Atlas (Team cannot USES Project)

  "Project Atlas uses Neo4j."
  Do NOT extract: Project Atlas -[DEPENDS_ON]-> Neo4j (Neo4j is a Technology, not a Service)

  "Alice maintains Project Atlas."
  Do NOT extract: Alice -[MANAGES]-> Project Atlas (MANAGES requires Employee -> Team)

Output format:
{{
  "entities": [
    {{"name": "Entity Name", "type": "EntityType"}}
  ],
  "relationships": [
    {{"source": "Source Entity", "type": "RELATIONSHIP_TYPE", "target": "Target Entity"}}
  ]
}}

Text:
{text}

JSON output:"""

JSON_CORRECTION_PROMPT = """The following text should be valid JSON but has errors. Fix it and return only valid JSON:

{text}

Fixed JSON:"""

def build_extraction_prompt(text: str) -> str:
    entity_types = ", ".join(t.value for t in EntityType)
    rel_types = ", ".join(t.value for t in RelationshipType)
    return EXTRACTION_PROMPT.format(
        entity_types=entity_types,
        relationship_types=rel_types,
        text=text,
    )

def validate_extraction(raw: dict) -> GraphExtraction:
    """Validate and clean raw extraction output."""
    valid_entities = []
    entity_names = set()
    
    for e in raw.get("entities", []):
        name = e.get("name", "").strip()
        etype = e.get("type", "")
        if not name:
            continue
        if not is_valid_entity_type(etype):
            logger.warning(f"Rejected invalid entity type: {etype}")
            continue
        key = (name.lower(), etype)
        if key in entity_names:
            continue
        entity_names.add(key)
        valid_entities.append(ExtractedEntity(name=name, type=EntityType(etype)))
    
    entity_name_set = {e.name.lower() for e in valid_entities}
    entity_type_by_name = {e.name.lower(): e.type for e in valid_entities}
    
    valid_rels = []
    rel_keys = set()
    for r in raw.get("relationships", []):
        source = r.get("source", "").strip()
        rtype = r.get("type", "")
        target = r.get("target", "").strip()
        if not source or not target:
            continue
        if not is_valid_relationship_type(rtype):
            logger.warning(f"Rejected invalid relationship type: {rtype}")
            continue
        if source.lower() not in entity_name_set:
            logger.warning(f"Relationship source not in entities: {source}")
            continue
        if target.lower() not in entity_name_set:
            logger.warning(f"Relationship target not in entities: {target}")
            continue
        source_type = entity_type_by_name[source.lower()]
        target_type = entity_type_by_name[target.lower()]
        rel_enum = RelationshipType(rtype)
        if not is_valid_relationship_signature(rel_enum, source_type, target_type):
            logger.warning(
                "Rejected invalid relationship signature: "
                "%s (%s) -[%s]-> %s (%s)",
                source, source_type.value, rtype, target, target_type.value,
            )
            continue
        key = (source.lower(), rtype, target.lower())
        if key in rel_keys:
            continue
        rel_keys.add(key)
        valid_rels.append(ExtractedRelationship(
            source=source, type=rel_enum, target=target
        ))
    
    return GraphExtraction(entities=valid_entities, relationships=valid_rels)

async def extract_graph_from_text(text: str, llm_client) -> GraphExtraction:
    """Extract entities and relationships from text using LLM.
    
    Raises ExtractionError if the LLM call or JSON parsing fails after retry.
    A valid LLM response with zero entities/relationships is NOT a failure.
    """
    prompt = build_extraction_prompt(text)
    
    try:
        response = await llm_client.generate_json(prompt, temperature=0)
        return validate_extraction(response)
    except Exception as e:
        logger.warning(f"First extraction attempt failed: {e}, retrying...")
        try:
            raw_response = await llm_client.generate(prompt, temperature=0)
            correction = JSON_CORRECTION_PROMPT.format(text=raw_response)
            response = await llm_client.generate_json(correction, temperature=0)
            return validate_extraction(response)
        except Exception as e2:
            logger.error(f"Graph extraction failed after retry: {e2}")
            raise ExtractionError(
                f"Graph extraction failed after retry: {e2}"
            ) from e2
