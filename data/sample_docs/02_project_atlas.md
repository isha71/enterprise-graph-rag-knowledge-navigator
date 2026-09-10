# Project Atlas — Internal Project Brief

**Document Classification:** Internal — Engineering  
**Last Updated:** August 2026  
**Owner:** AI Platform Team

## Project Summary

Project Atlas is a knowledge graph platform for enterprise intelligence, designed to unify Northstar Technologies' disparate data sources into a queryable, interconnected graph of organizational knowledge. The platform enables advanced analytics, semantic search, and multi-hop reasoning across engineering, product, and operational data.

The AI Platform Team owns Project Atlas and is accountable for its roadmap, architecture, and operational health. As the flagship initiative of the AI Platform Team, Project Atlas represents a strategic investment in Northstar's long-term data intelligence capabilities.

## Team & Ownership

Alice Morgan maintains Project Atlas, serving as the lead engineer responsible for day-to-day development, code reviews, and production stability. Alice drives the platform's core graph ingestion and query interfaces, and acts as the primary point of contact for cross-team integration requests.

Daniel Kim also works on Project Atlas, contributing primarily to the data pipeline layer and the entity resolution subsystem. Daniel's work ensures that incoming documents and structured records are correctly parsed, deduplicated, and linked within the knowledge graph.

Both Alice and Daniel operate under the direction of Bob Chen, manager of the AI Platform Team.

## Technology Stack

Project Atlas uses Neo4j as its primary graph database, leveraging Neo4j's native graph storage and Cypher query language to model complex relationships between entities such as people, teams, projects, services, and policies. The choice of Neo4j enables efficient traversal of multi-hop relationships — a core requirement for enterprise intelligence use cases.

On the application layer, Project Atlas uses FastAPI as its backend web framework. FastAPI provides high-performance asynchronous API endpoints that serve both internal consumers and the platform's interactive knowledge exploration interface. The combination of Neo4j and FastAPI allows Project Atlas to deliver sub-second query responses even on deeply nested graph traversals.

## Dependencies

Project Atlas depends on Identity Service for all authentication and authorization flows. Every API request to Project Atlas is validated against the Identity Service to ensure that only authorized users and service accounts can access the knowledge graph. This dependency is critical to maintaining data security and access control.

## Governance

Project Atlas is governed by the AI Data Governance Policy, which mandates strict controls around data ingestion, lineage tracking, model transparency, and ethical use of AI-derived insights. Compliance with this policy is reviewed quarterly, and Alice Morgan is responsible for ensuring that Project Atlas meets all governance requirements.

## Current Status

Project Atlas is currently in its second major release cycle, with active development focused on expanding entity extraction capabilities, improving query performance at scale, and integrating additional data sources from the Payments Platform Team and Core Services Team. The project maintains a biweekly release cadence with automated regression testing and staged rollouts.
