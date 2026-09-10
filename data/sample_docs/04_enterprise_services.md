# Enterprise Services Catalog

**Document Classification:** Internal — Platform Engineering  
**Last Updated:** August 2026  
**Owner:** Core Services Team & Payments Platform Team

## Overview

This document provides a reference catalog of the shared enterprise services maintained within Northstar Technologies' Engineering Department. These services form the foundational infrastructure that product teams depend on for authentication, identity management, and fraud prevention capabilities.

## Identity Service

The Core Services Team maintains Identity Service, which serves as the centralized identity and access management platform for Northstar Technologies. Identity Service provides SSO and OAuth capabilities, enabling seamless single sign-on experiences across all internal applications and supporting OAuth 2.0 flows for third-party integrations and API consumers.

Identity Service is consumed by virtually every application and platform within Northstar, including Project Atlas, internal dashboards, and developer tooling. It exposes a well-documented RESTful API for user authentication, token introspection, and role-based access control lookups.

Identity Service depends on Authentication Gateway for all low-level token operations, session management, and protocol translation. This architectural separation ensures that Identity Service can focus on business-level identity logic while delegating the cryptographic and protocol-specific concerns to the Authentication Gateway.

### Authentication Gateway

Authentication Gateway is a high-performance service that handles token validation for all inbound requests across Northstar's platform. It sits at the network edge and intercepts every authenticated API call, verifying JSON Web Tokens, checking token expiration, and enforcing scope-based access policies before requests reach downstream services.

Authentication Gateway uses Redis as its primary data store for session state, token caches, and rate-limiting counters. Redis's in-memory architecture allows the Authentication Gateway to perform token validation with sub-millisecond latency, which is essential given the service's position on the critical path of every authenticated request.

The gateway is maintained by the Core Services Team and is deployed in a multi-region configuration to ensure global availability.

## Fraud Detection Service

The Payments Platform Team maintains Fraud Detection Service, which provides real-time risk scoring and anomaly detection for financial transactions processed by Project Ledger and other payment systems. The service analyzes transaction patterns, device fingerprints, and behavioral signals to assign risk scores and flag potentially fraudulent activity.

Fraud Detection Service uses Python as its primary implementation language, leveraging Python's rich ecosystem of machine learning and data science libraries for its risk models. The service's scoring engine is built on scikit-learn and custom rule-based classifiers that are retrained on a weekly cadence.

Fraud Detection Service depends on Identity Service to verify the identity of users and service accounts initiating transactions. This dependency ensures that fraud risk assessments are always performed in the context of a verified identity, preventing spoofed or unauthenticated requests from bypassing fraud controls.

## Service Interdependencies

The services described in this catalog are tightly interconnected. Identity Service and Authentication Gateway form the core authentication stack, while Fraud Detection Service bridges the identity and payments domains. Teams consuming these services should consult the relevant service owners before making integration changes.
