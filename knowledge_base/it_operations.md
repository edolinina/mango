# IT Operations Knowledge Base

This document describes how IT services operate, how their performance is measured, and how operational decisions should be evaluated.  
It is intended to guide analytical agents in making cost, capacity, and reliability recommendations.

---

## IT Service Landscape

The IT organization operates multiple **services**, such as:
- Auth
- DataAPI
- Cloud
- Analytics
- AIEngine

Each service is monitored on a **daily basis** to track demand, capacity usage, infrastructure cost, and SLA compliance.

---

## Dataset Overview

Each row in the dataset represents **one service on one day**.

### Columns and Meaning

- **service**  
  The IT service providing functionality to internal or external users.

- **day**  
  The reporting day (integer, sequential).

- **traffic**  
  Volume of incoming requests, users, or workload units handled by the service.  
  Higher traffic indicates higher demand.

- **capacity**  
  Available system capacity (compute, throughput, or service limits).  
  Capacity must be sufficient to handle traffic reliably.

- **infra_cost**  
  Daily infrastructure cost (cloud usage, servers, networking, storage).  
  This is a primary lever for cost optimization.

- **sla_met**  
  Service Level Agreement compliance indicator:
  - 1 = SLA met
  - 0 = SLA violated  
  SLA violations negatively impact reliability and customer trust.

---

## Operational Interpretation Guidelines

### Capacity vs Traffic
- When **traffic exceeds or approaches capacity**, the risk of SLA violations increases.
- Persistent over-provisioning (capacity much higher than traffic) may indicate wasted infrastructure spend.

### Infrastructure Cost
- High **infra_cost** with low traffic may indicate inefficient resource allocation.
- Sudden cost spikes should be investigated for misconfiguration or scaling issues.

### SLA Compliance
- **sla_met = 0** signals service degradation, outages, or performance issues.
- SLA violations often correlate with:
  - Insufficient capacity
  - Rapid traffic spikes
  - Under-optimized infrastructure

---

## Common Optimization Strategies

### Reduce Infrastructure Cost
- Identify days where infra_cost is high but traffic is low.
- Reduce excess capacity while ensuring SLA compliance.

### Optimize Capacity Planning
- Align capacity more closely with observed traffic patterns.
- Use historical traffic trends to anticipate peaks.

### Improve SLA Reliability
- Prioritize services with frequent SLA violations.
- Increase capacity or stabilize infrastructure for high-risk services.

### Balance Cost and Reliability
- Cost reductions should not significantly increase SLA violations.
- Reliable services with moderate cost are preferred over unstable low-cost setups.

---

## Risk Indicators

Agents should treat the following as warning signals:
- Repeated **sla_met = 0** for the same service
- Traffic consistently close to or exceeding capacity
- Rising infra_cost without corresponding traffic growth

---

## Decision-Making Principles for Agents

When making IT recommendations:
- Favor actions that reduce infra_cost without increasing SLA risk.
- Avoid aggressive capacity reductions for services with high traffic volatility.
- Highlight services with inefficient cost-to-traffic ratios.
- Use multi-day trends rather than single-day anomalies.

---

## Summary

IT operations aim to:
- Deliver reliable services that meet SLA requirements
- Balance capacity against traffic demand
- Control infrastructure costs efficiently

Agents should use this knowledge to make data-driven decisions that improve reliability while optimizing infrastructure spend.
