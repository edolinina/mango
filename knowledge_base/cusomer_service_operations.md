# Customer Loyalty and Churn Knowledge Base

This dataset represents **customer usage, billing behavior, and support interactions** used to predict and reduce **customer churn**.

Each row represents one customer and their service usage profile.

The goal is to identify patterns that indicate whether a customer is **likely to churn (leave the service)**.

## Key Metrics

**tenure_months**  
Number of months the customer has been using the service.  
Longer tenure generally indicates stronger loyalty.

**monthly_charges**  
The monthly cost paid by the customer for the service plan.  
Very high charges without perceived value may increase churn risk.

**daily_usage_hours**  
Average hours the customer uses the service per day.  
Higher usage usually indicates stronger engagement.

**monthly_logins**  
Number of times a customer logs into the platform per month.  
Frequent logins indicate active engagement.

**features_used**  
Number of product features used by the customer.  
Customers using more features are typically more invested in the service.

**support_tickets**  
Number of support requests submitted by the customer.  
High ticket volume may indicate product issues or user frustration.

**complaint_count**  
Number of formal complaints filed by the customer.  
High complaint levels strongly correlate with churn.

**payment_delays**  
Number of late payments.  
Frequent delays may indicate dissatisfaction or financial friction.

**autopay_enabled**  
Indicates whether automatic billing is enabled.  
Autopay customers typically churn less often.

**satisfaction_score**  
Customer satisfaction rating based on surveys or feedback.

**churn**  
Target variable indicating whether the customer left the service.

- 0 = customer stayed  
- 1 = customer churned  

## Interpretation Guidelines

Customer churn risk typically increases when:

- complaints or support tickets are high  
- product usage decreases  
- payment delays increase  
- satisfaction scores drop  
- customers use few product features  

Customer retention improves when customers:

- actively use the product  
- log in frequently  
- use multiple features  
- have long tenure with the service  

## Optimization Principles

Agents should prioritize strategies that:

- reduce **complaints and support issues**
- increase **product engagement and feature adoption**
- improve **customer satisfaction**
- reduce **payment friction**

Retention strategies should focus on **improving customer experience and product value**, which naturally reduces churn.