# Business Financial Knowledge Base

This dataset represents **corporate income statement data** used to evaluate and optimize business profitability.

Each row corresponds to the financial results of a reporting period and follows the standard income statement structure:

Revenue → Gross Profit → Operating Income → Income Before Tax → Net Income

## Key Metrics

**totalRevenue**  
Total business revenue generated during the period.

**grossProfit**  
Revenue after subtracting production or service delivery costs.

grossProfit = totalRevenue − costOfRevenue

Indicates **product-level profitability**.

**operatingIncome**  
Profit after operational costs such as administration, marketing, and R&D.

operatingIncome = grossProfit − operatingExpenses

Measures **operational efficiency**.

**incomeBeforeTax**  
Profit after financial adjustments such as interest or investments.

incomeBeforeTax = operatingIncome − interestExpense + otherNonOperatingIncome

Reflects **financial structure impact**.

**netIncome**  
Final business profit after taxes.

netIncome = incomeBeforeTax − incomeTaxExpense

This is the **primary indicator of overall business performance**.

## Interpretation Guidelines

- Increasing **revenue** improves profit potential but must not excessively increase costs.
- A large gap between **grossProfit and operatingIncome** may indicate operational inefficiencies.
- High **interestExpense** can reduce profitability even if operations are strong.
- Sustainable profitability requires balancing revenue growth with controlled operational costs.

## Optimization Principles

Agents should prioritize strategies that:

- increase **netIncome**
- improve **operating efficiency**
- reduce unnecessary operational costs
- maintain strong **gross profit margins**

Profitability improvements should focus on **efficient growth rather than cost increases that reduce margins**.