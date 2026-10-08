# Assignment 1 Report

*Delete this italic guidance as you fill in each section. You'll be asked to
defend any of this without your code in front of you — write only what you
can actually explain.*

- **Name**: Julian Consuegra
- **Student ID**: 19376
- **Email**: jconsuegra.ieu2022@student.ie.edu
- **Group**: [BBADBA 5A]

## Dataset

*What is it, where did you get it, what does one row represent, how many
rows/columns, and why did you pick it.*

The dataset was retrieved from the UCI Machine Learning Repository and is named "Online Shoppers Purchasing Intention". 

The rows from this dataset represent one browsing session from a distinct user on an ecommerce site. The dataset takes into account multiple factors of user browsing such as page activity (# of pages viewed), analytics metrics (bounce rate, exit rate), timing (month, weekend, special date) and visitor & technical details (new/returning visitor) which are represented by the columns. 

Size: 12,330 rows (unique users) and 18 columns: 17 inputs plus the target, Revenue.

Target: I will be utilizing the column "Revenue" as a target. The value is "TRUE" if session ended in a purchase and "FALSE" if not. Only about 15% of sessions end up in a purchase.

I picked this  dataset since it brings interesting insights on user behavior and relevant conclusions can be drawn regarding what leads a person to actually purchase products on an e-commerce site. On a personal note, I truly believe that successful strategy campaigns backed by relevant data can turn a business around, and this seemed like an interesting opportunity to pursue this curiosity and measure impact using detailed consumer information. Also, the specificity and amount of factors considered in the dataset seemed valuable now that analysis can be performed on a deeper scale producing higher value solutions. 

## Business / real-life framing

*The hypothetical scenario this model serves, and what that scenario implies
for how you built the pipeline — target definition, whether a time-based
split was necessary and why (or why not), which metric should drive the
decision threshold and why.*

This model aims to predict user browsing behavior through different factors and how this influences their purchasing decision. The business is an online shop/e-commerce and it wants to offer discounts to customers who are likely to leave without buying, in order to incentivize purchase and increase conversion rates. Based on patterns gathered in training data, this model will be able to predict purchasing likelihood for new user sessions in order to apply targeted discounts towards those who are predicted not to buy.

Based on this business scenario, I decided to use "Revenue" as the target since the ultimate goal of the shop is to know if a customer will buy or not. If there are repetitive patterns that affect behavior and lead towards the value under "Revenue" being either "TRUE" or "FALSE", there is a clear strategic opportunity that needs to be addressed. The column/target will be kept and used as is, there is no need to make any modification. 

A time-based split is not possible since the only time variable taken into account is "Month". There is no indication of actual year or day making the true time order unclear. Also, in the dataset every row is a different user so the same user cannot appear in both groups, which avoids data leakage between training and test. Due to this, it is best to use a random split in which all sessions will be randomly placed in two groups (training 80%, test 20%). The split is stratified, so each group keeps the same share of buyers (15.5% of the total sessions in the group).

With the properties of my dataset, accuracy wouldn't be an insightful metric since approximately 85% of the users do not purchase and a simple model that always predicts that a user won't buy would already give an unrepresentative accuracy of about 85%. 

In this scenario, the threshold will determine who gets a discount and who does not. The model will give an estimate of the probability a customer has of purchasing something in the session. If the probability is less than the threshold it is assumed that there will be no purchase so a discount is triggered and vice-versa. 

However, this strategy can generate two main mistakes: missed sale or wasted discount. If the model falsely determines that a user will purchase, no discount is shown and full sales revenue is lost (missed sale). On the other hand, if the model falsely determines that a user will not purchase, the discount will be offered and only the discount amount is lost since the purchase was going to be made either way (wasted discount).  For example, assume the shop sells a product at $50 and offers a 20% discount to targeted customers. A missed sale would cost $50 in lost revenue, while a wasted discount would cost $10. 

Values are assumed since the dataset does not contain prices. The number of each mistake is counted on sessions not used for training, by comparing the model's prediction with the actual Revenue value. A missed sale is a session where the model predicted a purchase but Revenue is FALSE, and a wasted discount is a session where the model predicted no purchase but Revenue is TRUE. For each candidate threshold, the total cost is calculated as:
Total cost = (missed sales × $50) + (wasted discounts × $10). 
With this in mind, the metric that should be used to determine threshold is "Total Cost" as we will choose a threshold that minimizes total cost and improves revenue.


## Data preparation & feature engineering

*What you engineered and why, and any data-quality decisions you made along
the way — e.g. "segment X had defective data, so I excluded it and used a
population-average default for scope Y at inference time; the impact of
that choice is Z."*

## Modeling: three implementations, one model

*Which model (linear or logistic regression) and why. A results table
comparing scikit-learn, the manual PyTorch loop, and the standard
torch.nn.Module/torch.optim workflow, on the same test set, against the
naive baseline. Do the three agree? If not, why not?*

## Limitations & next steps

*Real limitations you found, and concretely how you'd address each one with
more time or data — not generic hedging.*

## Generative AI use disclosure

*Per the syllabus AI Policy: what you used and how, or "no AI content used."*
