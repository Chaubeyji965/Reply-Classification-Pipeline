# Reasoning Answers

## Question 1: Limited Data Improvement
**If you only had 200 labeled replies, how would you improve the model without collecting thousands more?**

With only 200 labeled samples, I would use data augmentation techniques like paraphrasing and synonym replacement to expand the training set. Additionally, I would leverage pre-trained embeddings like Word2Vec or GloVe to better capture semantic relationships, and apply transfer learning from models trained on similar sentiment analysis tasks to bootstrap performance.

## Question 2: Bias and Safety Prevention
**How would you ensure your reply classifier doesn't produce biased or unsafe outputs in production?**

I would implement comprehensive testing across different demographic groups and communication styles, establish confidence thresholds to flag uncertain predictions for human review, and create monitoring dashboards to track prediction distributions over time. Regular bias audits and diverse training data collection would help identify and mitigate systematic biases before they impact business decisions.

## Question 3: Personalized Cold Email Generation
**Suppose you want to generate personalized cold email openers using an LLM. What prompt design strategies would you use to keep outputs relevant and non-generic?**

I would use few-shot prompting with specific examples of successful openers, incorporate company-specific context and recent news into the prompt, and implement constraint-based generation with clear formatting requirements. Additionally, I would use chain-of-thought prompting to have the model reason about the recipient's likely interests and pain points before generating the opener.
