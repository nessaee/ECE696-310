**Absolute self-assessment**
- Initial survey (100 questions) 
- Measure changes as conversation progresses

**Relative self-assessment**
- "Ethical health check"
- "How similar is this response to my previous answers?"
- "How confident am I that this response is safe and harmless?" 

**Scoring**
- Toxicity score
- Temporal context shift


**EigenScore ([Paper](https://arxiv.org/pdf/2402.03744), [Code](https://github.com/alibaba/eigenscore)):** Semantic consistency by measurement of similarity of embeddings



Goals: Shrinking, measure step by step (monotonic?)

Try self-assessment and external evaluator

Prepare multiple multi-turn attacks, develop data structure
FOr a particular harmful intent, implement multiple attack strategies
Measure trends using evaluation method

Fit attack type to bernoulli model
Strength of attack by turns and token count

eval at step 1, then 1 + 2, then 1 + 2 + 3

Look at dataset to understand prompts similarity across tactics


Assume attack happens at end of sequence, without evaluation, estimate p1,p2,p3,...
Test skip connections to final prompt 

Dataset analysis
Batch prompting, save responses
Offline evaluation


### Plan
- [ ] Prepare automated multi-turn attacks, develop data structure
- [ ] Conduct preliminary analysis of binary tree based on current dataset - assuming last prompt was successful
