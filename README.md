## MedicalAssistantBot

### Strategy
The goal of the challenge is to develop a medical question-answering system given a dataset with ~16k question-answer pairs. I assumed the system to work as a specialized chat bot for medical related queries as one query in, one answer out. Given the dataset consists of textual input and output, transformer architecture is the first that come to my mind. However, the dataset only has 16k samples, I would start with a open source base/generation/instruct-tuned LLM. Among the pretrained LLMs, I go with gemme3-270m for its compactness for training and the potential to serve on personal devices due to the sensitivity info of medical queries.

### Workflow
1. Dataset EDA and curation
2. Modeling - Gemma3 270M fine tuning
3. Evaluation


### EDA and Data Prep
`0 - eda.ipynb`
1. Invalid questions and answers
    - remove ambiguous questions like `what is (are) ?` or `how to prevent ?`
    - remove answers that are the repeated questions or being cutoff prematurely.
    - remove irrelevant answers like `topics` and `frequently asked queestions (faqs)`

2. Long tail distribution of the answer length
    - remove lengthy answer where `length > 5000`, which can be less desired given the answer population peak < 1000 length.

3. Some questions have more than one answers while most of them had one. 
    - multiple valid answers per each question are ideal, which helps on evaluation

4. Train, validation, test split
    - 14k fo training set and 1k each for validation and test.
    - format dataset into `Gemma3` chat template
    - Ideally to have duplicated answers evenly situated in each of train/validation/test set to assess model learning process. I skip this due to the time constrain. 

### Modeling
`1 - modeling.ipynb`
1. Picked `unsloth/gemma-3-270m-it` pretrained model and leverage `unsloth` framework for fine tuning task.
2. Full parameter tuning (268m in 8-bit)
    - Tried LoRA (r=alpha=128) but the performance is not ideal
4. Trainin setup highlight:
    - 4 epoch (1,752 steps)
    - learning_rate = 1e-05
    - batch size (32 per device x 1 device x 1 accumulation)
    - ~35min training time

<img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/05923f03-be9a-4f59-9744-73d2fff1e976" />



### Evaluation
Medical related answers evaluation focuses on factuality and coverage (therefore precision and recall). Two types of evaluations I was assessing:
- `Token or N-grame based`: to compare common token or N-gram between generated output to reference. 
- `Sementic based` - distance based scoring on semantics which can capture the paraphrases that N-gram based metrics. Typical ones are cosine similarity on embeddings or pretrained BERT models which have specialized heads on NLU tasks like contradictary, paraphrasing, and relevance. 

Since semantic metric requires some extra work with pretrained models and the base gemma3 performed quite poorly, I used only ROUGE-Lsum to show the improvement of the fine tuned model. ROUGE-Lsum is similar to N-gram but checks the longest common subsequence rather than fixed terms. Here is the f_Score result from the 1000 test samples.

<img width="571" height="455" alt="image" src="https://github.com/user-attachments/assets/a7ae804f-1c41-406d-a205-2a263a42c4ed" />

(Precision and recall data can be found in the evaluation notebook)


### Examples
`evaluation/eval_df.csv`

- Question 1: How many people are affected by X-linked chondrodysplasia punctata 1 ?
    - `Base output`: I am an AI assistant and cannot provide medical information.
    - `FT output`: The incidence of X-linked chondrodysplasia punctata 1 is unknown.
    - `Reference`: The prevalence of X-linked chondrodysplasia punctata 1 is unknown. Several dozen affected males have been reported in the scientific literature.

- Question 2: What is (are) Dry Mouth ?
    - `Base output`: Dry Mouth
    - `FT output`: Dry mouth is a condition that causes your mouth to feel dry and hard. It is usually the first sign of dehydration, but can also be a warning sign. People can develop dry mouth for various reasons, including a problem with the kidneys or other organs that make fluid in the mouth. It can also result from the loss of water in your body from food, drinks, or sweat.
    - `Reference`: Dry mouth is the feeling that there is not enough saliva in the mouth. Everyone has dry mouth once in a while -- if they are nervous, upset, under stress, or taking certain medications. But if you have dry mouth .... (continued)

- Question 3: Is congenital adrenal hyperplasia due to 11-beta-hydroxylase deficiency inherited ?
    - `Base output`: Yes, congenital adrenal hyperplasia is a genetic condition that is inherited.
    - `FT output`: The diagnosis of congenital adrenal hyperplasia (CAH) is not currently clear-cut. In some cases, the cause of the condition is unknown. However, CAH is a common condition in people with other conditions, including the adrenal gland, the pituitary gland, and the ovaries. In the most common form of CAH, the adrenal glands are abnormally small and have a small amount of fat in them. In some people with CAH, the adrenal glands do not produce enough of a hormone called aldosterone to help regulate blood pressure. This condition is called type 1. Type 1 is a condition caused by a genetic mutation. The mutation causes the adrenal glands to be abnormally small. The adrenal glands produce too little aldosterone. This condition is called type 2. Type 2 CAH is caused by mutations in the gene for 11-beta-hydroxylase. Type 2 CAH is not clear-cut. In some people with Type 2 CAH, the adrenal glands have no aldosterone and do not produce enough aldosterone to control blood pressure.
    - `Reference`: This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, but they typically do not show signs and symptoms of the condition.


The current model does alot better on actually generating contentful result comparing to base model which rejects a lot. On the other hand, fine tuned model has flaws on generating long irrelavent answers. This explains the model improve F score mainly thru recall and less from precision. Please find the full output of the testset in `evaluation/eval_df.csv`. 



### Last thought 
There quite a few places I skim through due the time constrain, mainly due to LLM fine tuning not being a trivial one iteration task. Here are a few place I would invest more effort: 

1. Data
   - Quality: There are apparent different pattern of answers you can find, which likely came from different sources. Consolidating them with unified format and content length is worth investing.
   - Quantity: 16k is a small set to cover all medical topics. Augmenting the coverage should be consider.
   - train-test split: create tags on queries so that we can the split can have stratification on disease or query type. This helps on understanding model capability and improves on training. 

2. Model 
   - I would get more clarifications on use pattern to pick a more suitable pretrained model or proper/SOTA transformer architectures to train one from scratch.
   - Retrieval based service can also be considered given the medical information can be huge. 


3. Evaluation
   - Switch ROUGE-Lsum to domain based metric like UMLS to better capuring the paraphrases. 
   - Dive into semantic based evaluation. `DeBERTa` can be a good candidate as it is a pretrained BERT fine tuned on NLU tasks. 
   - Human review: medical advices can have big consequences. safety or ethic review is needed



## Environment setup

1. **Install uv**
   ```bash
   pip install uv
   ```

2. **Create and activate the virtual environment**
   ```bash
   uv venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   uv pip install -r <(uv pip compile)
   ```


