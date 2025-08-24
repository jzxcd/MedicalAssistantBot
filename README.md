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
    - 1 epoch (1,752 steps)
    - learning_rate = 1e-05
    - batch size (32 per device x 1 device x 1 accumulation)
    - ~35min training time

# add training log here


### Evaluation
Medical related answers evaluation focuses on factuality and coverage (therefore precision and recall). Two types of evaluations I was assessing:
- `Token or N-grame based`: to compare common token or N-gram between generated output to reference. 
- `Sementic based` - distance based scoring on semantics which can capture the paraphrases that N-gram based metrics. Typical ones are cosine similarity on embeddings or pretrained BERT models which have specialized heads on NLU tasks like contradictary, paraphrasing, and relevance. 

Since semantic metric requires some extra work with pretrained models and the base gemma3 performed quite poorly, I used only ROUGE-Lsum to show the improvement of the fine tuned model. ROUGE-Lsum is similar to N-gram but checks the longest common subsequence rather than fixed terms. Here is the result from the 1000 test samples



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


