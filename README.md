# KA Chatbot
This is a personal POC (proof-of-concept) project to create AI chatbot that can basically answer the question about some KrungsriAuto products such as Car4Cash, New Car and Bigbike.

<ins> **All data used in this project is scraped from KrungsriAuto webpages where all contents are publicly shared.** </ins>

## Use Case
This project aims to create a chatbot that can come up with the correct answer for customers’ questions on a specific domain.
E.g. Car4Cash, Installment Amount for a given car

## Process
To reach our goal, There are 2 main roads (concepts) that we can go.
1. Fine-Tuning GPT : Training GPT with well-prepared data.
2. Knowledge Base : Embed our data as vectors and keep it in vectorstore (chromadb) and recall it when there is a user's query. (vectorstore would do semantic search)
This case I decided to go for the second way because Fine-tuning required huge computational cost and takes a lot of time to prepared well-crafted data to fine-tune GPT.

The main code consists of 2 parts
1. Web scraping (WebScraping_ForLLM.ipynb) : This one is the Jupyter Notebook that contains code which I used to scrape information from Krungsri webpages and preprocessing it (split into chunks, then embed into vectors).
2. Chatbot (ChatBotApp.py) : This one is a python code that contains functions that create Chatbot (using chatgpt-3.5-turbo as the core LLM) ,which has an ability to retrieve some specific information collected in vectorstore (created in 1).
   That retrieved information would be included in a prompt that would be sent to LLM (GPT) model. Basically, the concept here that i am trying to work on is Retrieval-Augmented Generation (RAG).

Roughly what I tried to do in the code is just the same as this workflow shows below.

<img width="638" alt="screenshot1" src="https://github.com/Jui-Phongsathorn/KAChatbot/assets/112532175/d9e43612-82e8-48d6-8285-c7a4eab2580f">

## Result
Empirically, the KA Chatbot can answer some specific product-related questions (as shown in result.pptx) better than the general well-known free chatgpt-3.5. <check the result on the pptx file>
However, It can do better only if the question's context is related to what have been covered in Krungsri's Articles, which are scraped. For the question which is out-of-data-scoped ( ask about the context which is not related to what is in the vectorstore), the KA Chatbot does not give a satisfied answer. 


## Further Discussion
There are still many ways to improve this Chatbot's performance and I would do it in a foreseeable future.
1. Memory Configuration : KA chatbot does not seem to be able to remember the previous prompts when we talk to it in a chat, although in the code, I have already included the memory in langchain.ConversationalRetrievalChain(). The problem may occured because I use ConversationalRetrievalChain(). I plan to create my own custom chain that works similar to ConversationalRetrievalChain() but i would include memory in the right place.
2. Add more related documents in vectorstore: gather more articles and useful information from website.
3. Reinforcement Learning with Human Feedback (RLHF) : Fine-Tuning the whole model using RLHF to get rid of all unpleasant answers.

## NOTE:
To run code ChatBotApp.py and make it work, you need to have your own openai api key which was not provided in this repo!

