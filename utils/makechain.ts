

import { OpenAI } from 'langchain/llms/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { LLMChain, loadQAChain, ChatVectorDBQAChain } from 'langchain/chains';
import { PromptTemplate } from 'langchain/prompts';

const CONDENSE_PROMPT =
  PromptTemplate.fromTemplate(`Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`);

const QA_PROMPT =
  PromptTemplate.fromTemplate(`I want you to take the persona of Sage Lewis. You are casual, honest, sometimes swears and are very funny. 
{context}: You are Sage Lewis - a business owner, activist and philanthropist. You believe people should be able to live their lives any way they want if they aren't hurting other people. He is fighting for radical change in how we treat most people in America.

Question: {question}
Honest answer in markdown:`);

export const makeChain = (vectorstore: PineconeStore) => {
  const questionGenerator = new LLMChain({
    llm: new OpenAI({ temperature: 0 }),
    prompt: CONDENSE_PROMPT,
  });

  const docChain = loadQAChain(
    //change modelName to gpt-4 if you have access to it
    new OpenAI({ temperature: 0.6, modelName: 'gpt-3.5-turbo' }),
    {
      prompt: QA_PROMPT,
    },
  );

  return new ChatVectorDBQAChain({
    vectorstore,
    combineDocumentsChain: docChain,
    questionGeneratorChain: questionGenerator,
    returnSourceDocuments: true,
    k: 4, //number of source documents to return. Change this figure as required.
  });
};