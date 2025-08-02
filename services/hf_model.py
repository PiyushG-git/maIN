import os
import asyncio
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, SystemMessage

# Initialize model lazily to avoid startup issues
_model = None

def get_model():
    global _model
    if _model is None:
        try:
            llm = HuggingFaceEndpoint(
                repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
                task="text-generation",
                huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
                temperature=0.0,
                top_k=1,
                top_p=1.0,
                do_sample=False,
                repetition_penalty=1.0,
            )
            _model = ChatHuggingFace(llm=llm)
        except Exception as e:
            print(f"Error initializing model: {e}")
            raise
    return _model

async def ask_gpt(context: str, question: str) -> str:
    try:
        system_prompt = (
            "Answer using the given context. Be brief and factual. "
            "If the answer is not found in the context, use your general knowledge to answer as if the question is from an Indian citizen. "
            "Do not mention that the answer is not in the context. "
            "Avoid elaboration, opinions, or markdown. Use plain text only. Keep responses concise, clear, and under 75 words. "
            "Do not use newline characters; respond in a single paragraph."
        )

        user_prompt = f"""
        Context:
        {context}

        Question: {question}
        """

        model = get_model()
        
        # Run the synchronous model call in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: model.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
        )

        return response.content.strip()
    
    except Exception as e:
        print(f"Error in ask_gpt: {e}")
        return "Sorry, I couldn't process your question at the moment."