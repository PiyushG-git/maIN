# import os
# import asyncio
# from langchain_huggingface import HuggingFaceEndpointEmbeddings
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from langchain_core.messages import HumanMessage, SystemMessage

# # Initialize model lazily to avoid startup issues
# _model = None

# def get_model():
#     global _model
#     if _model is None:
#         try:
#             llm = HuggingFaceEndpoint(
#                 repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
#                 task="text-generation",
#                 huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
#                 temperature=0.0,
#                 top_k=1,
#                 top_p=1.0,
#                 do_sample=False,
#                 repetition_penalty=1.0,
#             )
#             _model = ChatHuggingFace(llm=llm)
#         except Exception as e:
#             print(f"Error initializing model: {e}")
#             raise
#     return _model

# async def ask_gpt(context: str, question: str) -> str:
#     try:
#         system_prompt = (
#             "Answer using the given context. Be brief and factual. "
#             "If the answer is not found in the context, use your general knowledge to answer as if the question is from an Indian citizen. "
#             "Do not mention that the answer is not in the context. "
#             "Avoid elaboration, opinions, or markdown. Use plain text only. Keep responses concise, clear, and under 75 words. "
#             "Do not use newline characters; respond in a single paragraph."
#         )

#         user_prompt = f"""
#         Context:
#         {context}

#         Question: {question}
#         """

#         model = get_model()
        
#         # Run the synchronous model call in a thread pool to avoid blocking
#         loop = asyncio.get_event_loop()
#         response = await loop.run_in_executor(
#             None,
#             lambda: model.invoke([
#                 SystemMessage(content=system_prompt),
#                 HumanMessage(content=user_prompt)
#             ])
#         )

#         return response.content.strip()
    
#     except Exception as e:
#         print(f"Error in ask_gpt: {e}")
#         return "Sorry, I couldn't process your question at the moment."


import os
import asyncio
from typing import Optional
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, SystemMessage
from config.settings import settings

# Initialize model lazily to avoid startup issues
_model = None
_model_lock = asyncio.Lock()

async def get_model():
    """Get or initialize the model with thread safety"""
    global _model
    if _model is None:
        async with _model_lock:
            if _model is None:  # Double-check pattern
                try:
                    llm = HuggingFaceEndpoint(
                        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
                        task="text-generation",
                        huggingfacehub_api_token=settings.HUGGINGFACEHUB_ACCESS_TOKEN,
                        temperature=0.1,  # Slightly increased for better responses
                        max_new_tokens=256,  # Reduced for faster responses
                        top_k=10,
                        top_p=0.9,
                        do_sample=True,
                        repetition_penalty=1.1,
                        timeout=60,  # Add timeout
                    )
                    _model = ChatHuggingFace(llm=llm, verbose=False)
                    print("✅ Model initialized successfully")
                except Exception as e:
                    print(f"❌ Error initializing model: {e}")
                    raise Exception(f"Failed to initialize language model: {str(e)}")
    return _model

async def ask_gpt(context: str, question: str) -> str:
    """
    Ask the language model a question given context
    """
    if not context.strip():
        return "I don't have enough context to answer this question."
    
    if not question.strip():
        return "Please provide a valid question."
    
    try:
        # Truncate context if too long to avoid token limits
        max_context_length = 2000
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."
            
        system_prompt = (
            "You are a helpful AI assistant. Answer the user's question based on the provided context. "
            "Be concise, accurate, and factual. If the context doesn't contain enough information "
            "to fully answer the question, provide the best answer you can based on what's available "
            "and mention that additional context might be needed. "
            "Keep your response under 100 words and avoid using markdown formatting."
        )

        user_prompt = f"""Context: {context.strip()}

Question: {question.strip()}

Answer:"""

        model = await get_model()
        
        # Create messages
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        # Run the model call with timeout
        try:
            response = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: model.invoke(messages)
                ),
                timeout=45.0  # 45 second timeout
            )
            
            if hasattr(response, 'content'):
                answer = response.content.strip()
            else:
                answer = str(response).strip()
                
            # Clean up the response
            answer = answer.replace('\n\n', ' ').replace('\n', ' ')
            
            # Ensure the answer is not too long
            if len(answer) > 500:
                answer = answer[:500] + "..."
                
            return answer if answer else "I couldn't generate a proper response to your question."
            
        except asyncio.TimeoutError:
            print("⚠️ Model response timeout")
            return "I'm having trouble processing your question right now. Please try again."
            
    except Exception as e:
        print(f"❌ Error in ask_gpt: {e}")
        
        # Fallback response based on question type
        question_lower = question.lower()
        if any(word in question_lower for word in ['what', 'define', 'explain']):
            return "I'm unable to provide a detailed explanation at the moment. Please try rephrasing your question."
        elif any(word in question_lower for word in ['how', 'steps', 'process']):
            return "I'm unable to provide step-by-step instructions at the moment. Please try again later."
        elif any(word in question_lower for word in ['when', 'time', 'date']):
            return "I'm unable to provide timing information at the moment. Please check the document directly."
        else:
            return "I'm having trouble answering your question right now. Please try again or rephrase your question."
