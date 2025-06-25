from waitress import serve
from flask import Flask, request, jsonify, session
from pymongo import MongoClient
from bson.objectid import ObjectId
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from apscheduler.schedulers.background import BackgroundScheduler
import os
import requests
import uuid
import json
import math
from flask import g
from difflib import SequenceMatcher
import re
import google.generativeai as genai
from langchain.llms.base import LLM
from typing import Optional, List, Any
import logging

from datetime import datetime
import pymongo

 


app = Flask(__name__)
app.secret_key = os.urandom(24)   

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import os
from dotenv import load_dotenv

load_dotenv()  

mongo_url = os.getenv("MONGODB_URI")
client = MongoClient(mongo_url)

db = client["ta7t-bety"]
users_collection = db["users"]
services_collection = db["chatbot_services"]
knowledge_collection = db["knowledge_base"]

user_sessions = {}
current_session_id = None

data_file = r"C:\Users\mostafa\Documents\GitHub\ta7t-bety-chatbot\all_provider_data.txt"
fetch_time_file = r"C:\Users\mostafa\Documents\GitHub\ta7t-bety-chatbot\.last_fetch.txt"

genai.configure(api_key="AIzaSyCl8kwTf_-H4jSAfg6EAhlHDn0NU28iX60")

class GeminiLLM(LLM):
    model_name: str = "gemini-1.5-flash"
    temperature: float = 0.0
    max_tokens: int = 1000
    
    @property
    def _llm_type(self) -> str:
        return "gemini"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        try:
            model = genai.GenerativeModel(self.model_name)
            generation_config = genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            )
            
            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"

def is_greeting_or_thanks(message):
    message_lower = message.lower().strip()
    
    greetings = [
        'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening',
        'أهلا', 'هاي', 'السلام عليكم', 'صباح الخير', 'مساء الخير', 'مرحبا'
    ]
    
    thanks = [
        'thanks', 'thank you', 'thx', 'ty', 'thanks a lot', 'thank you so much',
        'شكرا', 'شكراً', 'تسلم', 'جزاك الله خيرا', 'ربنا يكرمك', 'الله يعطيك العافية'
    ]
    
    for greeting in greetings:
        if greeting in message_lower:
            return 'greeting'
    
    for thank in thanks:
        if thank in message_lower:
            return 'thanks'
    
    return None

def get_greeting_response(lang='en', user_name=''):
    responses = {
        'en': [
            f"Hello {user_name}! 😊 How can I help you today?",
            f"Hi there {user_name}! 👋 What can I do for you?",
            f"Hey {user_name}! 🌟 Ready to find some great food?"
        ],
        'ar': [
            f"أهلاً {user_name}! 😊 إزاي ممكن أساعدك النهارده؟",
            f"هاي {user_name}! 👋 إيه اللي تحب أعمله لك؟",
            f"مرحباً {user_name}! 🌟 جاهز نلاقي أكل حلو؟"
        ]
    }
    
    import random
    return random.choice(responses.get(lang, responses['en']))

def get_thanks_response(lang='en'):
    responses = {
        'en': [
            "You're very welcome! 😊 Anything else I can help with?",
            "Happy to help! 🌟 Let me know if you need anything else!",
            "No problem at all! 👍 Feel free to ask more questions!"
        ],
        'ar': [
            "العفو! 😊 حاجة تانية ممكن أساعدك فيها؟",
            "أهلاً وسهلاً! 🌟 قولي لو محتاج أي حاجة تانية!",
            "ولا يهمك! 👍 اسأل براحتك أي سؤال تاني!"
        ]
    }
    
    import random
    return random.choice(responses.get(lang, responses['en']))\
    

def is_order_request(message):
    message_lower = message.lower().strip()
    
    order_keywords = [
        'order', 'طلب', 'اطلب', 'عايز اطلب', 'want to order' ,
        'بدي اطلب' , 'delivery', 'توصيل', 'اطلبلي' ,"عايز اعمل اوردر","عايز احجز اوردر" , "اوردر"
    ]
    
    for keyword in order_keywords:
        if keyword in message_lower:
            return True
    return False
    

def add_food_emojis(text):
    emoji_map = {
        'لحم': '🥩', 'لحمة': '🥩', 'لحوم': '🥩',
        'فراخ': '🍗', 'فرخة': '🍗', 'دجاج': '🍗', 'فرح': '🍗',
        'beef': '🥩', 'meat': '🥩', 'chicken': '🍗',
        'بطاطس': '🍟', 'بطاطا': '🍟', 'potato': '🍟', 'فريز': '🍟', 'fries': '🍟',
        'بيبسي': '🥤', 'كولا': '🥤', 'كوكا': '🥤', 'مشروب': '🥤', 'مشروبات': '🥤',
        'pepsi': '🥤', 'cola': '🥤', 'coke': '🥤', 'drink': '🥤',
        'عصير': '🧃', 'عصائر': '🧃', 'juice': '🧃',
        'مكرونة': '🍝', 'معكرونة': '🍝', 'باستا': '🍝', 'pasta': '🍝',
        'أرز': '🍚', 'رز': '🍚', 'rice': '🍚',
        'حلويات': '🍰', 'حلو': '🍰', 'حلى': '🍰', 'كيك': '🍰', 'كيكة': '🍰',
        'dessert': '🍰', 'cake': '🍰', 'sweet': '🍰', 'sweets': '🍰',
        'دونات': '🍩', 'دوناتس': '🍩', 'donut': '🍩', 'donuts': '🍩',
        'آيس كريم': '🍦', 'ايس كريم': '🍦', 'جيلاتي': '🍦', 'جلاتي': '🍦',
        'ice cream': '🍦', 'gelato': '🍦',
        'فواكه': '🍇', 'فاكهة': '🍇', 'فواكة': '🍇', 'fruit': '🍇', 'fruits': '🍇',
        'برجر': '🍔', 'برغر': '🍔', 'همبرجر': '🍔', 'ساندوتش': '🥪', 'ساندويش': '🥪',
        'burger': '🍔', 'hamburger': '🍔', 'sandwich': '🥪',
        'تاكو': '🌮', 'تاكوس': '🌮', 'taco': '🌮', 'tacos': '🌮',
        'بيتزا': '🍕', 'بيزا': '🍕', 'pizza': '🍕',
        'آسيوي': '🍱', 'اسيوي': '🍱', 'سوشي': '🍱', 'سوشى': '🍱',
        'asian': '🍱', 'sushi': '🍱', 'chinese': '🍱', 'japanese': '🍱',
        'كشري': '🍛', 'كشرى': '🍛', 'ملوخية': '🍲', 'فول': '🫘', 'طعمية': '🧆', 'فلافل': '🧆',
        'شاورما': '🌯', 'كباب': '🍖', 'كفتة': '🍖', 'مشوي': '🍖', 'مشاوي': '🍖',
        'محشي': '🫑', 'ورق عنب': '🍃', 'رقاق': '🫓', 'عيش': '🍞', 'خبز': '🍞',
        'شاي': '🍵', 'قهوة': '☕', 'نسكافيه': '☕', 'كابتشينو': '☕',
        'tea': '🍵', 'coffee': '☕', 'cappuccino': '☕', 'latte': '☕'
    }
    
    result = text
    for food, emoji in emoji_map.items():
        pattern = r'\b' + re.escape(food) + r'\b'
        result = re.sub(pattern, f'{emoji} {food}', result, flags=re.IGNORECASE)
    
    return result

def format_menu_with_stars(text):
    lines = text.split('\n')
    formatted_lines = []
    
    for line in lines:
        if ',' in line or ' و ' in line or ' and ' in line:
            items = re.split(r'[,،]|\s+و\s+|\s+and\s+', line)
            if len(items) > 1:
                formatted_line = ' ⭐ '.join(item.strip() for item in items if item.strip())
                formatted_lines.append(formatted_line)
            else:
                formatted_lines.append(line)
        else:
            formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)

def get_last_fetch_time():
    if os.path.exists(fetch_time_file):
        with open(fetch_time_file, "r") as f:
            return f.read().strip()
    return None

def set_last_fetch_time(oid_str):
    with open(fetch_time_file, "w") as f:
        f.write(oid_str)

def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371 
    
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c

def create_consolidated_data_file(session_id, user_lat, user_lng, radius=1000):
    try:
        api_url = f"https://ta7t-bety.vercel.app/api/v1/providers/{user_lat}/{user_lng}/{radius}/all/all"
        response = requests.get(api_url)
        if response.status_code != 200:
            return None

        data = response.json()
        providers = data.get("data", {}).get("providers", [])
        provider_texts = []

        for provider in providers:
            provider_id = provider.get("userId")
            if not provider_id:
                continue

            detailed_url = f"https://ta7t-bety.vercel.app/api/v1/providers/{provider_id}"
            provider_response = requests.get(detailed_url)
            if provider_response.status_code == 200:
                provider_data = provider_response.json().get("data", {})
                providerID = provider_data.get("providerID", {})
                name = providerID.get("name", "Unknown")
                provider_type = provider_data.get("providerType", "N/A")
                address = provider_data.get("locations", [{}])[0].get("address", "Unknown Address")
                
                coords = provider_data.get("locations", [{}])[0].get("coordinates", {}).get("coordinates", [])
                distance = None
                if len(coords) >= 2:
                    distance = calculate_distance(user_lat, user_lng, coords[1], coords[0])
                
                posts = provider_data.get("posts", [])
                post_lines = []
                for post in posts:
                    title = post.get('title', '').strip()
                    content = post.get('content', '').strip()
                    price = post.get('price', '')
                    if title:
                        item_text = f"MENU_ITEM: {title}"
                        if content and content != title:
                            item_text += f" - {content}"
                        if price:
                            item_text += f" - PRICE: {price} EGP"
                        post_lines.append(item_text)
                
                reviews = provider_data.get("reviews", [])
                review_lines = []
                for r in reviews:
                    rating = r.get('rating', '')
                    review_text = r.get('review', '').strip()
                    if rating and review_text:
                        review_lines.append(f"REVIEW: {rating}/5 stars - {review_text}")
                
                avg_rating = provider_data.get("avgRating")
                reviews_count = provider_data.get("reviewsCount", 0)

                full_text = f"""RESTAURANT_START
RESTAURANT_NAME: {name}
SERVICE_TYPE: {provider_type}
ADDRESS: {address}"""

                if distance is not None:
                    if distance < 1:
                        full_text += f"\nDISTANCE: {int(distance * 1000)} meters from user location"
                    else:
                        full_text += f"\nDISTANCE: {distance:.1f} kilometers from user location"

                if avg_rating is not None:
                    full_text += f"\nOVERALL_RATING: {avg_rating} stars based on {reviews_count} reviews"

                if post_lines:
                    full_text += f"\nMENU_ITEMS:\n{chr(10).join(post_lines)}"

                if review_lines:
                    full_text += f"\nCUSTOMER_REVIEWS:\n{chr(10).join(review_lines)}"

                full_text += "\nRESTAURANT_END"
                provider_texts.append(full_text.strip())

        session_data_file = f"session_{session_id}_data.txt"
        
        with open(session_data_file, "w", encoding="utf-8") as f:
            f.write(f"USER_LOCATION_INFO: User is located at coordinates {user_lat}, {user_lng}\n\n")
            
            knowledge_docs = list(knowledge_collection.find())
            for doc in knowledge_docs:
                question = doc.get("question", "").strip()
                answer = doc.get("answer", "").strip()
                if question and answer:
                    f.write(f"FAQ_START\nQUESTION: {question}\nANSWER: {answer}\nFAQ_END\n\n")
            
            for provider_text in provider_texts:
                f.write(provider_text + "\n\n")
        
        return session_data_file
        
    except Exception as e:
        logger.error(f"Error creating consolidated data file: {str(e)}")
        return None

def detect_language(text):
    arabic_chars = sum(1 for char in text if '\u0600' <= char <= '\u06FF')
    total_chars = len([char for char in text if char.isalpha()])
    
    if total_chars == 0:
        return 'en'
    
    arabic_ratio = arabic_chars / total_chars
    return 'ar' if arabic_ratio > 0.3 else 'en'

def create_enhanced_prompt():
    template = """You are a food and restaurant expert assistant. Answer ONLY using information from the provided context below.

CRITICAL INSTRUCTIONS:
1. ONLY use information that exists in the CONTEXT DATA below
2. If specific information is NOT in the context, respond: "لا توجد معلومات محددة" (Arabic) or "No specific information available" (English)
3. Match the language of the question exactly
4. Be precise and specific - include restaurant names, prices, ratings, distances when available
5. For restaurant recommendations, mention specific names from the context
6. For menu items, quote exact items and prices from MENU_ITEM entries
7. For locations, use ADDRESS and DISTANCE information
8. For ratings, use OVERALL_RATING and REVIEW information

RESPONSE FORMAT:
- Give direct answers without mentioning "context" or "provided information"
- Include specific details like names, prices, ratings
- Be conversational and helpful

CONTEXT DATA:
{context}

QUESTION: {question}

ANSWER:"""

    return PromptTemplate(template=template, input_variables=["context", "question"])

def preprocess_text_for_chunking(text):
    processed_text = text
    
    processed_text = re.sub(r'\n+', '\n', processed_text)
    processed_text = re.sub(r'\s+', ' ', processed_text)
    
    sections = []
    current_section = []
    
    for line in processed_text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        if line.startswith(('RESTAURANT_START', 'FAQ_START')):
            if current_section:
                sections.append(' '.join(current_section))
                current_section = []
            current_section.append(line)
        elif line.startswith(('RESTAURANT_END', 'FAQ_END')):
            current_section.append(line)
            sections.append(' '.join(current_section))
            current_section = []
        else:
            current_section.append(line)
    
    if current_section:
        sections.append(' '.join(current_section))
    
    return '\n\n'.join(sections)

def create_session_vectorstore(session_data_file):
    try:
        loader = TextLoader(session_data_file, encoding="utf-8")
        documents = loader.load()
        
        processed_content = preprocess_text_for_chunking(documents[0].page_content)
        documents[0].page_content = processed_content
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\nRESTAURANT_START", "\n\nFAQ_START", "\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        
        filtered_chunks = []
        for chunk in chunks:
            content = chunk.page_content.strip()
            if len(content) >= 50 and any(keyword in content for keyword in [
                'RESTAURANT_NAME', 'MENU_ITEM', 'QUESTION', 'ANSWER', 'SERVICE_TYPE', 
                'PRICE', 'RATING', 'ADDRESS', 'REVIEW'
            ]):
                filtered_chunks.append(chunk)
        
        logger.info(f"Created {len(filtered_chunks)} high-quality chunks from {len(chunks)} total chunks")
        
        if not filtered_chunks:
            logger.error("No valid chunks created!")
            return None, None
        
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        vectorstore = FAISS.from_documents(filtered_chunks, embedding_model)
        
        session_index_path = f"session_{session_data_file.split('_')[1]}_faiss_index"
        vectorstore.save_local(session_index_path)
        
        vectorstore = FAISS.load_local(session_index_path, embedding_model, allow_dangerous_deserialization=True)
        
        llm = GeminiLLM(
            model_name="gemini-1.5-flash",
            temperature=0.0,
            max_tokens=1200
        )
        
        class ScoreFilteredRetriever:
            def __init__(self, vectorstore, score_threshold=0.5, k=8):
                self.vectorstore = vectorstore
                self.score_threshold = score_threshold
                self.k = k
            
            def get_relevant_documents(self, query):
                docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=self.k * 2)
                
                logger.info(f"Query: '{query}'")
                for i, (doc, score) in enumerate(docs_and_scores[:5]):
                    logger.info(f"Doc {i+1} (score: {score:.3f}): {doc.page_content[:100]}...")
                
                filtered_docs = [doc for doc, score in docs_and_scores if score <= self.score_threshold]
                
                if not filtered_docs:
                    filtered_docs = [doc for doc, score in docs_and_scores[:3]]
                    logger.warning(f"No docs met score threshold {self.score_threshold}, using top 3")
                
                return filtered_docs[:self.k]
        
        retriever = ScoreFilteredRetriever(vectorstore, score_threshold=0.7, k=6)
        
        prompt = create_enhanced_prompt()
        
        class CustomRetrievalQA:
            def __init__(self, llm, retriever, prompt):
                self.llm = llm
                self.retriever = retriever
                self.prompt = prompt
            
            def invoke(self, inputs):
                query = inputs["query"]
                docs = self.retriever.get_relevant_documents(query)
                
                context = "\n\n".join([doc.page_content for doc in docs])
                
                formatted_prompt = self.prompt.format(context=context, question=query)
                response = self.llm._call(formatted_prompt)
                
                return {"result": response, "source_documents": docs}
        
        qa_chain = CustomRetrievalQA(llm, retriever, prompt)
        
        return qa_chain, session_index_path
        
    except Exception as e:
        logger.error(f"Error creating session vectorstore: {str(e)}")
        return None, None

def clean_response(response_text):
    unwanted_phrases = [
        "Based on the provided information,", "Based on the information provided",
        "Based on the provided context,", "According to the information available,",
        "From the data provided,", "From the context provided,",
        "Based on the context,", "According to the context,",
        "The provided information shows", "The context indicates",
        "PRECISE ANSWER (same language as question):",
        "USER QUESTION:", "AVAILABLE DATA:", "CRITICAL INSTRUCTIONS:",
        "ANSWER:", "CONTEXT DATA:", "QUESTION:",
        "بناءً على المعلومات المتوفرة،", "وفقاً للسياق المتاح،",
        "حسب المعلومات الموجودة،", "من البيانات المقدمة،"
    ]
    
    cleaned = response_text
    for phrase in unwanted_phrases:
        cleaned = cleaned.replace(phrase, "")
    
    cleaned = " ".join(cleaned.split())
    cleaned = cleaned.strip()
    
    return cleaned

def update_text_file_incrementally():
    last_oid_str = get_last_fetch_time()
    query = {}
    if last_oid_str:
        try:
            query["_id"] = {"$gt": ObjectId(last_oid_str)}
        except Exception:
            pass

    new_service_docs = list(services_collection.find(query))
    new_knowledge_docs = list(knowledge_collection.find(query))

    if not new_service_docs and not new_knowledge_docs:
        logger.info("No new documents to update.")
        return

    with open(data_file, "a", encoding="utf-8") as f:
        for doc in new_service_docs:
            name = doc.get("restaurantName", "Unknown Provider")
            f.write(f'Provider name is "{name}"\n')
            for item in doc.get("menu", []):
                menu_item = item.get("item", "").strip()
                if menu_item:
                    f.write(f"Restaurant offers {menu_item}\n")
            f.write("\n")

        for doc in new_knowledge_docs:
            question = doc.get("question", "").strip()
            answer = doc.get("answer", "").strip()
            if question and answer:
                f.write(f"Q: {question}\nA: {answer}\n\n")

    max_id = max([doc["_id"] for doc in new_service_docs + new_knowledge_docs], default=ObjectId())
    set_last_fetch_time(str(max_id))
    logger.info("Incremental update complete.")

scheduler = BackgroundScheduler()
scheduler.add_job(update_text_file_incrementally, 'interval', hours=0.1)
scheduler.start()

if os.path.exists(data_file):
    loader = TextLoader(data_file, encoding="utf-8")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local("faiss_index")

    vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

    llm = GeminiLLM(
        model_name="gemini-1.5-flash",
        temperature=0.0,
        max_tokens=1000
    )
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
else:
    qa_chain = None



 
user_order_sessions = {}

def start_order_process(session_id):
    """Initialize order process for a session"""
    user_order_sessions[session_id] = {
        'step': 'provider_selection',
        'provider_name': None,
        'provider_id': None,
        'available_providers': [],
        'available_products': [],
        'selected_items': [],    
        'total_price': 0,
        'order_data': {}
    }


def fetch_available_providers(lat, lng):
    """Fetch available providers (restaurants and supermarkets only)"""
    try:
        api_url = f"https://ta7t-bety.vercel.app/api/v1/providers/{lat}/{lng}/1000/all/all"
        response = requests.get(api_url)
        if response.status_code != 200:
            return []

        data = response.json()
        providers = data.get("data", {}).get("providers", [])
        
        # فلترة المزودين للمطاعم والسوبرماركت فقط
        filtered_providers = []
        for provider in providers:
            provider_type = provider.get("providerType", "")
            if provider_type in ["F-Restaurants", "M-Supermarket"]:
                filtered_providers.append({
                    'name': provider.get('name', 'Unknown'),
                    'userId': provider.get('userId'),
                    'providerId': provider.get('userId'),  # ✅ استخدم userId بدل providerId
                    'providerType': provider_type
                })
        
        return filtered_providers
    except Exception as e:
        logger.error(f"Error fetching providers: {str(e)}")
        return []
    
# === 3. إضافة دالة لجلب منتجات المزود ===
def fetch_provider_products(provider_id):
    """Fetch products from specific provider"""
    try:
        api_url = f"https://ta7t-bety.vercel.app/api/v1/providers/{provider_id}"
        response = requests.get(api_url)
        if response.status_code != 200:
            return []

        data = response.json()

        # posts جوا data['data']
        posts = data.get("data", {}).get("posts", [])

        products = []
        for post in posts:
            products.append({
                'id': post.get('id'),
                'title': post.get('title', '') ,
                'content': post.get('content', '') ,
                'price': post.get('price', 0) 
            })

        print(products)
        return products

    except Exception as e:
        logger.error(f"Error fetching provider products: {str(e)}")
        return []

 

# استبدل دالة find_product_by_title الموجودة بهذه الدالة المحسنة

def find_product_by_title(products, title):
    """Find product by title with flexible matching"""
    title_lower = title.lower().strip()
    
    # 1. البحث الدقيق أولاً
    for product in products:
        if product['title'].lower().strip() == title_lower:
            return product
    
    # 2. البحث الجزئي - إذا كان العنوان يحتوي على النص المطلوب
    for product in products:
        if title_lower in product['title'].lower() or product['title'].lower() in title_lower:
            return product
    
    # 3. البحث بالرقم إذا كان المستخدم كتب رقم المنتج
    try:
        item_number = int(title_lower)
        if 1 <= item_number <= len(products):
            return products[item_number - 1]
    except ValueError:
        pass
    
    # 4. البحث المرن - إزالة المسافات والعلامات
    import re
    title_clean = re.sub(r'[^\w\u0600-\u06FF]', '', title_lower)
    
    for product in products:
        product_clean = re.sub(r'[^\w\u0600-\u06FF]', '', product['title'].lower())
        if title_clean == product_clean or title_clean in product_clean or product_clean in title_clean:
            return product
    
    return None

def get_order_step_question(step, lang='en', **kwargs):
    """Get question based on current order step"""
    questions = {
        'provider_selection': {
            'en': "Of course , Here are the available restaurants and supermarkets near you:\n\n{providers_list}\n\nPlease choose the name of the provider you want to order from 🏪",
            'ar': "دي المطاعم والسوبرماركت المتاحة قريب منك:\n\n{providers_list}\n\nاختار اسم المطعم أو السوبرماركت اللي عايز تطلب منه 🏪"
        },
        'product_selection': {
            'en': f"Here are the available items from {{provider_name}}:\n\n{{products_list}}\n\nPlease select the items you want by writing their names separated by commas (,) 🍽️",
            'ar': f"دي المنتجات المتاحة من {{provider_name}}:\n\n{{products_list}}\n\nاختار المنتجات اللي عايزها واكتب أسمائها مفصولة بفاصلة (,) 🍽️"
        },
        'quantity_selection': {
            'en': "Great! Now please tell me the quantity for each item you selected:\n\n{selected_items}\n\nPlease write the quantities in the same order, separated by commas. For example: 2, 1, 3 📦",
            'ar': "ممتاز! دلوقتي قولي عايز كمية قد إيه من كل منتج:\n\n{selected_items}\n\nاكتب الكميات بنفس الترتيب، مفصولة بفاصلة. مثلاً: 2, 1, 3 📦"
        },
        'confirmation': {
            'en': "Please review your order above. Do you want to confirm this order? (yes/no) ✅",
            'ar': "راجع طلبك فوق. عايز تأكد الطلب ده؟ (نعم/لا) ✅"
        }
    }
    
    return questions.get(step, {}).get(lang, questions[step]['en'])


def process_order_step(session_id, user_message, lang='en'):
    """Process current order step and return response"""
    if session_id not in user_order_sessions:
        return None, False
    
    order_session = user_order_sessions[session_id]
    current_step = order_session['step']
    session_data = user_sessions[session_id]
    
    # Handle cancellation
    if user_message.lower().strip() in ['cancel', 'إلغاء', 'الغي', 'stop', 'خلاص']:
        del user_order_sessions[session_id]
        cancel_msg = {
            'en': "Order cancelled. How can I help you today? 😊",
            'ar': "تم إلغاء الطلب. إزاي ممكن أساعدك النهارده؟ 😊"
        }
        return cancel_msg.get(lang, cancel_msg['en']), True
    
    if current_step == 'provider_selection':
        # جلب المزودين المتاحين
        providers = fetch_available_providers(session_data['lat'], session_data['lng'])
        if not providers:
            error_msg = {
                'en': "Sorry, no restaurants or supermarkets available in your area right now.",
                'ar': "آسف، مفيش مطاعم أو سوبرماركت متاحة في منطقتك دلوقتي."
            }
            return error_msg.get(lang, error_msg['en']), True
        
        order_session['available_providers'] = providers
        
        # تنسيق قائمة المزودين
        providers_list = []
        for i, provider in enumerate(providers, 1):
            provider_type_display = "🍽️ مطعم" if provider['providerType'] == "F-Restaurants" else "🛒 سوبرماركت"
            if lang == 'en':
                provider_type_display = "🍽️ Restaurant" if provider['providerType'] == "F-Restaurants" else "🛒 Supermarket"
            
            providers_list.append(f"{i}. {provider['name']} ({provider_type_display})")
        
        providers_text = "\n".join(providers_list)
        question = get_order_step_question('provider_selection', lang)
        
        order_session['step'] = 'awaiting_provider_choice'
        return question.format(providers_list=providers_text), False
    
    elif current_step == 'awaiting_provider_choice':
        # البحث عن المزود المختار
        selected_provider = None
        user_choice = user_message.strip()
        
        for provider in order_session['available_providers']:
            if provider['name'].lower() == user_choice.lower():
                selected_provider = provider
                break
        
        if not selected_provider:
            error_msg = {
                'en': "Please enter the exact name of the provider from the list above.",
                'ar': "من فضلك اكتب الاسم الصحيح للمطعم أو السوبرماركت من القائمة اللي فوق."
            }
            return error_msg.get(lang, error_msg['en']), False
        
        # حفظ المزود المختار وجلب منتجاته
        order_session['provider_name'] = selected_provider['name']
        order_session['provider_id'] = selected_provider['providerId']
        
        products = fetch_provider_products(selected_provider['providerId'])
        if not products:
            error_msg = {
                'en': f"Sorry, no products available from {selected_provider['name']} right now.",
                'ar': f"آسف، مفيش منتجات متاحة من {selected_provider['name']} دلوقتي."
            }
            return error_msg.get(lang, error_msg['en']), True
        
        order_session['available_products'] = products
        
        # تنسيق قائمة المنتجات
        products_list = []
        for i, product in enumerate(products, 1):
            product_info = f"{i}. **{product['title']}**"
            if product['content']:
                product_info += f"\n   {product['content']}"
            if product['price']:
                product_info += f"\n   💰 السعر: {product['price']} جنيه" if lang == 'ar' else f"\n   💰 Price: {product['price']} EGP"
            products_list.append(product_info)
        
        products_text = "\n\n".join(products_list)
        question = get_order_step_question('product_selection', lang)
        
        order_session['step'] = 'awaiting_product_choice'
        return question.format(provider_name=selected_provider['name'], products_list=products_text), False
    
    elif current_step == 'awaiting_product_choice':
        # معالجة المنتجات المختارة
        selected_titles = [title.strip() for title in user_message.split(',')]
        found_products = []
        not_found = []
        
        for title in selected_titles:
            if not title:
                continue
            
            product = find_product_by_title(order_session['available_products'], title)
            if product:
                found_products.append(product)
            else:
                not_found.append(title)
        
        if not found_products:
            error_msg = {
                'en': "No products found with the names you entered. Please check the spelling and try again.",
                'ar': "مفيش منتجات بالأسماء دي. راجع الكتابة وجرب تاني."
            }
            return error_msg.get(lang, error_msg['en']), False
        
        # حفظ المنتجات المختارة (بدون كميات لسه)
        order_session['selected_products'] = found_products
        
        # إنشاء قائمة المنتجات المختارة لسؤال الكمية
        selected_items_list = []
        for i, product in enumerate(found_products, 1):
            selected_items_list.append(f"{i}. {product['title']} - {product['price']} جنيه" if lang == 'ar' else f"{i}. {product['title']} - {product['price']} EGP")
        
        selected_items_text = "\n".join(selected_items_list)
        
        not_found_msg = ""
        if not_found:
            not_found_msg = f"\n⚠️ لم يتم العثور على: {', '.join(not_found)}\n" if lang == 'ar' else f"\n⚠️ Not found: {', '.join(not_found)}\n"
        
        question = get_order_step_question('quantity_selection', lang)
        
        order_session['step'] = 'awaiting_quantity_choice'
        return not_found_msg + question.format(selected_items=selected_items_text), False
    
    elif current_step == 'awaiting_quantity_choice':
        # معالجة الكميات
        try:
            quantities = [int(q.strip()) for q in user_message.split(',')]
        except ValueError:
            error_msg = {
                'en': "Please enter valid numbers for quantities, separated by commas. For example: 2, 1, 3",
                'ar': "من فضلك اكتب أرقام صحيحة للكميات، مفصولة بفاصلة. مثلاً: 2, 1, 3"
            }
            return error_msg.get(lang, error_msg['en']), False
        
        if len(quantities) != len(order_session['selected_products']):
            error_msg = {
                'en': f"Please enter {len(order_session['selected_products'])} quantities (one for each item).",
                'ar': f"من فضلك اكتب {len(order_session['selected_products'])} كميات (واحدة لكل منتج)."
            }
            return error_msg.get(lang, error_msg['en']), False
        
        # حفظ المنتجات مع الكميات
        selected_items = []
        total_price = 0
        
        for product, quantity in zip(order_session['selected_products'], quantities):
            if quantity <= 0:
                error_msg = {
                    'en': "Quantities must be greater than 0.",
                    'ar': "الكميات لازم تكون أكبر من صفر."
                }
                return error_msg.get(lang, error_msg['en']), False
            
            item_total = product['price'] * quantity
            selected_items.append({
                'product': product,
                'quantity': quantity,
                'item_total': item_total
            })
            total_price += item_total
        
        order_session['selected_items'] = selected_items
        order_session['total_price'] = total_price
        
        # إنشاء ملخص الطلب مع الكميات
        summary_lines = []
        for item in selected_items:
            product = item['product']
            quantity = item['quantity']
            item_total = item['item_total']
            
            if lang == 'ar':
                summary_lines.append(f"• {product['title']} × {quantity} = {item_total} جنيه")
            else:
                summary_lines.append(f"• {product['title']} × {quantity} = {item_total} EGP")
        
        summary = {
            'en': f"""📋 **Order Summary:**
Restaurant/Store: {order_session['provider_name']}
Items:
{chr(10).join(summary_lines)}

Total Price: {total_price} EGP 💰

{get_order_step_question('confirmation', lang)}""",
            'ar': f"""📋 **ملخص الطلب:**
المطعم/المتجر: {order_session['provider_name']}
المنتجات:
{chr(10).join(summary_lines)}

إجمالي السعر: {total_price} جنيه 💰

{get_order_step_question('confirmation', lang)}"""
        }
        
        order_session['step'] = 'confirmation'
        return summary.get(lang, summary['en']), False
    
    elif current_step == 'confirmation':
        user_response = user_message.lower().strip()
        if user_response in ['yes', 'نعم', 'أيوة', 'اه', 'موافق', 'تمام']:
            return "CONFIRM_ORDER", False
        elif user_response in ['no', 'لا', 'مش عايز', 'إلغاء']:
            del user_order_sessions[session_id]
            cancel_msg = {
                'en': "Order cancelled. How can I help you today? 😊",
                'ar': "تم إلغاء الطلب. إزاي ممكن أساعدك النهارده؟ 😊"
            }
            return cancel_msg.get(lang, cancel_msg['en']), True
        else:
            return get_order_step_question('confirmation', lang), False
    
    return None, False

def save_order_to_database(session_id, user_id):
    """Save confirmed order to database with IDs"""
    if session_id not in user_order_sessions:
        logger.error(f"Session {session_id} not found in user_order_sessions")
        return False
    
    order_session = user_order_sessions[session_id]
    
    try:
        orders_collection = db["orders"]
        
        # إنشاء قائمة المنتجات بـ IDs
        items = []
        post_ids = []
        
        for item in order_session['selected_items']:
            product = item['product']
            quantity = item['quantity']
            
            # تحقق من وجود ID المنتج
            if 'id' not in product or not product['id']:
                logger.error(f"Product missing ID: {product}")
                return False
            
            items.append({
                "postId": product['id'],
                "title": product['title'],
                "price": product['price'],
                "quantity": quantity,
                "itemTotal": item['item_total']
            })
            
            post_ids.append(ObjectId(product['id']))
        
        order_document = {
            "userID": ObjectId(user_id),
            "providerID": ObjectId(order_session['provider_id']),
            "sessionID": session_id,
            "postIDs": post_ids,
            "items": items,
            "totalPrice": order_session['total_price'],
            "status": "pending",
            "createdAt": datetime.utcnow()
        }
        
        logger.info(f"Attempting to save order: {order_document}")
        result = orders_collection.insert_one(order_document)
        
        if result.inserted_id:
            logger.info(f"Order saved successfully with ID: {result.inserted_id}")
            del user_order_sessions[session_id]
            return True
        else:
            logger.error("Failed to insert order - no ID returned")
            return False
        
    except Exception as e:
        logger.error(f"Error saving order: {str(e)}")
        logger.error(f"Order session data: {order_session}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False
    

@app.route('/start_chat', methods=['POST'])
def start_chat():
    try:
        global current_session_id
        user_id = request.json.get('user_id')
        lang = request.json.get('lang', 'en')   

        if not user_id:
            return jsonify({"error": "User ID is required"}), 400

        user = users_collection.find_one({"_id": ObjectId(user_id)})
        if not user:
            return jsonify({"error": "User not found"}), 404

        name = user.get("name", "User")
        favorite_location = next((loc for loc in user.get("locations", []) if loc.get("isFavorite")), None)

        if not favorite_location:
            return jsonify({"error": "No favorite location found"}), 404

        coords = favorite_location["coordinates"]["coordinates"]
        lng = coords[0]
        lat = coords[1]

        session_id = str(uuid.uuid4())
        
        logger.info(f"Creating session for user at location: {lat}, {lng}")
        
        session_data_file = create_consolidated_data_file(session_id, lat, lng)
        
        if not session_data_file:
            return jsonify({"error": "Failed to create session data"}), 500
        
        session_qa_chain, session_index_path = create_session_vectorstore(session_data_file)
        
        if not session_qa_chain:
            return jsonify({"error": "Failed to create session vectorstore"}), 500
        
        user_sessions[session_id] = {
            "user_id": user_id,
            "name": name,
            "lat": lat,
            "lng": lng,
            "lang": lang,
            "data_file": session_data_file,
            "qa_chain": session_qa_chain,
            "index_path": session_index_path
        }
        
        current_session_id = session_id

        message = {
            "en": f"Hi {name}! 😊 I'm ready to help you find the best restaurants and services near you. What are you looking for?",
            "ar": f"أهلاً {name}! 😊 أنا جاهز أساعدك تلاقي أحسن المطاعم والخدمات قريب منك. إيه اللي بتدور عليه؟"
        }

        logger.info(f"Session {session_id} created successfully")

        return jsonify({
            "message": message.get(lang, message["en"]),
            "location": {"lat": lat, "lng": lng},
            "lang": lang,
            "session_id": session_id
        })

    except Exception as e:
        logger.error(f"Error in start_chat: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        global current_session_id
        user_message = request.json.get('message')
        
        if not user_message:
            return jsonify({"error": "Message is required"}), 400
            
        if not current_session_id or current_session_id not in user_sessions:
            return jsonify({"error": "No active session. Please start a chat first"}), 400
            
        session_data = user_sessions[current_session_id]
        lang = session_data['lang']
        session_qa_chain = session_data['qa_chain']
        user_name = session_data['name']
        user_id = session_data['user_id']
        
        logger.info(f"Processing query: '{user_message}'")
        
        # Check if user is in ordering process
        if current_session_id in user_order_sessions:
            order_response, is_complete = process_order_step(current_session_id, user_message, lang)
            
            if order_response == "CONFIRM_ORDER":
                # Save order to database
                if save_order_to_database(current_session_id, user_id):
                    success_msg = {
                        'en': "✅ Order confirmed successfully! Your order has been placed and is pending. How can I help you today?",
                        'ar': "✅ تم تأكيد الطلب بنجاح! طلبك تم إرساله وهو قيد المعالجة. إزاي ممكن أساعدك النهارده؟"
                    }
                    answer = success_msg.get(lang, success_msg['en'])
                else:
                    error_msg = {
                        'en': "❌ Sorry, there was an error saving your order. Please try again.",
                        'ar': "❌ آسف، حدث خطأ في حفظ طلبك. جرب مرة تانية."
                    }
                    answer = error_msg.get(lang, error_msg['en'])
            else:
                answer = order_response
            
            return jsonify({
                "response": answer,
                "lang": lang
            })
        
        # Check if user wants to start ordering
        if is_order_request(user_message):
            start_order_process(current_session_id)
            order_response, is_complete = process_order_step(current_session_id, "", lang)
            
            return jsonify({
                "response": order_response,
                "lang": lang
            })
        # Regular chat processing (existing code)
        message_type = is_greeting_or_thanks(user_message)
        
        if message_type == 'greeting':
            answer = get_greeting_response(lang, user_name)
        elif message_type == 'thanks':
            answer = get_thanks_response(lang)
        else:
            try:
                response = session_qa_chain.invoke({"query": user_message})
                raw_answer = response["result"] if isinstance(response, dict) else str(response)
                
                logger.info(f"Raw AI response: {raw_answer[:200]}...")
                
                answer = clean_response(raw_answer)
                answer = add_food_emojis(answer)
                answer = format_menu_with_stars(answer)
                
                if (not answer or len(answer.strip()) < 15 or 
                    "no specific information" in answer.lower() or 
                    "لا توجد معلومات محددة" in answer or
                    "no information available" in answer.lower()):
                    
                    detected_lang = detect_language(user_message)
                    fallback_messages = {
                        "en": "I couldn't find specific details about that in current. you may ask about specific restaurants, menu items, prices, or services in your area 🛒",
                        "ar": "مقدرتش ألاقي تفاصيل محددة عن كده في قاعدة البيانات. ممكن تسأل عن مطاعم معينة أو أكلات أو أسعار أو خدمات في منطقتك؟ مثلاً: 'إيه المطاعم اللي بتقدم حدمة معينة؟' أو 'وريني أسعار'. كمان ممكن تقول 'طلب' عشان تعمل اوردر  ! 🛒"
                    }
                    answer = fallback_messages.get(detected_lang, fallback_messages["en"])
                
            except Exception as ai_error:
                logger.error(f"AI processing error: {str(ai_error)}")
                detected_lang = detect_language(user_message)
                error_messages = {
                    "en": "Sorry, I had trouble processing your request. Please try asking about specific restaurants or menu items, or say 'order' to place an order.",
                    "ar": "آسف، واجهت مشكلة في معالجة طلبك. جرب تسأل عن مطاعم معينة أو أكلات محددة، أو قول 'طلب' عشان تطلب اوردر."
                }
                answer = error_messages.get(detected_lang, error_messages["en"])
        
        logger.info(f"Final response: {answer[:100]}...")
        
        return jsonify({
            "response": answer,
            "lang": lang
        })

    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        return jsonify({"error": str(e)}), 500
    


@app.route('/end_chat', methods=['POST'])
def end_chat():
    try:
        global current_session_id
        
        if not current_session_id or current_session_id not in user_sessions:
            return jsonify({"error": "No active session"}), 400
        
        if current_session_id in user_order_sessions:
             del user_order_sessions[current_session_id]
        
        session_data = user_sessions[current_session_id]
        
        try:
            if os.path.exists(session_data['data_file']):
                os.remove(session_data['data_file'])
            
            index_path = session_data['index_path']
            import shutil
            if os.path.exists(index_path):
                 shutil.rmtree(index_path)
                    
        except Exception as e:
            logger.error(f"Error cleaning up session files: {str(e)}")
        
        del user_sessions[current_session_id]
        current_session_id = None
        
        logger.info(f"Session ended and cleaned up successfully")
        
        return jsonify({"message": "Chat session ended successfully"})
        

        
    except Exception as e:
        logger.error(f"Error in end_chat: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    serve(app, host='0.0.0.0', port=5000)
