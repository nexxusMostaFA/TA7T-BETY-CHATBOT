from waitress import serve
from flask import Flask, request, jsonify, session
from pymongo import MongoClient
from bson.objectid import ObjectId
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
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

app = Flask(__name__)
app.secret_key = os.urandom(24)   

client = MongoClient("mongodb+srv://projectDB:PEyHwQ2fF7e5saEf@cluster0.43hxo.mongodb.net/")
db = client["ta7t-bety"]
users_collection = db["users"]
services_collection = db["chatbot_services"]
knowledge_collection = db["knowledge_base"]

user_sessions = {}
current_session_id = None

data_file = r"C:\Users\mostafa\Documents\GitHub\ta7t-bety-chatbot\all_provider_data.txt"
fetch_time_file = r"C:\Users\mostafa\Documents\GitHub\ta7t-bety-chatbot\.last_fetch.txt"

def is_greeting_or_thanks(message):
    """Check if message is a greeting or thanks"""
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
    """Get appropriate greeting response"""
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
    """Get appropriate thanks response"""
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
    return random.choice(responses.get(lang, responses['en']))

def add_food_emojis(text):
    """Add appropriate food emojis to the response"""
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
    """Format menu items with star separators"""
    
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
    """Calculate distance between two coordinates in kilometers"""
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

def similarity(a, b):
    """Calculate similarity between two strings"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def consolidate_provider_data(static_providers, live_providers, user_lat, user_lng):
    """Consolidate static and live provider data with distance calculation"""
    consolidated = {}
 
    for provider in static_providers:
        name = provider.get('name', '').strip()
        if name:
            consolidated[name] = {
                'name': name,
                'menu': provider.get('menu', []),
                'type': provider.get('type', ''),
                'static_data': True,
                'live_data': False,
                'distance_km': None
            }
 
    for live_provider in live_providers:
        live_name = live_provider.get('name', '').strip()
        if not live_name:
            continue
            
      
        distance = None
        if live_provider.get('coordinates') and live_provider['coordinates'].get('lat') and live_provider['coordinates'].get('lng'):
            distance = calculate_distance(
                user_lat, user_lng,
                live_provider['coordinates']['lat'],
                live_provider['coordinates']['lng']
            )
   
        best_match = None
        best_similarity = 0
        
        for static_name in consolidated.keys():
            sim = similarity(live_name, static_name)
            if sim > best_similarity and sim >= 0.75:  
                best_similarity = sim
                best_match = static_name
        
        if best_match:
           
            consolidated[best_match].update({
                'provider_type': live_provider.get('providerType', ''),
                'address': live_provider.get('address', ''),
                'coordinates': live_provider.get('coordinates', {}),
                'avg_rating': live_provider.get('avgRating'),
                'reviews_count': live_provider.get('reviewsCount', 0),
                'live_data': True,
                'distance_km': distance
            })
        else:
          
            consolidated[live_name] = {
                'name': live_name,
                'provider_type': live_provider.get('providerType', ''),
                'address': live_provider.get('address', ''),
                'coordinates': live_provider.get('coordinates', {}),
                'avg_rating': live_provider.get('avgRating'),
                'reviews_count': live_provider.get('reviewsCount', 0),
                'menu': [],
                'static_data': False,
                'live_data': True,
                'distance_km': distance
            }
    
    return consolidated

def create_consolidated_data_file(session_id, user_lat, user_lng, radius=1000):
    """Create a consolidated data file for the session with user location context"""
    try:
     
        static_providers = []
        service_docs = list(services_collection.find())
        for doc in service_docs:
            static_providers.append({
                'name': doc.get("restaurantName", "Unknown Provider"),
                'menu': doc.get("menu", []),
                'type': 'restaurant'
            })
        
       
        live_providers = []
        api_url = f"https://ta7t-bety.vercel.app/api/v1/providers/{user_lat}/{user_lng}/{radius}/all/all"
        response = requests.get(api_url)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("data") and data.get("data").get("providers"):
                for provider in data["data"]["providers"]:
                
                    coordinates = {}
                    address = "Unknown address"
                    
                    if provider.get("locations") and len(provider["locations"]) > 0:
                        location = provider["locations"][0]
                        if location.get("coordinates") and location["coordinates"].get("coordinates"):
                            coords = location["coordinates"]["coordinates"]
                            coordinates = {
                                'lat': coords[1] if len(coords) > 1 else 0,
                                'lng': coords[0] if len(coords) > 0 else 0
                            }
                        address = location.get("address", "Unknown address")
                    
                    live_providers.append({
                        'name': provider.get("name", "Unknown Provider"),
                        'providerType': provider.get("providerType", "Unknown Service Type"),
                        'address': address,
                        'coordinates': coordinates,
                        'avgRating': provider.get("avgRating"),
                        'reviewsCount': provider.get("reviewsCount", 0)
                    })
        
        
        consolidated = consolidate_provider_data(static_providers, live_providers, user_lat, user_lng)
         
        session_data_file = f"session_{session_id}_data.txt"
        
        with open(session_data_file, "w", encoding="utf-8") as f:
         
            f.write(f"USER_LOCATION: User is located at coordinates {user_lat}, {user_lng}\n")
            f.write("USER_CONTEXT: When mentioning distances or locations, relate them to the user's current position.\n\n")
            
         
            knowledge_docs = list(knowledge_collection.find())
            for doc in knowledge_docs:
                question = doc.get("question", "").strip()
                answer = doc.get("answer", "").strip()
                if question and answer:
                    f.write(f"Q: {question}\nA: {answer}\n\n")
            
           
            f.write("AVAILABLE PROVIDERS NEAR YOU:\n\n")
            for provider_name, provider_data in consolidated.items():
                f.write(f"- {provider_name} is a ")
 
                if provider_data.get('provider_type'):
                    f.write(f"{provider_data['provider_type']}")
                elif provider_data.get('type'):
                    f.write(f"{provider_data['type']}")
                else:
                    f.write("service provider")
 
                if provider_data.get('address'):
                    f.write(f" located at {provider_data['address']}")
 
                if provider_data.get('distance_km') is not None:
                    dist = provider_data['distance_km']
                    if dist < 1:
                        f.write(f", just {int(dist * 1000)} meters away from you")
                    else:
                        f.write(f", approximately {dist:.1f} kilometers from you")

                f.write(". ")
 
                if provider_data.get('avg_rating') is not None:
                    rating = provider_data['avg_rating']
                    reviews = provider_data.get('reviews_count', 0)
                    f.write(f"They are rated {rating} stars with {reviews} reviews. ")
 
                if provider_data.get('menu'):
                    menu_items = [item.get("item", "").strip() if isinstance(item, dict) else str(item).strip() for item in provider_data['menu']]
                    menu_items = [item for item in menu_items if item]
                    if menu_items:
                        f.write(f"They offer items like {', '.join(menu_items[:5])}. ")

                f.write("\n\n")
        
        return session_data_file

        
    except Exception as e:
        print(f"Error creating consolidated data file: {str(e)}")
        return None

def detect_language(text):
    """Detect if the text is primarily in Arabic or English"""
    arabic_chars = sum(1 for char in text if '\u0600' <= char <= '\u06FF')
    total_chars = len([char for char in text if char.isalpha()])
    
    if total_chars == 0:
        return 'en'
    
    arabic_ratio = arabic_chars / total_chars
    return 'ar' if arabic_ratio > 0.3 else 'en'

def create_custom_prompt():
    """Create a custom prompt template for better responses in both Arabic and English"""
    template = """You are a helpful restaurant and service assistant that can understand and respond in both Arabic and English. Use the following context to answer the user's question in a friendly, concise way.

IMPORTANT INSTRUCTIONS:
1. Give short, friendly, and helpful answers
2. NEVER mention "based on the provided information" or similar phrases
3. When asked about a restaurant/service, provide: menu, location, rating, and distance
4. Use friendly language like "You can find", "It's located at", "They offer" (in English) or "ممكن تلاقي", "مكانه في", "بيقدموا" (in Arabic)
5. If something is close to the user (less than 2km), mention it's "nearby" or "close to you" (in English) or "قريب منك" or "جنبك" (in Arabic)
6. Don't include source information or technical details
7. Keep responses conversational and natural
8. RESPOND IN THE SAME LANGUAGE AS THE QUESTION:
   - If the question is in Arabic, respond in Arabic
   - If the question is in English, respond in English
   - If the question is mixed, respond in the dominant language

Arabic Keywords to understand:
- مطعم، مطاعم = restaurant, restaurants
- أكل، طعام = food
- فين، وين، أين = where
- قريب = near, close
- بعيد = far
- منيو، قائمة الطعام = menu
- عنوان، مكان = address, location
- تقييم، نجوم = rating, stars
- أسعار، فلوس = prices, money
- مفتوح، مقفول = open, closed
- توصيل = delivery
- احسن، أفضل = best
- رخيص، غالي = cheap, expensive

Context: {context}

Question: {question}

Answer in a friendly, helpful way in the same language as the question:"""

    return PromptTemplate(template=template, input_variables=["context", "question"])

def create_session_vectorstore(session_data_file):
    """Create vectorstore for session-specific data file with custom prompt"""
    try:
        loader = TextLoader(session_data_file, encoding="utf-8")
        documents = loader.load()
   
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400, 
            chunk_overlap=50,
            separators=["\nPROVIDER:", "\n\n", "\n", ". "]
        )
        chunks = splitter.split_documents(documents)
        
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embedding_model)
        
        session_index_path = f"session_{session_data_file.split('_')[1]}_faiss_index"
        vectorstore.save_local(session_index_path)
        
      
        vectorstore = FAISS.load_local(session_index_path, embedding_model, allow_dangerous_deserialization=True)
        
        llm = Ollama(model="llama3.2")
        retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
        
        
        prompt = create_custom_prompt()
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,  
            chain_type_kwargs={"prompt": prompt}
        )
        
        return qa_chain, session_index_path
        
    except Exception as e:
        print(f"Error creating session vectorstore: {str(e)}")
        return None, None

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
        print("⏳ No new documents to update.")
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
    print("✅ Incremental update complete.")

scheduler = BackgroundScheduler()
scheduler.add_job(update_text_file_incrementally, 'interval', hours=0.1)
scheduler.start()

 
if os.path.exists(data_file):
    loader = TextLoader(data_file)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local("faiss_index")

    vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

    llm = Ollama(model="llama3.2")
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
else:
    qa_chain = None

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
        
        print(f"🚀 Creating session for user at location: {lat}, {lng}")
        
       
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
            "en": f"Hi {name}! 😊 I'm here to help you find restaurants and services. Ask me anything!",
            "ar": f"أهلاً {name}! 😊 أنا هنا عشان أساعدك تلاقي المطاعم والخدمات حواليك. اسألني أي حاجة!"
        }

        print(f"✅ Session {session_id} created successfully with location context")

        return jsonify({
            "message": message.get(lang, message["en"]),
            "location": {"lat": lat, "lng": lng},
            "lang": lang,
            "session_id": session_id
        })

    except Exception as e:
        print(f"❌ Error in start_chat: {str(e)}")
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
        user_lat = session_data['lat']
        user_lng = session_data['lng']
        user_name = session_data['name']
        
        print(f"🔍 Processing query: '{user_message}' for user at {user_lat}, {user_lng}")
        
         
        message_type = is_greeting_or_thanks(user_message)
        
        if message_type == 'greeting':
            answer = get_greeting_response(lang, user_name)
        elif message_type == 'thanks':
            answer = get_thanks_response(lang)
        else:
             
            detected_lang = detect_language(user_message)
            
            
            response = session_qa_chain.invoke({"query": user_message})
            
            answer = response["result"] if isinstance(response, dict) else str(response)
            
            
            phrases_to_remove = [
                "Based on the provided information,",
                "Based on the provided context,", 
                "According to the information available,",
                "From the data provided,",
                "From the context provided,",
                "Based on the context,",
                "According to the context,",
                "The provided information shows",
                "The context indicates",
                "Please note that this answer is based solely on the provided context",
                "This information comes from",
                "may not be exhaustive or up-to-date",
                "User question:",
                "Please provide a comprehensive answer",
                "Context:",
                "Question:",
                "Answer in a friendly, helpful way:",
                "Answer in a friendly, helpful way in the same language as the question:",
              
                "بناءً على المعلومات المتوفرة،",
                "وفقاً للسياق المتاح،",
                "حسب المعلومات الموجودة،",
                "من البيانات المقدمة،"
            ]
            
            for phrase in phrases_to_remove:
                answer = answer.replace(phrase, "")
            
            
            answer = " ".join(answer.split())
            
            
            answer = add_food_emojis(answer)
            
           
            answer = format_menu_with_stars(answer)
            
     
            if not answer or len(answer.strip()) < 10:
          
                fallback_messages = {
                    "en": "I couldn't find specific information about that. Could you try asking about a specific restaurant or service?",
                    "ar": "مقدرتش ألاقي معلومات محددة عن ده. ممكن تسأل عن مطعم أو خدمة معينة؟"
                }
                answer = fallback_messages.get(detected_lang, fallback_messages["en"])
        
        print(f"✅ Response generated: {answer[:100]}...")
        
        return jsonify({
            "response": answer,
            "lang": lang
        })

    except Exception as e:
        print(f"❌ Error in chat: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/end_chat', methods=['POST'])
def end_chat():
    try:
        global current_session_id
        
        if not current_session_id or current_session_id not in user_sessions:
            return jsonify({"error": "No active session"}), 400
        
        session_data = user_sessions[current_session_id]
        
     
        try:
            if os.path.exists(session_data['data_file']):
                os.remove(session_data['data_file'])
            
      
            index_path = session_data['index_path']
            import shutil
            if os.path.exists(index_path):
                shutil.rmtree(index_path)
                    
        except Exception as e:
            print(f"Error cleaning up session files: {str(e)}")
        
        
        del user_sessions[current_session_id]
        current_session_id = None
        
        print(f"✅ Session ended and cleaned up successfully")
        
        return jsonify({"message": "Chat session ended successfully"})
        
    except Exception as e:
        print(f"❌ Error in end_chat: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    serve(app, host='0.0.0.0', port=5000)