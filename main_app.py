import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from itertools import chain # Tambahan untuk menggabungkan list

# ==========================================
# BAGIAN 1: LOGIKA & DATA (BACKEND)
# ==========================================

KEYWORD_MAPPING = {
    "menggambar": "desain visual art seni fotografi kreatif sketsa ilustrasi grafis",
    "gambar": "desain visual art seni fotografi kreatif sketsa ilustrasi grafis",
    "seni": "desain visual art seni fotografi kreatif sketsa ilustrasi grafis",
    "jualan": "marketing bisnis manajemen pemasaran retail sales perdagangan kewirausahaan entrepreneur",
    "dagang": "marketing bisnis manajemen pemasaran retail sales perdagangan kewirausahaan entrepreneur",
    "bisnis": "marketing bisnis manajemen pemasaran retail sales perdagangan kewirausahaan entrepreneur",
    "ngoding": "teknologi informasi sistem komputer data algoritma programming python web software aplikasi digital",
    "coding": "teknologi informasi sistem komputer data algoritma programming python web software aplikasi digital",
    "komputer": "teknologi informasi sistem komputer data algoritma programming python web software aplikasi digital",
    "hitung": "akuntansi statistika matematika ekonomi keuangan pajak finance analisis",
    "angka": "akuntansi statistika matematika ekonomi keuangan pajak finance analisis",
    "jalan-jalan": "pariwisata hospitality hotel tour travel guide tourism wisata perhotelan",
    "traveling": "pariwisata hospitality hotel tour travel guide tourism wisata perhotelan",
    "pariwisata": "pariwisata hospitality hotel tour travel guide tourism wisata perhotelan",
    "masak": "food beverage tata boga kitchen pastry kuliner makanan minuman chef",
    "memasak": "food beverage tata boga kitchen pastry kuliner makanan minuman chef",
    "kuliner": "food beverage tata boga kitchen pastry kuliner makanan minuman chef",
    "desain": "desain visual kreatif grafis komunikasi media digital",
    "komunikasi": "komunikasi media jurnalistik broadcast public relations PR",
    "film": "film broadcasting multimedia produksi sinema animasi video",
    "musik": "musik audio sound production recording entertainment",
    "olahraga": "sport fitness kesehatan health wellness management",
    "data": "data science analytics statistika machine learning artificial intelligence AI",
    "game": "game development interactive design programming unity multimedia",
}

# --- DATA DESKRIPSI JURUSAN (BARU) ---
PROGRAM_DESCRIPTIONS = {
    "Informatika": "Mempelajari pengembangan software, teknologi jaringan, dan komputasi cerdas untuk solusi masa depan.",
    "Sistem Informasi": "Menggabungkan ilmu komputer dengan manajemen bisnis untuk mengelola sistem perusahaan.",
    "Manajemen": "Fokus pada pengelolaan bisnis, strategi pemasaran, keuangan, dan kepemimpinan organisasi.",
    "Akuntansi": "Ahli dalam pencatatan, analisis, dan pelaporan keuangan untuk keputusan bisnis yang akurat.",
    "Ilmu Komunikasi": "Mempelajari strategi penyampaian pesan efektif melalui media digital, humas, dan jurnalistik.",
    "Hospitality dan Pariwisata": "Menyiapkan profesional di bidang perhotelan, kuliner, dan manajemen destinasi wisata.",
    "Desain Komunikasi Visual": "Mengembangkan solusi komunikasi visual yang kreatif, artistik, dan inovatif.",
    "Bahasa Inggris": "Mendalami bahasa, sastra, dan budaya Inggris untuk komunikasi profesional global.",
    "Bahasa Mandarin": "Mempelajari bahasa dan budaya Tiongkok untuk keunggulan bisnis internasional.",
    "Bisnis Digital": "Mengintegrasikan teknologi digital canggih dalam strategi dan operasional bisnis modern.",
    "Data Science": "Mengolah data besar (Big Data) menjadi wawasan berharga untuk prediksi dan keputusan.",
    "Desain Interaktif": "Fokus pada perancangan pengalaman pengguna (UX) dan antarmuka (UI) game serta media interaktif.",
    "Psikologi": "Mempelajari perilaku manusia dan proses mental untuk kesejahteraan individu dan organisasi."
}

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('List Mata Kuliah UBM.xlsx - Sheet1.csv')
        df = df.dropna()
        df['combined_features'] = df['Course'].astype(str) + ' ' + df['Program'].astype(str)
        return df
    except FileNotFoundError:
        st.error("File CSV tidak ditemukan. Pastikan file 'List Mata Kuliah UBM.xlsx - Sheet1.csv' ada di folder yang sama.")
        return pd.DataFrame()

def get_program_description(program_name):
    """Mencari deskripsi yang cocok berdasarkan nama jurusan."""
    for key, desc in PROGRAM_DESCRIPTIONS.items():
        if key in program_name:
            return desc
    return "Jurusan unggulan yang siap mencetak profesional handal di bidangnya."

def get_course_advice(course_name):
    """
    Memberikan tips dan deskripsi umum berdasarkan kata kunci pada nama mata kuliah.
    Ini adalah 'Smart Logic' karena kita tidak punya data tips spesifik per matkul.
    """
    course_lower = course_name.lower()
    
    if any(x in course_lower for x in ['matematika', 'kalkulus', 'statistika', 'akuntansi', 'keuangan', 'fisika']):
        return {
            "desc": "Mata kuliah ini banyak melibatkan logika, rumus, perhitungan, dan ketelitian angka.",
            "tip": "💡 **Tips Sukses:** Jangan hanya menghapal rumus, tapi pahami konsep dasarnya. Perbanyak latihan soal mandiri agar terbiasa dengan berbagai variasi kasus perhitungan."
        }
    elif any(x in course_lower for x in ['program', 'coding', 'algoritma', 'data', 'sistem', 'web', 'mobile', 'software']):
        return {
            "desc": "Fokus pada pengembangan logika teknis, struktur data, dan penulisan kode (coding) untuk membangun aplikasi.",
            "tip": "💻 **Tips Sukses:** Praktek langsung (ngoding) jauh lebih efektif daripada cuma baca teori. Jangan takut error, itu bagian dari proses belajar! Manfaatkan sumber belajar online seperti StackOverflow."
        }
    elif any(x in course_lower for x in ['desain', 'gambar', 'visual', 'art', 'sketsa', 'nirmana', 'tipografi']):
        return {
            "desc": "Mengasah kreativitas, estetika, rasa seni, dan kemampuan visualisasi ide ke dalam bentuk karya.",
            "tip": "🎨 **Tips Sukses:** Sering-sering cari referensi (Pinterest/Behance) untuk memperkaya wawasan visual. Mulai bangun portofolio dari tugas-tugas kuliah ini. Jangan ragu eksperimen gaya baru!"
        }
    elif any(x in course_lower for x in ['bisnis', 'manajemen', 'marketing', 'pemasaran', 'ekonomi', 'entrepreneur']):
        return {
            "desc": "Mempelajari strategi bisnis, pengelolaan organisasi, dinamika pasar, dan perilaku konsumen.",
            "tip": "📊 **Tips Sukses:** Perbanyak baca studi kasus nyata (case study) perusahaan. Latih kemampuan presentasi dan networking karena soft skill ini sangat krusial di dunia bisnis."
        }
    elif any(x in course_lower for x in ['bahasa', 'english', 'mandarin', 'komunikasi', 'writing', 'speaking']):
        return {
            "desc": "Meningkatkan kemampuan verbal dan non-verbal untuk komunikasi efektif dalam konteks profesional.",
            "tip": "🗣️ **Tips Sukses:** Kuncinya adalah 'Active Speaking'. Jangan malu salah grammar saat bicara, yang penting berani ngomong dulu! Praktikkan dengan teman atau native speaker jika ada kesempatan."
        }
    elif any(x in course_lower for x in ['hotel', 'wisata', 'tour', 'kitchen', 'pastry', 'food']):
        return {
            "desc": "Mata kuliah praktikal yang berhubungan langsung dengan industri pelayanan, kuliner, dan pariwisata.",
            "tip": "👨‍🍳 **Tips Sukses:** Perhatikan detail kebersihan (hygiene) dan standar pelayanan (service excellence). Disiplin dan attitude adalah nilai jual utama di industri hospitality."
        }
    else:
        # Default tips jika tidak ada kata kunci yang cocok
        return {
            "desc": "Mata kuliah ini dirancang untuk memperkuat kompetensi dasar atau keahlian spesifik di jurusan kamu.",
            "tip": "📝 **Tips Sukses:** Catat poin-poin penting dosen yang tidak ada di slide. Aktif bertanya dan berdiskusi di kelas bisa jadi nilai tambah untuk pemahamanmu."
        }

def detect_chatbot_responses(user_input):
    user_input_lower = user_input.lower()
    responses_shown = []
    
    if "tidur" in user_input_lower or "rebahan" in user_input_lower or "malas" in user_input_lower:
        st.info("😴 Wah, butuh istirahat ya? Sayangnya belum ada jurusan 'Tidur', tapi coba cek matkul santai ini...")
        responses_shown.append("relax")
    
    if "duit" in user_input_lower or "uang" in user_input_lower or "kaya" in user_input_lower or "cuan" in user_input_lower:
        st.success("💰 Orientasi masa depan mantap! Cek mata kuliah bisnis ini biar makin cuan.")
        responses_shown.append("money")
    
    if "game" in user_input_lower or "gaming" in user_input_lower:
        st.success("🎮 Daripada cuma main, mending bikin gamenya di jurusan ini!")
        responses_shown.append("game")
    
    if ("menggambar" in user_input_lower or "gambar" in user_input_lower or "seni" in user_input_lower or 
        "melukis" in user_input_lower or "desain" in user_input_lower) and "game" not in responses_shown:
        st.success("🎨 Kreativitas tanpa batas! Jurusan desain ini cocok buat kamu yang suka berkarya.")
        responses_shown.append("art")
    
    if "musik" in user_input_lower or "nyanyi" in user_input_lower or "band" in user_input_lower:
        st.info("🎵 Passion di musik? Cek mata kuliah ini untuk mengasah skill kamu!")
        responses_shown.append("music")
    
    if "olahraga" in user_input_lower or "sport" in user_input_lower or "fitness" in user_input_lower or "atlet" in user_input_lower:
        st.success("⚽ Sehat itu penting! Lihat mata kuliah yang cocok untuk kamu yang aktif.")
        responses_shown.append("sports")
    
    if ("komunikasi" in user_input_lower or "presenter" in user_input_lower or "mc" in user_input_lower or 
        "public speaking" in user_input_lower) and "game" not in responses_shown:
        st.success("🎤 Jago ngomong? Perfect! Ini mata kuliah untuk kamu yang suka berkomunikasi.")
        responses_shown.append("communication")
    
    if "masak" in user_input_lower or "memasak" in user_input_lower or "kuliner" in user_input_lower or "chef" in user_input_lower:
        st.success("👨‍🍳 MasterChef vibes! Cek mata kuliah kuliner dan hospitality ini.")
        responses_shown.append("culinary")
    
    if ("jalan" in user_input_lower and "jalan" in user_input_lower) or "traveling" in user_input_lower or "wisata" in user_input_lower or "tour" in user_input_lower:
        st.success("✈️ Hobi jalan-jalan? Ini mata kuliah pariwisata yang cocok buat kamu!")
        responses_shown.append("travel")
    
    if ("akuntansi" in user_input_lower or "akuntan" in user_input_lower) and "money" not in responses_shown:
        st.info("📊 Teliti sama angka? Akuntansi bisa jadi pilihan karir cemerlang!")
        responses_shown.append("accounting")
    
    if "bahasa" in user_input_lower or "english" in user_input_lower or "mandarin" in user_input_lower or "translator" in user_input_lower:
        st.success("🗣️ Multilingual skill itu valuable! Lihat program bahasa yang tersedia.")
        responses_shown.append("language")
    
    if "data" in user_input_lower or "analytics" in user_input_lower or "ai" in user_input_lower or "machine learning" in user_input_lower:
        st.success("📈 Data is the new oil! Cek jurusan Data Science dan AI ini.")
        responses_shown.append("data")
    
    if "film" in user_input_lower or "video" in user_input_lower or "sinematografi" in user_input_lower or "youtuber" in user_input_lower:
        st.success("🎬 Content creator masa depan! Ini mata kuliah media dan film untuk kamu.")
        responses_shown.append("film")
    
    return len(responses_shown) > 0

def process_negation(user_input):
    negation_patterns = [
        r'\b(tidak\s+suka|gak\s+suka|ga\s+suka)\s+(\w+)',
        r'\b(benci)\s+(\w+)',
        r'\b(anti)\s+(\w+)',
    ]
    
    words_to_remove = []
    cleaned_text = user_input.lower()
    
    for pattern in negation_patterns:
        matches = re.finditer(pattern, cleaned_text)
        for match in matches:
            if len(match.groups()) >= 2:
                negated_word = match.group(len(match.groups()))
                words_to_remove.append(negated_word)
                cleaned_text = cleaned_text.replace(match.group(0), '')
    
    if words_to_remove:
        st.warning(f"⚠️ Sistem mendeteksi kata yang tidak disukai: {', '.join(words_to_remove)}. Mata kuliah terkait akan dihindari.")
    
    return cleaned_text, words_to_remove

def analyze_sentiment(user_input):
    try:
        blob = TextBlob(user_input)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        if polarity > 0.5:
            st.success("😊 Wow, semangat banget! Energi positif kamu keren! Mari kita cari mata kuliah yang pas.")
        elif polarity > 0.1:
            st.info("🙂 Terlihat antusias! Yuk kita cari rekomendasi terbaik untuk kamu.")
        elif polarity < -0.1:
            st.info("🤔 Kayaknya masih bingung ya? Tenang, sistem ini akan bantu kamu menemukan arah yang tepat!")
        
        if subjectivity > 0.7:
            st.caption("💭 Tips: Semakin spesifik minat kamu, semakin akurat rekomendasinya!")
        
        return {
            'polarity': polarity,
            'subjectivity': subjectivity,
            'sentiment': 'positive' if polarity > 0.1 else ('negative' if polarity < -0.1 else 'neutral')
        }
    except:
        return {'polarity': 0, 'subjectivity': 0, 'sentiment': 'neutral'}

def expand_query(user_query, selected_keywords=None):
    expanded_query = user_query.lower()
    
    # 1. Expand dari KEYWORD_MAPPING
    for keyword, expansion in KEYWORD_MAPPING.items():
        if keyword in expanded_query:
            expanded_query += ' ' + expansion
    
    # 2. Tambahkan keyword yang dipilih pengguna
    if selected_keywords:
        for keyword in selected_keywords:
            if keyword in KEYWORD_MAPPING:
                expanded_query += ' ' + KEYWORD_MAPPING[keyword]
            else:
                # Untuk keyword yang tidak ada di mapping (mungkin hanya kata kunci utama)
                expanded_query += ' ' + keyword
    
    return expanded_query

def get_recommendations(user_query, df_filtered, words_to_remove=None, selected_keywords=None, top_n=10):
    if not user_query.strip() and not selected_keywords:
        return pd.DataFrame()
    
    if df_filtered.empty:
        return pd.DataFrame()
    
    if words_to_remove:
        for word in words_to_remove:
            df_filtered = df_filtered[~df_filtered['combined_features'].str.lower().str.contains(word, na=False)]
    
    if df_filtered.empty:
        return pd.DataFrame()
    
    # Gunakan selected_keywords dalam proses expansion
    expanded_query = expand_query(user_query, selected_keywords)
    
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df_filtered['combined_features'])
    query_vec = tfidf_vectorizer.transform([expanded_query])
    
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    df_results = df_filtered.copy()
    df_results['Similarity Score'] = cosine_similarities
    
    df_results = df_results[df_results['Similarity Score'] > 0]
    df_results = df_results.sort_values('Similarity Score', ascending=False).head(top_n)
    df_results['Similarity Score'] = (df_results['Similarity Score'] * 100).round(2)
    
    return df_results[['Program', 'Semester', 'Course', 'Similarity Score']]

def recommend_career_paths(courses_list):
    if not courses_list:
        return []
    
    career_mapping = {
        'Informatika': ['Software Engineer', 'Full Stack Developer', 'DevOps Engineer', 'System Analyst', 'IT Consultant'],
        'Data Science': ['Data Scientist', 'Data Analyst', 'Machine Learning Engineer', 'Business Intelligence Analyst', 'AI Researcher'],
        'Desain Komunikasi Visual': ['Graphic Designer', 'UI/UX Designer', 'Art Director', 'Brand Designer', 'Creative Director'],
        'Desain Interaktif': ['UX Designer', 'Game Designer', 'Motion Graphics Designer', 'Interactive Media Designer', 'Web Designer'],
        'Manajemen': ['Business Manager', 'Project Manager', 'Marketing Manager', 'HR Manager', 'Entrepreneur'],
        'Akuntansi': ['Accountant', 'Tax Consultant', 'Financial Analyst', 'Auditor', 'Finance Manager'],
        'Sistem Informasi': ['System Analyst', 'Business Analyst', 'IT Project Manager', 'ERP Consultant', 'Database Administrator'],
        'Bahasa Inggris': ['Translator', 'Teacher', 'Content Writer', 'Editor', 'International Relations Specialist'],
        'Bahasa Mandarin': ['Mandarin Translator', 'Language Teacher', 'International Business Specialist', 'Tour Guide'],
        'Bisnis Digital': ['Digital Marketing Specialist', 'E-commerce Manager', 'Social Media Manager', 'Digital Strategist', 'Growth Hacker'],
        'Hospitality dan Pariwisata': ['Hotel Manager', 'Event Planner', 'Tour Guide', 'Travel Consultant', 'F&B Manager'],
        'Ilmu Komunikasi': ['Public Relations Specialist', 'Journalist', 'Content Creator', 'Social Media Manager', 'Communications Manager']
    }
    
    course_keywords_mapping = {
        'programming|algoritma|web|mobile|software': ['Software Developer', 'Programmer', 'Web Developer'],
        'data|analytics|machine learning|ai|artificial': ['Data Scientist', 'Data Analyst', 'ML Engineer'],
        'desain|design|visual|grafis|ui|ux': ['Designer', 'UI/UX Designer', 'Graphic Designer'],
        'game|gaming|interactive': ['Game Developer', 'Game Designer'],
        'bisnis|business|manajemen|marketing': ['Business Analyst', 'Marketing Specialist', 'Manager'],
        'akuntansi|accounting|finance|keuangan': ['Accountant', 'Financial Analyst'],
        'komunikasi|media|jurnalis|public': ['Communications Specialist', 'Media Specialist', 'Journalist'],
        'hospitality|pariwisata|tourism|hotel': ['Tourism Professional', 'Hotel Manager']
    }
    
    programs = [course['Program'] for course in courses_list]
    course_names = ' '.join([course['Course'].lower() for course in courses_list])
    
    recommended_careers = set()
    
    for program in programs:
        for key, value in career_mapping.items():
            if key in program:
                recommended_careers.update(value[:3])
    
    for pattern, careers in course_keywords_mapping.items():
        if re.search(pattern, course_names):
            recommended_careers.update(careers)
    
    return list(recommended_careers)[:8]

# ==========================================
# BAGIAN 2: UI/UX & LANDING PAGE
# ==========================================

def local_css():
    st.markdown("""
    <style>
    /* Import Font */
    @import url('https://fonts.googleapis.com/css2?family=Segoe+UI&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Background Gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    /* --- TEKS UMUM (PUTIH) --- */
    h1, h2, h3, .css-10trblm, .stMarkdown p, .stMarkdown li, label {
        color: white !important;
    }

    /* --- KARTU HASIL (HITAM) --- */
    .result-card, .result-card div, .result-card h3, .result-card p {
        color: #31333F !important; 
    }
    
    /* --- EXPANDER TIPS (HITAM) --- */
    /* Mengatur warna teks di dalam expander agar hitam */
    .streamlit-expanderHeader {
        color: #31333F !important;
        font-weight: 600;
        background-color: #e6e9ef !important;
        border-radius: 8px;
    }
    .streamlit-expanderContent {
        background-color: #ffffff !important;
        color: #31333F !important;
        border-bottom-left-radius: 8px;
        border-bottom-right-radius: 8px;
        border: 1px solid #e6e9ef;
    }
    .streamlit-expanderContent p, .streamlit-expanderContent li {
        color: #31333F !important; /* Paksa teks hitam di dalam tips */
    }

    /* --- SIDEBAR (HITAM) --- */
    section[data-testid="stSidebar"] {
        background-color: white !important;
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] span, 
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] li {
        color: #2d3748 !important;
    }

    /* --- LANDING PAGE --- */
    .landing-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 20px;
        color: white;
    }
    .bot-icon {
        font-size: 80px;
        background: rgba(255, 255, 255, 0.2);
        width: 140px;
        height: 140px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 20px auto;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    .main-title {
        font-size: 48px;
        font-weight: 700;
        margin-bottom: 10px;
        text-align: center;
        color: #ffffff !important;
    }
    .subtitle {
        font-size: 20px;
        color: rgba(255, 255, 255, 0.9);
        margin-bottom: 40px;
        text-align: center;
    }
    
    /* Feature Cards Landing */
    .feature-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        height: 100%;
        color: #2d3748;
    }
    /* Paksa teks dalam kartu fitur jadi hitam juga */
    .feature-title { font-size: 20px; font-weight: 600; color: #2d3748 !important; margin-bottom: 10px; }
    .feature-desc { font-size: 15px; color: #718096 !important; }
    .feature-icon { font-size: 40px; margin-bottom: 15px; }

    /* Button Styles */
    div.stButton > button {
        background: #ffffff !important;
        color: #667eea !important;
        border: none !important;
        padding: 15px 40px !important;
        font-size: 18px !important;
        font-weight: 600 !important;
        border-radius: 50px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2) !important;
        width: 100%;
        margin: 0 auto !important;
        display: block !important;
    }
    div.stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 30px rgba(0, 0, 0, 0.3) !important;
        color: #764ba2 !important;
    }
    </style>
    """, unsafe_allow_html=True)

def render_landing_page():
    st.markdown("""
        <div class="landing-container">
            <div class="bot-icon">🎓</div>
            <div class="main-title">AI Course Advisor</div>
            <div class="subtitle">Dapatkan rekomendasi mata kuliah yang sesuai dengan minat dan tujuan karir Anda</div>
        </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1, 1]) 
    with c2:
        if st.button("Mulai Chat Sekarang 💬", use_container_width=True):
            st.session_state['app_started'] = True
            st.rerun()

    st.markdown("<br><br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="feature-card"><div class="feature-icon">🎯</div><div class="feature-title">Rekomendasi Personal</div><div class="feature-desc">Saran sesuai minatmu</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="feature-card"><div class="feature-icon">⚡</div><div class="feature-title">Respons Cepat</div><div class="feature-desc">Jawaban instan 24/7</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="feature-card"><div class="feature-icon">✨</div><div class="feature-title">Mudah Digunakan</div><div class="feature-desc">Tinggal ketik & tanya</div></div>', unsafe_allow_html=True)

# Tambahan: Ambil Keyword Utama dari Mapping
def get_main_keywords():
    return sorted(list(KEYWORD_MAPPING.keys()))

def main_app():
    df = load_data()
    
    # Inisialisasi session state untuk keyword yang dipilih
    if 'selected_keywords' not in st.session_state:
        st.session_state.selected_keywords = []

    # --- SIDEBAR ---
    with st.sidebar:
        st.title("🔍 Menu")
        if st.button("🏠 Home"):
            st.session_state['app_started'] = False
            st.session_state.selected_keywords = [] # Reset keyword
            st.rerun()
        
        st.markdown("---")
        st.subheader("Filter Data")
        program_list = ["Semua Jurusan"] + sorted(df['Program'].unique().tolist())
        selected_program = st.selectbox("Program Studi", options=program_list)
        semester_list = ["Semua Semester"] + sorted(df['Semester'].unique().tolist())
        selected_semester = st.selectbox("Semester", options=semester_list)
        
        st.markdown("---")
        st.subheader("Bookmark")
        if st.session_state.bookmarks:
            st.info(f"{len(st.session_state.bookmarks)} tersimpan")
            with st.expander("Lihat Daftar"):
                for idx, bm in enumerate(st.session_state.bookmarks):
                    st.markdown(f"**{bm['Course']}**")
                    # Menggunakan st.form dan st.form_submit_button untuk tombol di dalam expander
                    with st.form(key=f"del_form_{idx}"):
                        if st.form_submit_button("Hapus", type="secondary"):
                            st.session_state.bookmarks.pop(idx)
                            st.rerun()
                    st.divider()
            if st.button("Clear All"):
                st.session_state.bookmarks = []
                st.rerun()
            st.markdown("---")
            st.subheader("Karir")
            careers = recommend_career_paths(st.session_state.bookmarks)
            if careers:
                for c in careers:
                    st.markdown(f"- {c}")
        else:
            st.caption("Belum ada bookmark.")

    # --- MAIN CONTENT ---
    st.markdown('<h1 style="text-align: center;">🎓 AI Course Advisor</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #f0f0f0;">Ceritakan minatmu, dan AI akan mencarikan mata kuliah yang pas!</p>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # 1. CARA MENGGUNAKAN
    st.markdown("""
    ### 💡 Cara Menggunakan:
    1. **Pilih Filter** di sidebar (jurusan & semester) - opsional.
    2. **Ketik minat/hobi** kamu dengan bahasa santai.
    3. Jika kamu ragu, tambahkan **Keyword Pembantu** di bawah input teks.
    4. Klik tombol **Cari** untuk melihat hasil!
    """)
    st.markdown("<br>", unsafe_allow_html=True)

    # Filter logic
    df_filtered = df.copy()
    if selected_program != "Semua Jurusan":
        df_filtered = df_filtered[df_filtered['Program'] == selected_program]
    if selected_semester != "Semua Semester":
        df_filtered = df_filtered[df_filtered['Semester'] == selected_semester]

    # 2. INPUT SECTION
    c_in, c_btn = st.columns([4, 1])
    with c_in:
        user_input = st.text_area("Minat", placeholder="Contoh: Saya suka desain tapi tidak suka hitungan...", height=80, key="user_input_area", label_visibility="collapsed")
    with c_btn:
        st.markdown("<br>", unsafe_allow_html=True) 
        btn_cari = st.button("Cari 🚀")
    
    # Bagian Baru: Memilih Keyword
    # Menggunakan session state untuk menyimpan pilihan keyword
    
    # 3. SELECT KEYWORD SECTION
    st.markdown("### ✨ Keyword Pembantu (Opsional)")
    
    # Menampilkan daftar keyword untuk dipilih
    main_keywords = get_main_keywords()
    
    # Gunakan st.multiselect untuk memilih beberapa keyword
    selected_keywords_update = st.multiselect(
        "Pilih kata kunci yang paling mewakili minatmu:",
        options=main_keywords,
        default=st.session_state.selected_keywords,
        key="keyword_multiselect"
    )
    # Update session state saat ada perubahan
    st.session_state.selected_keywords = selected_keywords_update

    # 4. HASIL PENCARIAN
    if btn_cari:
        if not user_input and not st.session_state.selected_keywords:
            st.warning("Mohon masukkan minat kamu atau pilih minimal satu Keyword Pembantu!")
        else:
            st.markdown("---")
            with st.spinner("Sedang berpikir..."):
                detect_chatbot_responses(user_input)
                cleaned_input, words_to_remove = process_negation(user_input)
                
                # Panggil get_recommendations dengan selected_keywords
                recs = get_recommendations(cleaned_input, df_filtered, words_to_remove, st.session_state.selected_keywords)
                
                if not recs.empty:
                    st.subheader(f"Hasil: {len(recs)} Mata Kuliah")
                    for idx, row in recs.iterrows():
                        prog_desc = get_program_description(row['Program'])
                        advice = get_course_advice(row['Course']) # Ambil Tips Cerdas

                        # Kartu Hasil (Warna Abu muda #f0f2f6 & Teks Hitam)
                        st.markdown(f"""
                        <div class="result-card" style="background: #f0f2f6; padding: 20px; border-radius: 15px; margin-bottom: 15px; border-left: 5px solid #667eea;">
                            <h3 style="margin:0; font-weight: 700;">{row['Course']}</h3>
                            <p style="margin:5px 0 0 0; font-size: 0.9rem;">
                                🎓 {row['Program']} | 📅 Semester {row['Semester']} | ⭐ {row['Similarity Score']}%
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # --- BAGIAN EXPANDER (TIPS & DESKRIPSI) ---
                        with st.expander(f"💡 Lihat Tips & Deskripsi Matkul: {row['Course']}"):
                            st.markdown(f"""
                            **ℹ️ Deskripsi Jurusan:** {prog_desc}
                            
                            ---
                            
                            **📝 Info Mata Kuliah:** {advice['desc']}
                            
                            ---
                            {advice['tip']}
                            """)

                        # Tombol Simpan Bookmark
                        is_saved = any(b['Course'] == row['Course'] for b in st.session_state.bookmarks)
                        if not is_saved:
                            # Gunakan st.form untuk menghindari masalah tombol di Streamlit
                            with st.form(key=f"save_form_{idx}"):
                                if st.form_submit_button(f"🔖 Simpan", type="primary"):
                                    st.session_state.bookmarks.append(row.to_dict())
                                    st.rerun()
                        else:
                            st.button(f"✅ Tersimpan", key=f"saved_{idx}", disabled=True)
                        
                        st.markdown("<br>", unsafe_allow_html=True) # Spacer antar kartu
                else:
                    st.warning("Tidak ditemukan yang cocok. Coba ganti kata kunci atau hapus filter.")

    # 5. INFO TAMBAHAN
    st.markdown("---")
    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        with st.expander("📖 Kata Kunci yang Dipahami Sistem"):
            st.markdown("""
            Sistem memahami berbagai kata santai seperti:
            - **Kreatif/Seni:** menggambar, gambar, seni, desain
            - **Bisnis:** jualan, dagang, bisnis
            - **Teknologi:** ngoding, coding, komputer
            - **Keuangan:** hitung, angka, uang
            - **Pariwisata:** jalan-jalan, traveling, pariwisata
            - **Kuliner:** masak, memasak, kuliner
            - **Media:** komunikasi, film, musik
            - **Data:** data, analytics, machine learning
            - **Olahraga:** olahraga, sport, fitness
            """)
    
    with col_exp2:
        with st.expander("ℹ️ Tentang Sistem"):
            st.markdown("""
            **Sistem Rekomendasi Mata Kuliah UBM** menggunakan algoritma *TF-IDF & Cosine Similarity* untuk mencocokkan minat kamu dengan kurikulum yang tersedia.
            
            📊 Total Database: 356 mata kuliah
            
            🎯 Algoritma: TF-IDF + Cosine Similarity
            
            🧠 Smart Search: Keyword Expansion
            
            🎓 Program Studi: 12 jurusan
            
            📅 Semester: 1 - 8
            
            Sistem ini menggunakan AI untuk menemukan mata kuliah yang paling sesuai dengan minat dan hobi Anda!
            """)

def main():
    st.set_page_config(page_title="AI Course Advisor", page_icon="🎓", layout="wide")
    local_css()
    
    if 'bookmarks' not in st.session_state:
        st.session_state.bookmarks = []
    if 'app_started' not in st.session_state:
        st.session_state['app_started'] = False
        
    if not st.session_state['app_started']:
        render_landing_page()
    else:
        main_app()

if __name__ == "__main__":
    main()
