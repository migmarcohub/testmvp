import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from itertools import chain

# ==========================================
# BAGIAN 1: KONFIGURASI DAN DATA
# ==========================================

# MAPPING KEYWORD: Memastikan semua program terwakili dan mencakup minat luas
KEYWORD_MAPPING = {
    # --- Kategori Kreatif/Seni ---
    "menggambar": "desain visual art seni fotografi kreatif sketsa ilustrasi grafis",
    "seni": "desain visual art seni fotografi kreatif sketsa ilustrasi grafis",
    "desain": "desain visual kreatif grafis komunikasi media digital",
    "film": "film broadcasting multimedia produksi sinema animasi video",
    "musik": "musik audio sound production recording entertainment",
    
    # --- Kategori Bisnis/Manajemen/Keuangan ---
    "jualan": "marketing bisnis manajemen pemasaran retail sales perdagangan kewirausahaan entrepreneur",
    "bisnis": "marketing bisnis manajemen pemasaran retail sales perdagangan kewirausahaan entrepreneur",
    "hitung": "akuntansi statistika matematika ekonomi keuangan pajak finance analisis",
    "keuangan": "akuntansi statistika matematika ekonomi keuangan pajak finance analisis",
    
    # --- Kategori Teknologi/Data ---
    "ngoding": "teknologi informasi sistem komputer data algoritma programming python web software aplikasi digital informatika",
    "komputer": "teknologi informasi sistem komputer data algoritma programming python web software aplikasi digital informatika",
    "data": "data science analytics statistika machine learning artificial intelligence AI",
    "game": "game development interactive design programming unity multimedia",
    
    # --- Kategori Kemanusiaan/Pariwisata/Kuliner ---
    "komunikasi": "komunikasi media jurnalistik broadcast public relations PR",
    "emosi": "psikologi perilaku mental manusia konseling organisasi", 
    "orang": "psikologi perilaku mental manusia konseling organisasi", 
    "jalan-jalan": "pariwisata hospitality hotel tour travel guide tourism wisata perhotelan",
    "kuliner": "food beverage tata boga kitchen pastry kuliner makanan minuman chef",
    "bahasa": "bahasa inggris mandarin sastra translation interpreter",
    
    # --- Kategori Lain ---
    "olahraga": "sport fitness kesehatan health wellness management",
}

# Deskripsi Program Studi
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

def get_main_keywords():
    return sorted(list(KEYWORD_MAPPING.keys()))

# ==========================================
# BAGIAN 2: FUNGSI PEMROSESAN DATA & REKOMENDASI
# ==========================================

@st.cache_data
def load_data():
    """Memuat data mata kuliah dari file CSV."""
    file_name = 'List Mata Kuliah UBM.xlsx - Sheet1.csv' 
    try:
        df = pd.read_csv(file_name)
        df = df.dropna()
        df['combined_features'] = df['Course'].astype(str) + ' ' + df['Program'].astype(str)
        return df
    except FileNotFoundError:
        st.error(f"File data '{file_name}' tidak ditemukan. Mohon pastikan file ada.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Gagal memuat data. Error: {e}")
        return pd.DataFrame()

def get_program_description(program_name):
    """Mengambil deskripsi singkat program studi."""
    return PROGRAM_DESCRIPTIONS.get(program_name, "Jurusan unggulan yang siap mencetak profesional handal di bidangnya.")

def get_course_advice(course_name):
    """Memberikan saran dan tips belajar berdasarkan kategori mata kuliah."""
    course_lower = course_name.lower()
    
    if any(x in course_lower for x in ['matematika', 'kalkulus', 'statistika', 'akuntansi', 'keuangan', 'fisika', 'ekonomi']):
        return {
            "desc": "Mata kuliah ini banyak melibatkan logika, rumus, perhitungan, dan ketelitian angka.",
            "tip": "💡 **Tips Sukses:** Jangan hanya menghapal rumus, tapi pahami konsep dasarnya. Perbanyak latihan soal mandiri agar terbiasa dengan berbagai variasi kasus perhitungan."
        }
    elif any(x in course_lower for x in ['program', 'coding', 'algoritma', 'data', 'sistem', 'web', 'mobile', 'software', 'komputer']):
        return {
            "desc": "Fokus pada pengembangan logika teknis, struktur data, dan penulisan kode (coding) untuk membangun aplikasi.",
            "tip": "💻 **Tips Sukses:** Praktek langsung (ngoding) jauh lebih efektif daripada cuma baca teori. Jangan takut error, itu bagian dari proses belajar! Manfaatkan sumber belajar online seperti StackOverflow."
        }
    elif any(x in course_lower for x in ['desain', 'gambar', 'visual', 'art', 'sketsa', 'nirmana', 'tipografi', 'film', 'sinema']):
        return {
            "desc": "Mengasah kreativitas, estetika, rasa seni, dan kemampuan visualisasi ide ke dalam bentuk karya.",
            "tip": "🎨 **Tips Sukses:** Sering-sering cari referensi (Pinterest/Behance) untuk memperkaya wawasan visual. Mulai bangun portofolio dari tugas-tugas kuliah ini. Jangan ragu eksperimen gaya baru!"
        }
    elif any(x in course_lower for x in ['bisnis', 'manajemen', 'marketing', 'pemasaran', 'entrepreneur', 'organisasi', 'hukum']):
        return {
            "desc": "Mempelajari strategi bisnis, pengelolaan organisasi, dinamika pasar, dan perilaku konsumen.",
            "tip": "📊 **Tips Sukses:** Perbanyak baca studi kasus nyata (case study) perusahaan. Latih kemampuan presentasi dan networking karena soft skill ini sangat krusial di dunia bisnis."
        }
    elif any(x in course_lower for x in ['bahasa', 'english', 'mandarin', 'komunikasi', 'writing', 'speaking', 'psikologi', 'perilaku']):
        return {
            "desc": "Meningkatkan kemampuan verbal, non-verbal, dan pemahaman perilaku manusia untuk komunikasi efektif.",
            "tip": "🗣️ **Tips Sukses:** Untuk bahasa, kuncinya adalah 'Active Speaking'. Untuk psikologi, perbanyak observasi dan analisis kasus nyata. Praktikkan di kehidupan sehari-hari."
        }
    elif any(x in course_lower for x in ['hotel', 'wisata', 'tour', 'kitchen', 'pastry', 'food', 'travel', 'hospitality']):
        return {
            "desc": "Mata kuliah praktikal yang berhubungan langsung dengan industri pelayanan, kuliner, dan pariwisata.",
            "tip": "👨‍🍳 **Tips Sukses:** Perhatikan detail kebersihan (hygiene) dan standar pelayanan (service excellence). Disiplin dan attitude adalah nilai jual utama di industri hospitality."
        }
    else:
        return {
            "desc": "Mata kuliah ini dirancang untuk memperkuat kompetensi dasar atau keahlian spesifik di jurusan kamu.",
            "tip": "📝 **Tips Sukses:** Catat poin-poin penting dosen yang tidak ada di slide. Aktif bertanya dan berdiskusi di kelas bisa jadi nilai tambah untuk pemahamanmu."
        }

def process_negation(user_input):
    """Mengidentifikasi dan menghapus kata-kata yang tidak disukai dari input pengguna."""
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

def expand_query(user_query, selected_keywords=None):
    """Memperluas query pengguna dengan kata kunci dari KEYWORD_MAPPING."""
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
                expanded_query += ' ' + keyword
    
    return expanded_query

def forced_recommendation_based_on_keyword(selected_keywords, df_filtered):
    """Memberikan rekomendasi paksa jika pencarian utama (similarity) gagal."""
    program_match = []
    
    keyword_to_program = {
        'film': ['Ilmu Komunikasi', 'Desain Komunikasi Visual', 'Desain Interaktif'],
        'komunikasi': ['Ilmu Komunikasi'],
        'emosi': ['Psikologi'],
        'orang': ['Psikologi', 'Ilmu Komunikasi'],
        'seni': ['Desain Komunikasi Visual', 'Desain Interaktif'],
        'ngoding': ['Informatika', 'Data Science', 'Sistem Informasi', 'Bisnis Digital'],
        'bisnis': ['Manajemen', 'Bisnis Digital', 'Sistem Informasi'],
        'hitung': ['Akuntansi', 'Manajemen', 'Data Science'],
        'kuliner': ['Hospitality dan Pariwisata'],
        'data': ['Data Science', 'Informatika', 'Sistem Informasi'],
        'bahasa': ['Bahasa Inggris', 'Bahasa Mandarin'],
        'desain': ['Desain Komunikasi Visual', 'Desain Interaktif'],
        'jalan-jalan': ['Hospitality dan Pariwisata'],
    }
    
    for kw in selected_keywords:
        program_match.extend(keyword_to_program.get(kw, []))
        
    unique_programs = list(set(program_match))
    
    if unique_programs:
        df_forced = df_filtered[df_filtered['Program'].isin(unique_programs)].copy()
        
        if not df_forced.empty:
            # Ambil 5 mata kuliah unik, gunakan random_state agar hasil konsisten
            if len(df_forced) > 5:
                return df_forced.sample(min(len(df_forced), 5), random_state=1)[['Program', 'Semester', 'Course']]
            else:
                return df_forced[['Program', 'Semester', 'Course']]
                
    return pd.DataFrame()

def get_recommendations(user_query, df_filtered, words_to_remove=None, selected_keywords=None, top_n=10):
    """Fungsi utama untuk mendapatkan rekomendasi menggunakan Cosine Similarity."""
    if not user_query.strip() and not selected_keywords:
        return pd.DataFrame()
    
    if df_filtered.empty:
        return pd.DataFrame()
    
    if words_to_remove:
        for word in words_to_remove:
            df_filtered = df_filtered[~df_filtered['combined_features'].str.lower().str.contains(word, na=False)]
    
    if df_filtered.empty:
        return pd.DataFrame()

    expanded_query = expand_query(user_query, selected_keywords)
    
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    
    try:
        tfidf_matrix = tfidf_vectorizer.fit_transform(df_filtered['combined_features'])
        query_vec = tfidf_vectorizer.transform([expanded_query])
    except ValueError:
        return pd.DataFrame() 
    
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    df_results = df_filtered.copy()
    df_results['Similarity Score'] = cosine_similarities
    
    df_results = df_results[df_results['Similarity Score'] > 0]
    df_results = df_results.sort_values('Similarity Score', ascending=False).head(top_n)
    
    # Logika Rekomendasi Paksa
    if df_results.empty and selected_keywords:
        st.info("⚠️ Pencarian Kosong. Mencoba rekomendasi paksa berdasarkan Keyword Pembantu...")
        df_forced = forced_recommendation_based_on_keyword(selected_keywords, df_filtered)
        if not df_forced.empty:
            df_forced['Similarity Score'] = 1.0 
            df_forced = df_forced.rename(columns={'Course': 'Course', 'Program': 'Program', 'Semester': 'Semester'})
            df_results = df_forced
    
    if not df_results.empty:
        df_results['Similarity Score'] = (df_results['Similarity Score'] * 100).round(2)
        return df_results[['Program', 'Semester', 'Course', 'Similarity Score']]
        
    return pd.DataFrame()

# ==========================================
# BAGIAN 3: USER INTERFACE (STREAMLIT APP)
# ==========================================

def render_landing_page():
    st.markdown('<h1 style="text-align: center;">Selamat Datang di AI Course Advisor 🎓</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center;">Temukan mata kuliah yang paling sesuai dengan minat dan passion kamu!</p>', unsafe_allow_html=True)
    st.markdown("---")
    st.image("https://images.unsplash.com/photo-1523287560-6b6a6c1171ac?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D", caption="Temukan karir yang sesuai passion.", use_column_width=True)
    st.markdown("### Mengapa Menggunakan Aplikasi Ini?")
    st.markdown("""
    * **Personalized:** Rekomendasi berdasarkan hobi, minat, bahkan hal yang tidak kamu sukai.
    * **Spesifik:** Hasilnya berupa daftar mata kuliah yang relevan, bukan hanya nama jurusan.
    * **Informatif:** Dapatkan tips belajar dan deskripsi singkat jurusan terkait.
    """)
    
    # PERBAIKAN ERROR (MENGGANTI use_column_width=True)
    if st.button("Mulai Konsultasi Sekarang 🚀", use_container_width=True, type="primary"):
        st.session_state['app_started'] = True
        st.rerun()

def main_app():
    df = load_data()
    
    if df.empty:
        st.error("Aplikasi tidak dapat berjalan karena data mata kuliah kosong atau gagal dimuat.")
        return

    # --- SIDEBAR (Filter dan Bookmark) ---
    with st.sidebar:
        st.title("🔍 Menu")
        if st.button("🏠 Home"):
            st.session_state['app_started'] = False
            st.session_state.selected_keywords = [] 
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
            st.info(f"{len(st.session_state.bookmarks)} mata kuliah tersimpan")
            with st.expander("Lihat Daftar"):
                for idx, bm in enumerate(st.session_state.bookmarks):
                    st.markdown(f"**{bm.get('Course', 'N/A')}** - {bm.get('Program', 'N/A')}")
                    with st.form(key=f"del_form_{idx}"):
                        if st.form_submit_button("Hapus", type="secondary"):
                            st.session_state.bookmarks.pop(idx)
                            st.rerun()
                    st.divider()
            if st.button("Clear All"):
                st.session_state.bookmarks = []
                st.rerun()
        else:
            st.caption("Belum ada mata kuliah yang di-bookmark.")

    # --- MAIN CONTENT ---
    st.markdown('<h1 style="text-align: center;">🎓 AI Course Advisor</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #f0f0f0;">Ceritakan minatmu, dan AI akan mencarikan mata kuliah yang pas!</p>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Filter logic
    df_filtered = df.copy()
    if selected_program != "Semua Jurusan":
        df_filtered = df_filtered[df_filtered['Program'] == selected_program]
    if selected_semester != "Semua Semester":
        df_filtered = df_filtered[df_filtered['Semester'] == selected_semester]

    # INPUT SECTION
    c_in, c_btn = st.columns([4, 1])
    with c_in:
        user_input = st.text_area("Minat", placeholder="Contoh: Saya suka desain tapi tidak suka hitungan...", height=80, key="user_input_area", label_visibility="collapsed")
    with c_btn:
        st.markdown("<br>", unsafe_allow_html=True) 
        btn_cari = st.button("Cari 🚀")
    
    # SELECT KEYWORD SECTION
    st.markdown("### ✨ Keyword Pembantu (Opsional)")
    main_keywords = get_main_keywords()
    
    selected_keywords_update = st.multiselect(
        "Pilih kata kunci yang paling mewakili minatmu:",
        options=main_keywords,
        default=st.session_state.selected_keywords,
        key="keyword_multiselect"
    )
    st.session_state.selected_keywords = selected_keywords_update

    # HASIL PENCARIAN
    st.markdown("---")
    if btn_cari:
        if not user_input and not st.session_state.selected_keywords:
            st.warning("Mohon masukkan minat kamu atau pilih minimal satu Keyword Pembantu!")
        else:
            with st.spinner("Sedang berpikir..."):
                cleaned_input, words_to_remove = process_negation(user_input)
                
                recs = get_recommendations(cleaned_input, df_filtered, words_to_remove, st.session_state.selected_keywords)
                
                if not recs.empty:
                    st.subheader(f"✅ Ditemukan: {len(recs)} Mata Kuliah Paling Relevan")
                    for idx, row in recs.iterrows():
                        prog_desc = get_program_description(row['Program'])
                        advice = get_course_advice(row['Course']) 

                        # Kartu Hasil
                        st.markdown(f"""
                        <div class="result-card" style="background: #e6f0ff; padding: 20px; border-radius: 10px; margin-bottom: 15px; border-left: 5px solid #007bff;">
                            <h3 style="margin:0; font-weight: 700; color: #004d99;">{row['Course']}</h3>
                            <p style="margin:5px 0 0 0; font-size: 0.9rem; color: #333;">
                                🎓 **{row['Program']}** | 📅 Semester **{row['Semester']}** | ⭐ **{row['Similarity Score']}%**
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # EXPANDER
                        with st.expander(f"💡 Lihat Tips & Deskripsi Matkul: {row['Course']}"):
                            st.markdown(f"**ℹ️ Deskripsi Jurusan:** {prog_desc}")
                            st.markdown("---")
                            st.markdown(f"**📝 Info Mata Kuliah:** {advice['desc']}")
                            st.markdown("---")
                            st.markdown(advice['tip'])

                        # Tombol Simpan Bookmark
                        is_saved = any(b['Course'] == row['Course'] for b in st.session_state.bookmarks)
                        col_save, _ = st.columns([1, 4])
                        with col_save:
                            if not is_saved:
                                with st.form(key=f"save_form_{idx}"):
                                    if st.form_submit_button(f"🔖 Simpan", type="primary"):
                                        st.session_state.bookmarks.append(row.to_dict())
                                        st.rerun()
                            else:
                                st.button(f"✅ Tersimpan", key=f"saved_{idx}", disabled=True)
                        
                        st.markdown("<br>", unsafe_allow_html=True) 
                else:
                    st.error("❌ Tidak ditemukan yang cocok. Coba ganti kata kunci atau hapus filter.")

# ==========================================
# BAGIAN 4: INICIALISASI
# ==========================================

def main():
    st.set_page_config(
        page_title="AI Course Advisor",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Inisialisasi Session State (Penting untuk mempertahankan data)
    if 'app_started' not in st.session_state:
        st.session_state['app_started'] = False
    if 'bookmarks' not in st.session_state:
        st.session_state['bookmarks'] = []
    if 'selected_keywords' not in st.session_state:
        st.session_state['selected_keywords'] = []

    if not st.session_state['app_started']:
        render_landing_page()
    else:
        main_app()

if __name__ == "__main__":
    main()
