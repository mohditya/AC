from flask import Flask, request, jsonify
from google import genai
import base64
import os
from pathlib import Path
import time
import traceback

app = Flask(__name__)
from flask import send_from_directory

@app.route('/manifest.json')
def serve_manifest():
    return send_from_directory('.', 'manifest.json')
# Configure Gemini API with your key from Environment Variables
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

if not GEMINI_API_KEY:
    # This prevents the app from crashing silently if the key is missing
    print("⚠️ WARNING: GEMINI_API_KEY not found! Set it in your environment variables.")
client = genai.Client(api_key=GEMINI_API_KEY)

# Language mapping for Gemini - ALL 60+ Indian Languages
LANGUAGE_NAMES = {
    'en': 'English',
    'hi': 'Hindi (हिन्दी)',
    'bn': 'Bengali (বাংলা)',
    'te': 'Telugu (తెలుగు)',
    'mr': 'Marathi (मराठी)',
    'ta': 'Tamil (தமிழ்)',
    'gu': 'Gujarati (ગુજરાતી)',
    'kn': 'Kannada (ಕನ್ನಡ)',
    'ml': 'Malayalam (മലയാളം)',
    'pa': 'Punjabi (ਪੰਜਾਬੀ)',
    'or': 'Odia (ଓଡ଼ିଆ)',
    'as': 'Assamese (অসমীয়া)',
    'ur': 'Urdu (اردو)',
    'sa': 'Sanskrit (संस्कृतम्)',
    'pra': 'Prakrit (प्राकृतम्)',
    'ks': 'Kashmiri (कॉशुर)',
    'ne': 'Nepali (नेपाली)',
    'sd': 'Sindhi (सिन्धी)',
    'kok': 'Konkani (कोंकणी)',
    'mai': 'Maithili (मैथिली)',
    'mni': 'Manipuri (মৈতৈলোন্)',
    'brx': 'Bodo (बड़ो)',
    'doi': 'Dogri (डोगरी)',
    'sat': 'Santali (ᱥᱟᱱᱛᱟᱲᱤ)',
    'bho': 'Bhojpuri (भोजपुरी)',
    'raj': 'Rajasthani (राजस्थानी)',
    'hne': 'Chhattisgarhi (छत्तीसगढ़ी)',
    'awa': 'Awadhi (अवधी)',
    'mag': 'Magahi (मगही)',
    'anp': 'Angika (अंगिका)',
    'bfy': 'Bajjika (बज्जिका)',
    'bun': 'Bundeli (बुन्देली)',
    'kfy': 'Kumaoni (कुमाऊँनी)',
    'gbm': 'Garhwali (गढ़वळि)',
    'tcy': 'Tulu (ತುಳು)',
    'kfa': 'Kodava',
    'kha': 'Khasi',
    'lus': 'Mizo',
    'trp': 'Kokborok',
    'ajz': 'Karbi',
    'hoc': 'Ho',
    'unr': 'Mundari',
    'kru': 'Kurukh',
    'gon': 'Gondi (గోండీ)',
    'lmn': 'Lambadi/Banjara (లంబాడీ)',
    'lep': 'Lepcha',
    'lif': 'Limbu',
    'njm': 'Angami',
    'gju': 'Gujjari (गुज्जरी)',
    'kfr': 'Kachhi/Kutchi (કચ્છી)',
    'rkt': 'Kamtapuri (কামতাপুরী)',
    'kyw': 'Kurmali (कुड़माली)',
    'lbj': 'Ladakhi',
    'pli': 'Pali (पालि)',
    'sck': 'Sadri/Nagpuri (नागपुरी)',
    'sip': 'Sikkimese (སི་ཀི་མེས་)',
    'nic': 'Nicobarese',
    'skr': 'Saraiki (سرائیکی)',
    'unx': 'Sambalpuri (ସମ୍ବଲପୁରୀ)',
    'dha': 'Dhatki (धातकी)',
    'hil': 'Himachali/Pahadi (पहाड़ी)'
}

class GeminiAPIFallback:
    def __init__(self, client):
        self.client = client
        self.models = [
            {'name': 'Gemini 2.5 Flash', 'model_id': 'models/gemini-2.0-flash-exp'},
            {'name': 'Gemini 2.0 Flash', 'model_id': 'models/gemini-2.0-flash'},
            {'name': 'Gemini Flash Latest', 'model_id': 'models/gemini-flash-latest'},
        ]
        self.current_model_index = 0

    def analyze_image(self, image_path, description, language='en', user_symptoms=''):
        """Analyze medical image with Gemini - Multi-language support with optional symptoms"""
        with open(image_path, 'rb') as f:
            image_data = base64.standard_b64encode(f.read()).decode('utf-8')

        file_ext = Path(image_path).suffix.lower()
        mime_type = self._get_mime_type(file_ext)
        lang_name = LANGUAGE_NAMES.get(language, 'English')

        # Build symptoms section if provided
        symptoms_section = ""
        if user_symptoms and user_symptoms.strip():
            symptoms_section = (
                f"\n\n🗣️ PATIENT'S REPORTED SYMPTOMS:\n{user_symptoms}\n"
                "⚠️ IMPORTANT: Consider these symptoms along with the image analysis!\n"
            )

        for attempt in range(len(self.models)):
            model_info = self.models[self.current_model_index]
            model_name = model_info['name']
            model_id = model_info['model_id']
            print(f"\n🤖 Trying {model_name}...")

            prompt = f"""You are an expert medical analyst with knowledge of both modern medicine and traditional Ayurveda.

CRITICAL INSTRUCTION: Respond ENTIRELY in {lang_name} language ONLY. Do NOT mix with English.

Analyze this medical image:

Patient Description: {description}{symptoms_section}

Output Language: {lang_name}

Provide comprehensive analysis in this format (IN {lang_name}):

=================================================
📋 CLINICAL FINDINGS & SYMPTOMS
=================================================
[Visible symptoms and observations in {lang_name}]
{"[Also considering patient's reported symptoms]" if user_symptoms else ""}

=================================================
🎯 DISEASE IDENTIFICATION
=================================================
[Top 3 possible conditions in {lang_name}]

=================================================
⚠️ DISEASE SEVERITY
=================================================
Severity: [MILD/MODERATE/SEVERE/CRITICAL - in {lang_name}]
Score: [1-10]
Explanation: [In {lang_name}]

=================================================
💊 MODERN MEDICAL TREATMENT
=================================================
[Allopathic treatment in {lang_name}]
- Medications: [In {lang_name}]
- Therapy: [In {lang_name}]
- Lifestyle: [In {lang_name}]
- Emergency Signs: [In {lang_name}]

=================================================
🌿 AYURVEDIC TREATMENT
=================================================
[Ayurvedic approach in {lang_name}]
- Dosha: [Vata/Pitta/Kapha in {lang_name}]
- Herbs: [In {lang_name}]
- Therapies: [In {lang_name}]
- Diet: [In {lang_name}]
- Lifestyle: [In {lang_name}]

REMEMBER: Write ENTIRELY in {lang_name}!
NOTE:- PLEASE SAY THE BULLET POINTS such as Diet, Lifestyle etc entirelyin {lang_name}"""

            try:
                response = self.client.models.generate_content(
                    model=model_id,
                    contents={
                        "parts": [
                            {"inline_data": {"mime_type": mime_type, "data": image_data}},
                            {"text": prompt}
                        ]
                    }
                )
                print(f"✅ Success with {model_name}")
                return response.text

            except Exception as e:
                print(f"❌ {model_name} failed: {str(e)}")
                self.current_model_index = (self.current_model_index + 1) % len(self.models)
                time.sleep(1)
                continue

        raise Exception("All Gemini models failed")

    def _get_mime_type(self, file_ext):
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        return mime_types.get(file_ext.lower(), 'image/jpeg')

gemini_fallback = GeminiAPIFallback(client)

@app.route('/')
def index():
    try:
        # Try multiple possible filenames
        html_files = ['index.html', 'index-Copy.html', 'Aayushman.html']
        for html_file in html_files:
            if os.path.exists(html_file):
                with open(html_file, 'r', encoding='utf-8') as f:
                    return f.read()
        return "⚠ No HTML file found! Put index.html in the same folder!"
    except Exception as e:
        return f"⚠ HTML File error: {str(e)}"

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        img1 = request.files.get('bodyImage1')
        img2 = request.files.get('bodyImage2')
        desc1 = request.form.get('bodyDesc1', 'Medical Image 1')
        desc2 = request.form.get('bodyDesc2', 'Medical Image 2')
        language = request.form.get('language', 'en')
        user_symptoms = request.form.get('symptoms', '').strip()

        if not img1:
            return jsonify({'success': False, 'error': 'Image required'}), 400

        img1_path = f'temp_img1_{int(time.time())}.jpg'
        try:
            img1.save(img1_path)
            print(f"\n{'='*80}\n📸 Image 1: {desc1}\n🌐 {LANGUAGE_NAMES.get(language)}\n{'='*80}")
            analysis = gemini_fallback.analyze_image(img1_path, desc1, language, user_symptoms)

            if img2:
                img2_path = f'temp_img2_{int(time.time())}.jpg'
                try:
                    img2.save(img2_path)
                    print(f"\n{'='*80}\n📸 Image 2: {desc2}\n{'='*80}")
                    analysis2 = gemini_fallback.analyze_image(img2_path, desc2, language, user_symptoms)
                    analysis = f"{analysis}\n\n{'='*80}\n📸 IMAGE 2 ANALYSIS\n{'='*80}\n{analysis2}"
                finally:
                    if os.path.exists(img2_path):
                        os.remove(img2_path)

            return jsonify({'success': True, 'analysis': analysis})
        finally:
            if os.path.exists(img1_path):
                os.remove(img1_path)

    except Exception as e:
        print("ERROR in /analyze endpoint:")
        traceback.print_exc()
        return jsonify({'success': False, 'error': f"Failed: {str(e)}"}), 500

if __name__ == '__main__':
    print("\n" + "="*80)
    print("✅ AAYUSHMAN CHIKITSALAYA - LIVE")
    print("🌐 60+ Indian Languages")
    print("🤖 Gemini API with Auto-Fallback")
    print("🏥 Ayurvedic + Modern Medicine")
    print("💬 Optional Symptoms Input Support")
    print("="*80)
    app.run()
