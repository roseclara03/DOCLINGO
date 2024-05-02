from flask import Flask, render_template, request, send_file
from googletrans import Translator
import fitz  # PyMuPDF
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from collections import defaultdict
from fpdf import FPDF
import pyttsx3

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Define language codes and names
languages = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "ja": "Japanese",
    # Add more languages as needed
}

# Initialize NLTK and download resources
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Function to generate summary
def generate_summary(text, n):
    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

    # Calculate frequency distribution
    freq_dist = FreqDist(filtered_tokens)

    # Sort the frequency distribution in descending order
    sorted_freq = sorted(freq_dist.items(), key=lambda x: x[1], reverse=True)

    # Get the top n words
    top_words = [w[0] for w in sorted_freq[:n]]

    # Create a dictionary to store sentences containing top words
    sentence_rank = defaultdict(int)

    # Tokenize sentences
    sentences = text.split('.')

    # Calculate rank of each sentence
    for i, sentence in enumerate(sentences):
        for word in word_tokenize(sentence.lower()):
            if word in top_words:
                sentence_rank[i] += 1

    # Sort sentences by rank
    sorted_sentences = sorted(sentence_rank.items(), key=lambda x: x[1], reverse=True)

    # Generate summary
    summary = ''
    for i in range(n):
        summary += sentences[sorted_sentences[i][0]] + '.\n'

    return summary

# Function to translate PDF
def translate_pdf(input_path, source_language, target_language):
    # Open the PDF file
    pdf_document = fitz.open(input_path)

    # Initialize translator
    translator = Translator()

    # Create a new PDF document for translated content
    translated_pdf = fitz.open()

    translated_text = ""

    for page_num in range(pdf_document.page_count):
        # Extract text from the original PDF page
        page = pdf_document[page_num]
        text = page.get_text()

        # Translate the extracted text
        translated_page_text = translator.translate(text, src=source_language, dest=target_language).text

        translated_text += translated_page_text + "\n"

        # Create a new page in the translated PDF with the translated text
        translated_page = translated_pdf.new_page(width=page.rect.width, height=page.rect.height)
        translated_page.insert_text((10, 10), translated_page_text)

    # Save the translated PDF
    translated_pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], f"translated_{source_language}_{target_language}.pdf")
    translated_pdf.save(translated_pdf_path)
    translated_pdf.close()
    pdf_document.close()

    return translated_pdf_path

# Function to convert text to PDF
def text_to_pdf(text, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, text)
    pdf.output(filename)
    return filename

# Function to generate audio file from text using pyttsx3
def generate_audio(text, filename):
    engine = pyttsx3.init()
    audio_path = os.path.join('static', filename)  # Save audio in static folder
    engine.save_to_file(text, audio_path)
    engine.runAndWait()
    return audio_path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        source_language = request.form['source_language']
        target_language = request.form['target_language']
        input_pdf = request.files['input_pdf']

        if source_language and target_language and input_pdf:
            input_pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], input_pdf.filename)
            input_pdf.save(input_pdf_path)

            if 'submit_translated_pdf' in request.form:
                # Translate PDF and generate audio
                translated_pdf_path = translate_pdf(input_pdf_path, source_language, target_language)
                translated_text = ""
                with fitz.open(translated_pdf_path) as pdf:
                    for page in pdf:
                        translated_text += page.get_text()
                translated_audio_path = generate_audio(translated_text, 'translated_audio.mp3')
                return render_template('index.html', languages=languages, translated_pdf_path=translated_pdf_path, translated_audio_path=translated_audio_path)
            elif 'submit_translated_summary' in request.form:
                # Translate PDF and generate summary
                translated_pdf_path = translate_pdf(input_pdf_path, source_language, target_language)
                translated_text = ""
                with fitz.open(translated_pdf_path) as pdf:
                    for page in pdf:
                        translated_text += page.get_text()
                summary = generate_summary(translated_text, 3)  # Generating a summary of 3 sentences
                summary_pdf_path = text_to_pdf(summary, 'translated_summary.pdf')
                summary_audio_path = generate_audio(summary, 'summarized_audio.mp3')
                return render_template('index.html', languages=languages, translated_summary_path=summary_pdf_path, summarized_audio_path=summary_audio_path)

    return render_template('index.html', languages=languages)

@app.route('/download_summary', methods=['POST'])
def download_translated_summary():
    if request.method == 'POST':
        translated_summary_path = request.form['translated_summary_path']
        
        if translated_summary_path:
            return send_file(translated_summary_path, as_attachment=True)

    return "Error: Translated summary could not be downloaded."

@app.route('/download_translated_pdf', methods=['POST'])
def download_translated_pdf():
    if request.method == 'POST':
        translated_pdf_path = request.form['translated_pdf_path']
        
        if translated_pdf_path:
            return send_file(translated_pdf_path, as_attachment=True)

    return "Error: Translated PDF could not be downloaded."

@app.route('/download_translated_audio', methods=['POST'])
def download_translated_audio():
    if request.method == 'POST':
        translated_audio_path = request.form['translated_audio_path']
        
        if translated_audio_path:
            return send_file(translated_audio_path, as_attachment=True)

    return "Error: Translated audio could not be downloaded."

@app.route('/download_summarized_audio', methods=['POST'])
def download_summarized_audio():
    if request.method == 'POST':
        summarized_audio_path = request.form['summarized_audio_path']
        
        if summarized_audio_path:
            return send_file(summarized_audio_path, as_attachment=True)

    return "Error: Summarized audio could not be downloaded."

if __name__== '__main__':
    app.run(debug=True)