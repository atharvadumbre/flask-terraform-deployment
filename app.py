from flask import Flask, request, jsonify
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.kl import KLSummarizer
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

PROMPT = """
# Women-Specific Incident Tagging Prompt for Discos/Nightclubs

You are an AI assistant tasked with analyzing reviews of discos and nightclubs to identify incidents that particularly affect women. Your job is to carefully read the given review and determine if it mentions or implies any of the following categories of incidents:

1. Racism: Discriminatory behavior, comments, or policies based on race or ethnicity, particularly targeting women of color.
2. Assault: Physical attacks or unwanted aggressive physical contact towards women, including pushing, grabbing, or any form of violence.
3. Harassment: Persistent unwanted behavior towards women, including verbal abuse, intimidation, catcalling, or persistent unwanted attention.
4. Sexual Misconduct: Any non-consensual sexual behavior targeting women, including groping, inappropriate touching, sexual comments, drink spiking, or pressure to engage in sexual activities.
5. Drugs: Presence, use, or distribution of illegal substances, particularly instances of pressuring women to use drugs or using drugs to take advantage of women.

Instructions:
1. Thoroughly read the provided disco/nightclub review.
2. For each category, determine if the review mentions or implies any relevant incidents specifically affecting women.
3. Include in your response only the categories that are present in the review.
4. Your response should be a comma-separated list of the relevant categories found.
5. If you don't find anything try to give tags which are nearer to the given tags.

Important notes:
- Be attentive to both explicit mentions and subtle implications of these incidents.
- Consider the specific challenges women might face in a disco/nightclub environment.
- Look for incidents involving staff (bouncers, bartenders, DJs) as well as other patrons.
- Be aware that some incidents might fall into multiple categories.
- Pay attention to power dynamics and situations where women might feel unsafe or discriminated against.
- Consider both individual incidents and broader patterns of behavior or policies that might be problematic for women.
- Maintain objectivity while being sensitive to the varied experiences of women in these settings.

Disco/Nightclub review to analyze:
{review}
"""
def analyze_review(review_text):
    try:
        GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
        genai.configure(api_key=GOOGLE_API_KEY)

        model = genai.GenerativeModel('gemini-pro')
        full_prompt = PROMPT.format(review=review_text)
        response = model.generate_content(full_prompt)

        if hasattr(response, 'text'):
            content = response.text.strip()
            if content:
                return content
            else:
                return "Error: Empty response from the model"
        elif hasattr(response, 'parts') and response.parts:
            content = ''.join(part.text for part in response.parts).strip()
            if content:
                return content
            else:
                return "Error: Empty response from the model parts"
        else:
            print(f"Debug - Full response: {response}")
            return "Error: Unexpected or empty response format"

    except AttributeError as e:
        return f"AttributeError: {str(e)}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

@app.route('/summarize', methods=['POST'])
def summarize_text():
    data = request.get_json()
    original_text = data.get('text', '')
    
    if not original_text:
        return jsonify({"error": "Text input is required"}), 400
    
    parser = PlaintextParser.from_string(original_text, Tokenizer('english'))
    kl_summarizer = KLSummarizer()
    kl_summary = kl_summarizer(parser.document, sentences_count=3)
    summary = ' '.join([str(sentence) for sentence in kl_summary])
    return jsonify({"summary": summary})


@app.route('/tag', methods=['POST'])
def analyze_review_route():
    data = request.get_json()
    review_text = data.get('text', '')

    if not review_text:
        return jsonify({"error": "Review text is required"}), 400

    result = analyze_review(review_text)
    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(debug=True)
