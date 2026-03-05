"""
utils/gemini_helper.py
BrandSphere AI — Gemini API Integration Helper
Handles all Gemini API calls with caching, error handling, and prompt templates.
"""

import google.generativeai as genai
import streamlit as st
import json, re, time, os

# ─── Configure API ────────────────────────────────────────────────────────────
def configure_gemini():
    """Configure Gemini API from Streamlit secrets or environment."""
    try:
        api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            st.error("⚠️ Gemini API key not found. Add GEMINI_API_KEY to Streamlit secrets.")
            return None
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-1.5-flash")
    except Exception as e:
        st.warning(f"Gemini configuration error: {e}")
        return None


def safe_generate(model, prompt: str, fallback: str = "AI content unavailable") -> str:
    """Call Gemini with retry logic and error handling."""
    if model is None:
        return fallback
    for attempt in range(3):
        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return fallback
    return fallback


# ─── Tagline Generation ───────────────────────────────────────────────────────
def generate_taglines(model, company_name: str, industry: str, tone: str, target_audience: str) -> list[str]:
    """Generate 5 brand taglines using Gemini API."""
    prompt = f"""You are a senior brand copywriter. Generate exactly 5 unique, memorable brand taglines for:

Company: {company_name}
Industry: {industry}
Brand Tone: {tone}
Target Audience: {target_audience}

Requirements:
- Each tagline should be 3–8 words
- Reflect the brand tone: {tone}
- Be original and avoid clichés
- Output ONLY a JSON array of 5 strings, nothing else

Example format: ["Tagline 1", "Tagline 2", "Tagline 3", "Tagline 4", "Tagline 5"]"""

    result = safe_generate(model, prompt)
    try:
        # Extract JSON array from response
        match = re.search(r'\[.*?\]', result, re.DOTALL)
        if match:
            taglines = json.loads(match.group())
            return [str(t) for t in taglines[:5]]
    except Exception:
        pass
    # Fallback: split by newline
    lines = [l.strip().strip('"').strip("'") for l in result.split('\n') if l.strip()]
    return lines[:5] if lines else [f"The Future of {industry}", f"{company_name}: Excellence Redefined"]


# ─── Brand Narrative ─────────────────────────────────────────────────────────
def generate_brand_narrative(model, company_name: str, industry: str, tone: str, audience: str) -> dict:
    """Generate brand story and 3 creative messages."""
    prompt = f"""Act as a world-class brand strategist for {company_name}, a {industry} brand.
Tone: {tone} | Audience: {audience}

Output ONLY valid JSON with this exact structure:
{{
  "brand_story": "150-word brand origin and vision story",
  "creative_messages": ["Message 1 (20 words)", "Message 2 (20 words)", "Message 3 (20 words)"]
}}"""

    result = safe_generate(model, prompt)
    try:
        match = re.search(r'\{.*\}', result, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return {
        "brand_story": f"{company_name} was founded with a vision to revolutionize the {industry} industry. We believe in excellence, innovation, and creating lasting impact for our customers.",
        "creative_messages": [
            f"Empowering your {industry.lower()} journey with {company_name}.",
            f"Where innovation meets {tone.lower()} excellence.",
            f"Built for {audience} who demand the best."
        ]
    }


# ─── Multilingual Translation ─────────────────────────────────────────────────
def translate_taglines(model, taglines: list[str], languages: list[str] = None) -> dict:
    """Translate taglines into multiple languages."""
    if languages is None:
        languages = ["Hindi", "French", "Spanish", "Arabic", "Mandarin"]

    primary_tagline = taglines[0] if taglines else "Innovation at its finest"
    prompt = f"""Translate this brand tagline into the following languages.
Tagline: "{primary_tagline}"
Languages: {', '.join(languages)}

Output ONLY valid JSON: {{"language": "translation"}} for each language.
Keep translations short, impactful, and culturally appropriate."""

    result = safe_generate(model, prompt)
    translations = {}
    try:
        match = re.search(r'\{.*\}', result, re.DOTALL)
        if match:
            translations = json.loads(match.group())
    except Exception:
        pass

    # Ensure all requested languages have entries
    fallbacks = {
        "Hindi": f"{primary_tagline} (Hindi)",
        "French": f"{primary_tagline} (French)",
        "Spanish": f"{primary_tagline} (Spanish)",
        "Arabic": f"{primary_tagline} (Arabic)",
        "Mandarin": f"{primary_tagline} (Mandarin)",
    }
    for lang in languages:
        if lang not in translations:
            translations[lang] = fallbacks.get(lang, primary_tagline)
    return translations


# ─── Social Media Campaign Generator ─────────────────────────────────────────
def generate_campaign_content(model, company: str, industry: str, platform: str,
                               objective: str, region: str, description: str) -> dict:
    """Generate complete social media campaign content."""
    char_limits = {"Instagram": 2200, "Facebook": 63206, "Twitter/X": 280}
    char_limit = char_limits.get(platform, 280)

    prompt = f"""You are a social media marketing expert. Create a campaign for:
Company: {company} | Industry: {industry}
Platform: {platform} | Objective: {objective}
Target Region: {region}
Product/Service: {description}

Output ONLY valid JSON:
{{
  "caption": "engaging post caption under {char_limit} chars",
  "hashtags": ["hashtag1", "hashtag2", ..., "hashtag10"],
  "cta": "strong call-to-action phrase",
  "regional_strategy": "2-sentence strategy for {region}",
  "posting_time": "best day and time to post"
}}"""

    result = safe_generate(model, prompt)
    try:
        match = re.search(r'\{.*\}', result, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass

    return {
        "caption": f"Discover {company}'s latest innovation in {industry}! 🚀 Transform your experience today.",
        "hashtags": [f"#{company.replace(' ','')}", f"#{industry.replace(' ','')}", "#Innovation", "#BrandNew", "#Success"],
        "cta": "Shop Now — Limited Time Offer!",
        "regional_strategy": f"Focus on {region}'s growing market with localized content.",
        "posting_time": "Tuesday–Thursday, 6–9 PM local time"
    }


# ─── Feedback Summarizer ──────────────────────────────────────────────────────
def summarize_feedback(model, feedback_list: list[str]) -> str:
    """Summarize user feedback into actionable insights."""
    if not feedback_list:
        return "No feedback available yet."
    combined = '\n'.join([f"- {f}" for f in feedback_list[:20]])
    prompt = f"""Analyze these user feedback comments for an AI branding tool and provide 3 specific, actionable improvements:

{combined}

Output 3 numbered improvement suggestions, each 1–2 sentences. Be specific and constructive."""
    return safe_generate(model, prompt, fallback="Feedback analysis unavailable.")
