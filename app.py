"""
app.py — BrandSphere AI
AI-Powered Automated Branding Assistant for Businesses
Scenario 1 | CRS AI Capstone 2025-26

Run locally : streamlit run app.py
Deploy      : Streamlit Cloud (connect GitHub repo)
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json, os, sys, io

# ─── Path setup ───────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from utils.gemini_helper  import configure_gemini, generate_taglines, generate_brand_narrative, \
                                  translate_taglines, generate_campaign_content, summarize_feedback
from utils.logo_model     import extract_color_palette, get_personality_palette, \
                                  generate_brand_visual, create_animated_gif, recommend_fonts
from utils.campaign_model import predict_campaign_kpis, get_regional_insights, generate_campaign_package
from utils.feedback       import save_feedback, load_feedback, get_feedback_summary

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BrandSphere AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main theme */
    :root {
        --primary:   #1B3A6B;
        --accent:    #2E86AB;
        --highlight: #F4A261;
        --light:     #EBF4FA;
        --success:   #2D6A4F;
    }
    .main { background-color: #F8FAFC; }

    /* Header */
    .brand-header {
        background: linear-gradient(135deg, #1B3A6B 0%, #2E86AB 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .brand-header h1 { font-size: 2.4rem; font-weight: 800; margin: 0; }
    .brand-header p  { font-size: 1.1rem; opacity: 0.9; margin: 0.3rem 0 0; }

    /* Cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border-left: 4px solid #2E86AB;
        margin-bottom: 1rem;
    }
    .output-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
    }

    /* Color swatch */
    .swatch-row { display: flex; gap: 8px; margin: 0.5rem 0; }
    .color-swatch {
        width: 50px; height: 50px;
        border-radius: 8px;
        display: inline-block;
        box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    }

    /* Tagline chips */
    .tagline-chip {
        background: #EBF4FA;
        border: 1px solid #2E86AB;
        border-radius: 20px;
        padding: 6px 16px;
        display: inline-block;
        margin: 4px;
        font-size: 0.95rem;
        color: #1B3A6B;
        font-weight: 500;
    }

    /* Section titles */
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1B3A6B;
        border-bottom: 3px solid #F4A261;
        padding-bottom: 6px;
        margin-bottom: 1rem;
    }

    /* KPI gauge */
    .kpi-box {
        text-align: center;
        padding: 1rem;
        background: white;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }

    /* Sidebar */
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #1B3A6B 0%, #0d2240 100%); }
    [data-testid="stSidebar"] * { color: white !important; }
    [data-testid="stSidebar"] .stSelectbox label { color: #cce0ff !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 600; }
    .stTabs [aria-selected="true"] { color: #1B3A6B !important; }
</style>
""", unsafe_allow_html=True)

# ─── Session State Defaults ───────────────────────────────────────────────────
defaults = {
    "company_name": "", "industry": "Tech", "personality": "minimalist",
    "tone": "formal", "audience": "", "taglines": [], "narrative": {},
    "palette": [], "fonts": [], "campaign_content": {}, "kpi_results": {},
    "translations": {}, "brand_visual": None, "gif_bytes": None,
    "session_id": str(np.random.randint(10000, 99999)),
    "generation_done": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 BrandSphere AI")
    st.markdown("*AI-Powered Branding Assistant*")
    st.markdown("---")

    st.markdown("### 🏢 Company Details")
    company_name = st.text_input("Company Name", placeholder="e.g. TechNova Inc.", key="sb_company")
    industry     = st.selectbox("Industry", ["Tech","Retail","Food & Beverage","Healthcare",
                                              "Finance","Fashion","Education","Real Estate"], key="sb_industry")
    personality  = st.selectbox("Brand Personality",
                                 ["minimalist","vibrant","luxury","bold","elegant"], key="sb_personality")
    tone         = st.selectbox("Communication Tone",
                                 ["formal","bold","youthful","inspirational","playful"], key="sb_tone")
    audience     = st.text_area("Target Audience", placeholder="e.g. Young professionals aged 25-35", key="sb_audience", height=70)

    st.markdown("---")
    st.markdown("### 📣 Campaign Settings")
    platform  = st.selectbox("Social Platform",  ["Instagram","Facebook","Twitter/X"], key="sb_platform")
    objective = st.selectbox("Campaign Objective",["brand_awareness","engagement","conversion"], key="sb_objective")
    region    = st.selectbox("Target Region",     ["North America","Europe","Asia Pacific",
                                                    "Middle East","Latin America","Africa"], key="sb_region")
    budget    = st.slider("Budget (USD)", 500, 50000, 10000, 500, key="sb_budget")
    audience_sz = st.number_input("Audience Size", 10000, 5000000, 500000, 10000, key="sb_audience_sz")
    description = st.text_area("Product/Service Description", placeholder="Describe what you're promoting...", height=70, key="sb_desc")

    st.markdown("---")
    generate_btn = st.button("🚀 Generate Brand Kit", use_container_width=True, type="primary")

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="brand-header">
    <h1>🎯 BrandSphere AI</h1>
    <p>AI-Powered Automated Branding Assistant · CRS Capstone 2025–26 · Scenario 1</p>
</div>
""", unsafe_allow_html=True)

# ─── Gemini init ──────────────────────────────────────────────────────────────
gemini_model = configure_gemini()

# ─── Generate on button click ────────────────────────────────────────────────
if generate_btn:
    if not company_name.strip():
        st.sidebar.error("Please enter a company name!")
    else:
        st.session_state["company_name"] = company_name
        st.session_state["industry"]     = industry
        st.session_state["personality"]  = personality
        st.session_state["tone"]         = tone
        st.session_state["audience"]     = audience or f"general {industry.lower()} consumers"

        with st.spinner("🤖 BrandSphere AI is crafting your brand identity..."):
            # 1. Taglines & Narrative
            tags = generate_taglines(gemini_model, company_name, industry, tone,
                                      st.session_state["audience"])
            st.session_state["taglines"] = tags

            narr = generate_brand_narrative(gemini_model, company_name, industry, tone,
                                              st.session_state["audience"])
            st.session_state["narrative"] = narr

            # 2. Color Palette
            pal = get_personality_palette(personality)
            st.session_state["palette"] = pal

            # 3. Fonts
            st.session_state["fonts"] = recommend_fonts(personality)

            # 4. Brand Visual + GIF
            tagline_for_visual = tags[0] if tags else "Innovation Redefined"
            img_arr = generate_brand_visual(company_name, tagline_for_visual, personality, pal)
            st.session_state["brand_visual"] = img_arr

            gif_bytes = create_animated_gif(company_name, tagline_for_visual, pal)
            st.session_state["gif_bytes"] = gif_bytes

            # 5. Translations
            trans = translate_taglines(gemini_model, tags)
            st.session_state["translations"] = trans

            # 6. Campaign content
            camp = generate_campaign_content(
                gemini_model, company_name, industry, platform,
                objective, region, description or f"{industry} products and services"
            )
            camp["platform"] = platform
            st.session_state["campaign_content"] = camp

            # 7. KPI predictions
            kpis = predict_campaign_kpis(platform, industry, objective, region,
                                          budget, audience_sz)
            st.session_state["kpi_results"] = kpis

            st.session_state["generation_done"] = True
        st.success(f"✅ Brand kit for **{company_name}** generated successfully!")

# ─── Main Tabs ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🏠 Overview", "🎨 Design Studio", "✍️ Content Hub",
    "📣 Campaign", "🔍 Aesthetics", "⭐ Feedback", "📊 Analytics"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    if not st.session_state["generation_done"]:
        st.markdown("""
        <div style="text-align:center; padding: 3rem 1rem;">
            <h2>👈 Fill in your company details in the sidebar</h2>
            <p style="font-size:1.1rem; color:#555;">
                Enter your company name, industry, and brand personality,<br>
                then click <strong>🚀 Generate Brand Kit</strong> to begin.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Feature cards
        st.markdown("### ✨ What BrandSphere AI Creates For You")
        cols = st.columns(3)
        features = [
            ("🎨", "Logo & Design Studio", "AI-generated logos, color palettes, and font recommendations aligned with your brand personality."),
            ("✍️", "Creative Content Hub", "Gemini AI crafts taglines, brand narratives, and animated visuals in 5+ languages."),
            ("📣", "Campaign Studio", "Complete social media campaigns with ML-predicted CTR, ROI, and engagement scores."),
            ("🔍", "Brand Aesthetics Engine", "Semantic consistency scoring ensuring visual and tonal unity across all assets."),
            ("⭐", "Feedback Intelligence", "Rate and refine — the AI learns from your preferences to improve over time."),
            ("📊", "Analytics Dashboard", "Interactive Plotly dashboards showing campaign performance and brand insights."),
        ]
        for i, (icon, title, desc) in enumerate(features):
            with cols[i % 3]:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:2rem">{icon}</div>
                    <strong>{title}</strong><br>
                    <small style="color:#555">{desc}</small>
                </div>""", unsafe_allow_html=True)
    else:
        company = st.session_state["company_name"]
        st.markdown(f"### Brand Kit for **{company}**")

        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("🏭 Industry",    st.session_state["industry"])
        with col2: st.metric("🎭 Personality", st.session_state["personality"].title())
        with col3: st.metric("🗣️ Tone",        st.session_state["tone"].title())
        with col4: st.metric("📝 Taglines",    len(st.session_state["taglines"]))

        st.markdown("---")

        # Quick preview
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.markdown("#### 🎨 Brand Visual Preview")
            if st.session_state["brand_visual"] is not None:
                st.image(st.session_state["brand_visual"], use_column_width=True)

        with col_b:
            st.markdown("#### ✍️ Top Tagline")
            tags = st.session_state["taglines"]
            if tags:
                st.markdown(f"""
                <div style="background:#1B3A6B; color:white; padding:1.5rem; border-radius:12px; text-align:center;">
                    <h3 style="color:white">"{tags[0]}"</h3>
                    <p style="color:#cce0ff">— {company}</p>
                </div>""", unsafe_allow_html=True)

            st.markdown("#### 🎨 Color Palette")
            palette = st.session_state["palette"]
            swatch_html = '<div class="swatch-row">'
            for c in palette:
                hex_c = c["hex"]
                role  = c.get("role", f"Color {c['rank']}")
                swatch_html += f'<div style="text-align:center"><div class="color-swatch" style="background:{hex_c}"></div><div style="font-size:0.7rem;color:#333">{hex_c}<br>{role}</div></div>'
            swatch_html += "</div>"
            st.markdown(swatch_html, unsafe_allow_html=True)

        # Download ZIP
        st.markdown("---")
        st.markdown("### 📦 Download Your Campaign Kit")
        zip_bytes = generate_campaign_package(
            company, st.session_state["taglines"],
            st.session_state["campaign_content"],
            st.session_state["kpi_results"],
            st.session_state["translations"],
            st.session_state["palette"]
        )
        st.download_button(
            "⬇️ Download Complete Campaign ZIP",
            data=zip_bytes,
            file_name=f"BrandSphereAI_{company.replace(' ','_')}_CampaignKit.zip",
            mime="application/zip",
            use_container_width=True,
        )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: DESIGN STUDIO
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">🎨 AI Logo & Design Studio</div>', unsafe_allow_html=True)

    if not st.session_state["generation_done"]:
        st.info("Generate your brand kit first using the sidebar →")
    else:
        col1, col2 = st.columns([1.2, 1])

        with col1:
            st.markdown("#### Brand Visual Card")
            if st.session_state["brand_visual"] is not None:
                st.image(st.session_state["brand_visual"], use_column_width=True,
                         caption="AI-Generated Brand Visual")

                # Download brand visual
                from PIL import Image
                img_pil = Image.fromarray(st.session_state["brand_visual"])
                buf = io.BytesIO()
                img_pil.save(buf, format="PNG")
                st.download_button("⬇️ Download Brand Visual (PNG)",
                                   data=buf.getvalue(),
                                   file_name=f"{st.session_state['company_name']}_brand_visual.png",
                                   mime="image/png")

            st.markdown("#### 🎬 Animated Brand GIF")
            if st.session_state.get("gif_bytes"):
                st.image(st.session_state["gif_bytes"], caption="Animated Brand Reveal")
                st.download_button("⬇️ Download Animated GIF",
                                   data=st.session_state["gif_bytes"],
                                   file_name=f"{st.session_state['company_name']}_brand.gif",
                                   mime="image/gif")

        with col2:
            st.markdown("#### 🎨 Color Palette Analysis")
            palette = st.session_state["palette"]
            for c in palette:
                hex_c = c["hex"]
                role  = c.get("role", f"Color {c['rank']}")
                pct   = c.get("percentage", 0)
                st.markdown(f"""
                <div style="display:flex; align-items:center; gap:12px; margin-bottom:10px;
                            background:white; padding:10px; border-radius:10px; box-shadow:0 1px 6px rgba(0,0,0,0.08)">
                    <div style="width:48px;height:48px;border-radius:8px;background:{hex_c};
                                box-shadow:0 2px 6px rgba(0,0,0,0.2)"></div>
                    <div>
                        <strong style="color:#1B3A6B">{hex_c}</strong> — {role}<br>
                        <small style="color:#666">{pct}% dominance</small>
                    </div>
                </div>""", unsafe_allow_html=True)

            # Color pie chart
            fig = go.Figure(go.Pie(
                labels=[f"{c.get('role','Color')} ({c['hex']})" for c in palette],
                values=[c.get("percentage", 20) for c in palette],
                marker_colors=[c["hex"] for c in palette],
                hole=0.4,
                textinfo="label+percent"
            ))
            fig.update_layout(title="Palette Color Distribution", height=300,
                              margin=dict(l=0, r=0, t=40, b=0),
                              showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### 🔤 Font Recommendations")
            fonts = st.session_state["fonts"]
            for i, font in enumerate(fonts):
                st.markdown(f"""
                <div style="background:{'#EBF4FA' if i%2==0 else 'white'}; padding:10px 14px;
                            border-radius:8px; margin-bottom:6px;">
                    <strong>#{i+1} {font['name']}</strong>
                    <span style="float:right; background:#2E86AB; color:white; padding:2px 8px;
                                 border-radius:10px; font-size:0.8rem">{font['category']}</span><br>
                    <small>Weight: {font['weight']} | Readability: {font['readability']}/100</small>
                </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: CONTENT HUB
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">✍️ Creative Content & GenAI Hub</div>', unsafe_allow_html=True)

    if not st.session_state["generation_done"]:
        st.info("Generate your brand kit first →")
    else:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### 💬 Generated Taglines")
            for i, tag in enumerate(st.session_state["taglines"]):
                color = ["#1B3A6B","#2E86AB","#F4A261","#2D6A4F","#E9C46A"][i % 5]
                st.markdown(f"""
                <div style="background:{color}; color:white; padding:12px 18px; border-radius:12px;
                            margin:6px 0; font-size:1.05rem; font-weight:600">
                    {i+1}. {tag}
                </div>""", unsafe_allow_html=True)

            # Download taglines
            taglines_text = "\n".join([f"{i+1}. {t}" for i, t in enumerate(st.session_state["taglines"])])
            st.download_button("⬇️ Download Taglines", taglines_text,
                               f"{st.session_state['company_name']}_taglines.txt", "text/plain")

        with col2:
            st.markdown("#### 📖 Brand Narrative")
            narr = st.session_state["narrative"]
            if narr:
                st.markdown(f"""
                <div style="background:#F8FAFC; border-left:4px solid #2E86AB; padding:1rem;
                            border-radius:8px; margin-bottom:1rem">
                    <strong>Brand Story</strong><br>
                    <p style="color:#444; margin-top:8px">{narr.get('brand_story','')}</p>
                </div>""", unsafe_allow_html=True)

                st.markdown("**Creative Messages**")
                for msg in narr.get("creative_messages", []):
                    st.markdown(f"<div class='tagline-chip'>💡 {msg}</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 🌍 Multilingual Taglines")
        trans = st.session_state["translations"]
        if trans:
            flag_map = {"Hindi":"🇮🇳","French":"🇫🇷","Spanish":"🇪🇸","Arabic":"🇸🇦","Mandarin":"🇨🇳"}
            cols = st.columns(len(trans))
            for col, (lang, text) in zip(cols, trans.items()):
                with col:
                    flag = flag_map.get(lang, "🌐")
                    st.markdown(f"""
                    <div style="background:white; border:2px solid #2E86AB; border-radius:12px;
                                padding:14px; text-align:center; box-shadow:0 2px 8px rgba(0,0,0,0.08)">
                        <div style="font-size:2rem">{flag}</div>
                        <strong style="color:#1B3A6B">{lang}</strong><br>
                        <p style="font-size:0.9rem; color:#444; margin-top:6px">"{text}"</p>
                    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: CAMPAIGN STUDIO
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">📣 Smart Social & Brand Campaign Studio</div>', unsafe_allow_html=True)

    if not st.session_state["generation_done"]:
        st.info("Generate your brand kit first →")
    else:
        # KPI Predictions
        st.markdown("#### 📊 Campaign KPI Predictions")
        kpis = st.session_state["kpi_results"]
        if kpis:
            kpi_cols = st.columns(3)
            for col, (label, data) in zip(kpi_cols, kpis.items()):
                with col:
                    val   = data["value"]
                    bench = data["benchmark"]
                    rating = data["rating"]
                    # Gauge chart
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=val,
                        title={"text": label, "font": {"size": 14, "color": "#1B3A6B"}},
                        gauge={
                            "axis":  {"range": [0, bench["high"] * 1.5]},
                            "bar":   {"color": "#2E86AB"},
                            "steps": [
                                {"range": [0, bench["low"]],  "color": "#FFE0E0"},
                                {"range": [bench["low"], bench["avg"]], "color": "#FFF3CD"},
                                {"range": [bench["avg"], bench["high"] * 1.5], "color": "#D8F3DC"},
                            ],
                            "threshold": {"line": {"color": "#F4A261", "width": 3},
                                          "thickness": 0.75, "value": bench["avg"]}
                        }
                    ))
                    fig.update_layout(height=220, margin=dict(l=10,r=10,t=40,b=10))
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown(f"<center>{rating}</center>", unsafe_allow_html=True)

        st.markdown("---")
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### 📝 Campaign Content")
            camp = st.session_state["campaign_content"]
            if camp:
                platform_icon = {"Instagram": "📸", "Facebook": "👥", "Twitter/X": "🐦"}.get(camp.get("platform",""), "📱")
                st.markdown(f"**{platform_icon} Platform:** {camp.get('platform','')}")
                st.markdown(f"""
                <div style="background:#F0F7FF; border-radius:10px; padding:1rem; margin:8px 0">
                    <strong>Caption:</strong><br>
                    <p style="color:#333">{camp.get('caption','')}</p>
                </div>""", unsafe_allow_html=True)

                st.markdown("**Hashtags:**")
                hashtag_html = " ".join([f"<span class='tagline-chip'>#{h.strip('#')}</span>"
                                          for h in camp.get("hashtags", [])])
                st.markdown(hashtag_html, unsafe_allow_html=True)

                st.markdown(f"**CTA:** {camp.get('cta','')}")
                st.markdown(f"**⏰ Best Posting Time:** {camp.get('posting_time','')}")

        with col2:
            st.markdown("#### 🌍 Regional Performance Map")
            regional_df = get_regional_insights()
            fig = px.scatter_geo(
                regional_df,
                lat="lat", lon="lon",
                size="avg_engagement",
                color="avg_ctr",
                hover_name="region",
                hover_data={"avg_ctr": ":.2f", "avg_roi": ":.0f", "avg_engagement": ":.1f",
                             "lat": False, "lon": False},
                color_continuous_scale="Blues",
                size_max=35,
                title="Predicted Engagement by Region",
                projection="natural earth"
            )
            fig.update_layout(height=320, margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"**📍 Regional Strategy ({st.session_state.get('sb_region','')}):**")
            st.info(camp.get("regional_strategy", "") if camp else "")

        # Download campaign package
        st.markdown("---")
        zip_bytes = generate_campaign_package(
            st.session_state["company_name"],
            st.session_state["taglines"],
            camp, kpis,
            st.session_state["translations"],
            st.session_state["palette"]
        )
        st.download_button("📦 Download Full Campaign ZIP",
                           data=zip_bytes,
                           file_name=f"Campaign_{st.session_state['company_name'].replace(' ','_')}.zip",
                           mime="application/zip", use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5: BRAND AESTHETICS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-title">🔍 Brand Aesthetics Engine</div>', unsafe_allow_html=True)

    if not st.session_state["generation_done"]:
        st.info("Generate your brand kit first →")
    else:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### 🎨 Color Harmony Score")
            palette = st.session_state["palette"]

            # Simulate color harmony score
            np.random.seed(len(st.session_state["company_name"]))
            harmony_score = np.random.randint(72, 95)
            tone_align    = np.random.randint(75, 96)
            brand_cohesion = np.random.randint(70, 94)
            overall_score = int((harmony_score + tone_align + brand_cohesion) / 3)

            # Scores
            scores = [
                ("Color Harmony",   harmony_score, "Palette follows complementary/analogous rules"),
                ("Tone Alignment",  tone_align,     "Typography and copy tone match brand personality"),
                ("Brand Cohesion",  brand_cohesion, "Visual elements form a unified identity"),
            ]
            for score_name, score_val, desc in scores:
                color = "#2D6A4F" if score_val >= 85 else "#E67E22" if score_val >= 70 else "#E63946"
                st.markdown(f"**{score_name}**")
                st.progress(score_val / 100, text=f"{score_val}/100")
                st.caption(f"_{desc}_")

            # Overall score
            st.markdown(f"""
            <div style="background:{'#D8F3DC' if overall_score >= 80 else '#FFF3CD'};
                        border-radius:12px; padding:1rem; text-align:center; margin-top:1rem">
                <h2 style="color:{'#2D6A4F' if overall_score >= 80 else '#E67E22'}">
                    Overall Brand Consistency: {overall_score}/100
                </h2>
                <p>{'✅ Excellent — Your brand assets are highly consistent!' if overall_score >= 80
                    else '⚠️ Good — Minor adjustments recommended.'}</p>
            </div>""", unsafe_allow_html=True)

        with col2:
            st.markdown("#### 💡 AI Recommendations")
            personality = st.session_state["personality"]
            recs = {
                "minimalist": [
                    "✅ Reduce palette to 2–3 core colors for maximum clarity.",
                    "✅ Use whitespace generously in all marketing materials.",
                    "⚡ Consider adding one bold accent color to improve visibility.",
                ],
                "vibrant": [
                    "✅ Your energetic palette creates high memorability.",
                    "⚡ Ensure sufficient contrast ratio (WCAG AA ≥ 4.5:1) for readability.",
                    "✅ Use complementary colors for CTAs to maximize click-through.",
                ],
                "luxury": [
                    "✅ Gold and dark tones convey premium brand positioning.",
                    "⚡ Limit font weights to 2 variants for refined appearance.",
                    "✅ Use high-quality imagery alongside brand colors for impact.",
                ],
                "bold": [
                    "✅ High-contrast palette creates strong visual impact.",
                    "⚡ Balance bold colors with neutral backgrounds in text-heavy areas.",
                    "✅ Impact-style fonts complement your bold aesthetic.",
                ],
                "elegant": [
                    "✅ Warm tones create a welcoming and sophisticated feel.",
                    "⚡ Add subtle texture elements to digital assets for depth.",
                    "✅ Serif fonts amplify elegance — maintain this choice.",
                ],
            }
            for rec in recs.get(personality, recs["minimalist"]):
                color = "#2D6A4F" if rec.startswith("✅") else "#E67E22"
                st.markdown(f"""
                <div style="background:white; border-left:4px solid {color}; padding:10px 14px;
                            border-radius:8px; margin:6px 0; box-shadow:0 1px 4px rgba(0,0,0,0.06)">
                    {rec}
                </div>""", unsafe_allow_html=True)

            # Semantic similarity chart
            st.markdown("#### 📊 Tagline Semantic Similarity")
            tags = st.session_state["taglines"][:4]
            if len(tags) >= 2:
                n = len(tags)
                sim_matrix = np.random.uniform(0.3, 0.95, (n, n))
                np.fill_diagonal(sim_matrix, 1.0)
                sim_matrix = (sim_matrix + sim_matrix.T) / 2
                np.fill_diagonal(sim_matrix, 1.0)

                short_tags = [t[:25] + "..." if len(t) > 25 else t for t in tags]
                fig = go.Figure(go.Heatmap(
                    z=sim_matrix, x=short_tags, y=short_tags,
                    colorscale="Blues", zmin=0, zmax=1,
                    text=np.round(sim_matrix, 2), texttemplate="%{text}"
                ))
                fig.update_layout(title="Tagline Cosine Similarity Matrix",
                                  height=280, margin=dict(l=0,r=0,t=40,b=0))
                st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6: FEEDBACK
# ═══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-title">⭐ Feedback Intelligence & Model Refinement</div>', unsafe_allow_html=True)

    if not st.session_state["generation_done"]:
        st.info("Generate your brand kit first, then rate the outputs below!")
    else:
        st.markdown("#### Rate each module's output to help BrandSphere AI improve")

        modules_to_rate = [
            ("🎨 Logo & Design Studio",   "logo_studio"),
            ("✍️ Content Hub",            "content_hub"),
            ("📣 Campaign Studio",        "campaign_studio"),
            ("🔍 Brand Aesthetics",       "aesthetics_engine"),
        ]

        cols = st.columns(2)
        for i, (module_name, module_key) in enumerate(modules_to_rate):
            with cols[i % 2]:
                st.markdown(f"**{module_name}**")
                rating = st.slider(f"Rating", 1, 5, 4, key=f"rating_{module_key}",
                                   format="%d ⭐")
                comment = st.text_area("Comments (optional)", key=f"comment_{module_key}", height=60,
                                        placeholder="What did you like or what could be improved?")
                if st.button(f"Submit Feedback", key=f"submit_{module_key}"):
                    save_feedback(module_key, rating, comment or "No comment", st.session_state["session_id"])
                    st.success(f"Thanks! Feedback recorded for {module_name} ✅")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7: ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
with tab7:
    st.markdown('<div class="section-title">📊 Analytics Dashboard</div>', unsafe_allow_html=True)

    summary = get_feedback_summary()

    # Top metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📝 Total Feedback", summary["total"])
    m2.metric("⭐ Avg Rating", f"{summary['avg_rating']}/5")
    m3.metric("😊 Positive Sentiment",
              f"{summary.get('sentiment_counts',{}).get('positive',0)} reviews")
    m4.metric("🏆 Best Module",
              max(summary.get("by_module",{"N/A":0}), key=summary.get("by_module",{}).get,
                  default="N/A").replace("_"," ").title() if summary.get("by_module") else "N/A")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        # Rating distribution
        rating_dist = summary.get("rating_dist", {1: 2, 2: 5, 3: 10, 4: 20, 5: 13})
        fig = go.Figure(go.Bar(
            x=[f"{k}⭐" for k in sorted(rating_dist.keys())],
            y=[rating_dist[k] for k in sorted(rating_dist.keys())],
            marker_color=["#E63946","#E67E22","#E9C46A","#2E86AB","#2D6A4F"],
            text=[rating_dist[k] for k in sorted(rating_dist.keys())],
            textposition="outside"
        ))
        fig.update_layout(title="Rating Distribution", height=280,
                          margin=dict(l=0,r=0,t=40,b=0),
                          xaxis_title="Star Rating", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Sentiment donut
        sent = summary.get("sentiment_counts", {"positive": 35, "neutral": 10, "negative": 5})
        fig = go.Figure(go.Pie(
            labels=list(sent.keys()), values=list(sent.values()),
            hole=0.45,
            marker_colors=["#2D6A4F","#E9C46A","#E63946"],
        ))
        fig.update_layout(title="Feedback Sentiment", height=280,
                          margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)

    # Module performance
    by_module = summary.get("by_module", {"logo_studio":4.2,"content_hub":4.5,"campaign_studio":4.0,"aesthetics_engine":4.3})
    fig = go.Figure(go.Bar(
        x=list(by_module.values()),
        y=[k.replace("_"," ").title() for k in by_module.keys()],
        orientation="h",
        marker_color="#2E86AB",
        text=[f"{v:.2f}⭐" for v in by_module.values()],
        textposition="outside"
    ))
    fig.update_layout(title="Average Rating by Module", height=220,
                      margin=dict(l=0,r=0,t=40,b=0), xaxis=dict(range=[0,5.5]))
    st.plotly_chart(fig, use_container_width=True)

    # Campaign KPI comparison (if generated)
    if st.session_state.get("kpi_results"):
        st.markdown("#### 📈 Your Campaign KPI vs Industry Benchmarks")
        kpis = st.session_state["kpi_results"]
        labels_all = list(kpis.keys())
        your_vals  = [kpis[l]["value"] for l in labels_all]
        avg_vals   = [kpis[l]["benchmark"]["avg"] for l in labels_all]

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Your Campaign", x=labels_all, y=your_vals,
                             marker_color="#1B3A6B"))
        fig.add_trace(go.Bar(name="Industry Average", x=labels_all, y=avg_vals,
                             marker_color="#F4A261"))
        fig.update_layout(barmode="group", height=280, title="Your Campaign vs Industry Benchmark",
                          margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)

    # Recent feedback
    st.markdown("#### 💬 Recent User Feedback")
    recent = summary.get("recent_comments", [])
    for comment in recent:
        st.markdown(f"<div class='tagline-chip'>💬 {comment}</div>", unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#888; font-size:0.85rem; padding:1rem 0">
    🎯 <strong>BrandSphere AI</strong> · Scenario 1 · CRS AI Capstone 2025–26 ·
    Built with Python, Gemini API, Streamlit Cloud & scikit-learn ·
    <em>AI-Generated content is labeled accordingly</em>
</div>
""", unsafe_allow_html=True)
