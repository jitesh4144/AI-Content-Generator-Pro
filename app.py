import os
import json
import requests
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
import datetime
import pandas as pd
from io import BytesIO
import base64


# Load .env from the script directory to ensure Streamlit finds it regardless of CWD
here = Path(__file__).resolve().parent
env_path = here / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=str(env_path))
else:
    # Fallback to default load (searches CWD and parents)
    load_dotenv()

# Configuration via .env (read after load_dotenv)
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "gemini").lower()  # 'gemini' or 'openai'
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")

# Normalize keys: strip surrounding quotes if the .env contains them
if GEMINI_API_KEY:
    GEMINI_API_KEY = GEMINI_API_KEY.strip().strip('"').strip("'")
if OPENAI_API_KEY:
    OPENAI_API_KEY = OPENAI_API_KEY.strip().strip('"').strip("'")


def generate_with_gemini(prompt: str, model: str = GEMINI_MODEL, api_key: str = None, max_tokens: int = 256, temperature: float = 0.7) -> str:
    """Generate text using Google Generative AI REST endpoint using API key."""
    if api_key is None:
        raise RuntimeError("GEMINI_API_KEY not set")

    # Use the correct endpoint format for Gemini Pro
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    
    headers = {
        'Content-Type': 'application/json',
    }
    
    params = {
        'key': api_key
    }
    
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }
    }

    try:
        resp = requests.post(url, json=body, headers=headers, params=params, timeout=30)
        
        if not resp.ok:
            try:
                err_body = resp.json()
            except Exception:
                err_body = resp.text
            raise RuntimeError(f"Gemini API error ({resp.status_code}): {err_body}")

        data = resp.json()
        candidates = data.get("candidates", [])
        if candidates and len(candidates) > 0:
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if parts and len(parts) > 0:
                return parts[0].get("text", "").strip()
        
        return "No response generated"
        
    except Exception as e:
        raise RuntimeError(f"Gemini generation failed: {e}")


def generate_with_openai(prompt: str, model: str = OPENAI_MODEL, api_key: str = None, max_tokens: int = 256, temperature: float = 0.7) -> str:
    """Minimal OpenAI completion fallback. Uses the openai package if available.
    If openai is not installed, raises ImportError.
    """
    try:
        import openai
    except Exception as e:
        raise RuntimeError("openai package not installed or unavailable: " + str(e))

    if api_key:
        openai.api_key = api_key

    resp = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


def generate_text(prompt: str, provider: str = MODEL_PROVIDER, **kwargs) -> str:
    provider = provider.lower()
    if provider == "gemini":
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY not found in environment. Set it in .env.")
        return generate_with_gemini(prompt, model=GEMINI_MODEL, api_key=GEMINI_API_KEY, **kwargs)
    elif provider == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not found in environment. Set it in .env.")
        return generate_with_openai(prompt, model=OPENAI_MODEL, api_key=OPENAI_API_KEY, **kwargs)
    else:
        raise ValueError(f"Unknown model provider: {provider}")


def build_prompt(keyword: str, content_type: str, language: str, tone: str, audience: str, rhyme_scheme: str, length_words: int) -> str:
    """Constructs a comprehensive prompt for the model to produce various types of content."""
    
    # Content type specific instructions
    content_instructions = {
        "Quote": "Write a memorable, inspiring quote",
        "Poem": f"Create a {length_words}-word poem (3-8 lines preferred)",
        "Haiku": "Write a traditional 3-line Haiku (5-7-5 syllable pattern)",
        "Motivational Saying": "Create an uplifting motivational saying",
        "Social Media Caption": "Write a catchy social media caption (under 280 characters)",
        "Song Lyrics": "Write song lyrics with rhythm and flow",
        "Story Beginning": "Write an engaging story opening paragraph"
    }
    
    # Language instructions
    language_map = {
        "English": "",
        "Hindi": "Write in Hindi language",
        "Marathi": "Write in Marathi language",
        "Spanish": "Write in Spanish language", 
        "French": "Write in French language",
        "German": "Write in German language"
    }
    
    # Tone instructions
    tone_map = {
        "Funny": "Make it humorous and witty",
        "Serious": "Keep it thoughtful and profound",
        "Romantic": "Make it romantic and heartfelt",
        "Professional": "Keep it professional and polished",
        "Inspirational": "Make it uplifting and motivating"
    }
    
    # Audience instructions
    audience_map = {
        "Kids": "Use simple, fun language suitable for children",
        "Adults": "Use mature, sophisticated language",
        "Professionals": "Use formal, business-appropriate language",
        "General": ""
    }
    
    # Rhyme scheme for poems
    rhyme_map = {
        "Free Verse": "Use free verse (no specific rhyme scheme)",
        "ABAB": "Use ABAB rhyme scheme",
        "AABB": "Use AABB rhyme scheme (couplets)",
        "ABCB": "Use ABCB rhyme scheme"
    }
    
    # Build the prompt
    prompt = f"{content_instructions.get(content_type, 'Write content')} about '{keyword}'. "
    
    if language != "English":
        prompt += f"{language_map.get(language, '')}. "
    
    if tone != "Inspirational":
        prompt += f"{tone_map.get(tone, '')}. "
    
    if audience != "General":
        prompt += f"{audience_map.get(audience, '')}. "
    
    if content_type == "Poem" and rhyme_scheme != "Free Verse":
        prompt += f"{rhyme_map.get(rhyme_scheme, '')}. "
    
    return prompt.strip()


def load_history():
    """Load history from JSON file."""
    history_file = "content_history.json"
    if os.path.exists(history_file):
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []


def save_to_history(record):
    """Save a record to history."""
    history = load_history()
    record["id"] = len(history) + 1
    record["timestamp"] = datetime.datetime.now().isoformat()
    record["favorite"] = False
    history.append(record)
    
    with open("content_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def toggle_favorite(item_id):
    """Toggle favorite status of an item."""
    history = load_history()
    for item in history:
        if item["id"] == item_id:
            item["favorite"] = not item.get("favorite", False)
            break
    
    with open("content_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def delete_content(item_id):
    """Delete a content item from history."""
    history = load_history()
    updated_history = [item for item in history if item["id"] != item_id]
    
    with open("content_history.json", "w", encoding="utf-8") as f:
        json.dump(updated_history, f, ensure_ascii=False, indent=2)
    
    return len(history) != len(updated_history)  # Return True if item was deleted


def search_history(query, filter_type=None, filter_favorite=False):
    """Search through history with filters."""
    history = load_history()
    results = []
    
    for item in history:
        # Filter by favorite
        if filter_favorite and not item.get("favorite", False):
            continue
            
        # Filter by type
        if filter_type and filter_type != "All" and item.get("content_type") != filter_type:
            continue
            
        # Search in content
        if query.lower() in item.get("keyword", "").lower() or \
           query.lower() in item.get("output", "").lower() or \
           query.lower() in item.get("tags", "").lower():
            results.append(item)
    
    return results


def export_history_as_text():
    """Export history as formatted text."""
    history = load_history()
    content = "AI Quote & Poem Generator - Content History\n"
    content += "=" * 50 + "\n\n"
    
    for item in history:
        content += f"ID: {item.get('id', 'N/A')}\n"
        content += f"Keyword: {item.get('keyword', 'N/A')}\n"
        content += f"Type: {item.get('content_type', 'N/A')}\n"
        content += f"Language: {item.get('language', 'N/A')}\n"
        content += f"Tone: {item.get('tone', 'N/A')}\n"
        content += f"Date: {item.get('timestamp', 'N/A')}\n"
        content += f"Favorite: {'Yes' if item.get('favorite') else 'No'}\n"
        content += f"Tags: {item.get('tags', 'None')}\n"
        content += f"Content:\n{item.get('output', 'N/A')}\n"
        content += "-" * 30 + "\n\n"
    
    return content


def apply_theme(theme_name):
    """Apply custom CSS themes."""
    themes = {
        "Light": {
            "bg_color": "#ffffff",
            "text_color": "#000000",
            "accent_color": "#007bff",
            "secondary_color": "#f8f9fa",
            "content_bg": "linear-gradient(135deg, #007bff 0%, #0056b3 100%)",
            "content_text": "#ffffff"
        },
        "Dark": {
            "bg_color": "#1a1a1a",
            "text_color": "#ffffff", 
            "accent_color": "#58a6ff",
            "secondary_color": "#2d2d2d",
            "content_bg": "linear-gradient(135deg, #1f6feb 0%, #4969f5 100%)",
            "content_text": "#ffffff"
        },
        "Ocean": {
            "bg_color": "#f0f8ff",
            "text_color": "#0d47a1",
            "accent_color": "#1976d2",
            "secondary_color": "#e3f2fd",
            "content_bg": "linear-gradient(135deg, #1976d2 0%, #42a5f5 100%)",
            "content_text": "#ffffff"
        },
        "Forest": {
            "bg_color": "#f1f8e9",
            "text_color": "#1b5e20",
            "accent_color": "#388e3c",
            "secondary_color": "#c8e6c9",
            "content_bg": "linear-gradient(135deg, #388e3c 0%, #66bb6a 100%)",
            "content_text": "#ffffff"
        }
    }
    
    theme = themes.get(theme_name, themes["Light"])
    
    # Special handling for Dark theme
    if theme_name == "Dark":
        css = f"""
        <style>
        /* Force dark theme text visibility */
        .stApp {{
            background-color: {theme['bg_color']} !important;
        }}
        
        .stMarkdown, p, h1, h2, h3, span, div {{
            color: {theme['text_color']} !important;
        }}
        
        /* Improve select box visibility */
        .stSelectbox > div > div {{
            background-color: {theme['secondary_color']} !important;
            border: 1px solid {theme['accent_color']} !important;
        }}
        
        .stSelectbox > div > div > div {{
            color: {theme['text_color']} !important;
        }}
        
        /* Style dropdown options */
        .stSelectbox div[data-baseweb="select"] > div {{
            background-color: #2b2b2b !important;
        }}
        
        .stSelectbox div[data-baseweb="select"] span {{
            color: #ffffff !important;
        }}
        
        /* Dropdown menu popup */
        div[data-baseweb="popover"] {{
            background-color: #2b2b2b !important;
        }}
        
        div[data-baseweb="popover"] div[role="listitem"] {{
            color: #ffffff !important;
        }}
        
        div[data-baseweb="popover"] div[role="listitem"]:hover {{
            background-color: #3d3d3d !important;
        }}
        
        /* Selected option */
        div[data-baseweb="select"] [data-testid="stMarkdown"] {{
            color: #ffffff !important;
        }}
        
        /* Dropdown options text */
        .stSelectbox [role="option"] {{
            color: #ffffff !important;
        }}
        
        /* Radio buttons */
        .stRadio > div {{
            background-color: transparent !important;
        }}
        
        .stRadio label {{
            color: {theme['text_color']} !important;
            font-weight: 500 !important;
        }}
        
        section[data-testid="stSidebar"] {{
            background-color: {theme['secondary_color']} !important;
        }}
        
        section[data-testid="stSidebar"] * {{
            color: {theme['text_color']} !important;
        }}
        
        button {{
            background-color: {theme['accent_color']} !important;
            color: #000000 !important;
        }}
        </style>
        """
    # Force light theme to have proper visibility
    elif theme_name == "Light":
        css = f"""
        <style>
        /* Force light theme styling */
        .stApp {{
            background-color: #ffffff !important;
        }}
        
        /* Sidebar styling for light theme */
        section[data-testid="stSidebar"] {{
            background-color: #f8f9fa !important;
        }}
        
        section[data-testid="stSidebar"] > div {{
            background-color: #f8f9fa !important;
        }}
        
        /* All text elements */
        .stApp, .stApp p, .stApp span, .stApp div, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {{
            color: #000000 !important;
        }}
        
        /* Sidebar text */
        section[data-testid="stSidebar"] * {{
            color: #000000 !important;
        }}
        
        /* Input fields */
        .stTextInput input {{
            background-color: #ffffff !important;
            color: #000000 !important;
            border: 1px solid #ced4da !important;
        }}
        
        .stSelectbox select {{
            background-color: #ffffff !important;
            color: #000000 !important;
        }}
        
        .stSelectbox > div > div {{
            background-color: #ffffff !important;
            color: #000000 !important;
        }}
        
        /* Radio buttons */
        .stRadio label {{
            color: #000000 !important;
        }}
        
        /* Buttons */
        .stButton button {{
            background-color: #007bff !important;
            color: #ffffff !important;
        }}
        
        /* Content box */
        .content-box {{
            background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
            color: #ffffff;
            padding: 20px;
            border-radius: 15px;
            margin: 10px 0;
            font-family: 'Georgia', serif;
            font-style: italic;
            text-align: center;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        }}
        
        /* History items */
        .history-item {{
            background-color: #f8f9fa !important;
            color: #000000 !important;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            border-left: 4px solid #007bff;
        }}
        
        .history-item * {{
            color: #000000 !important;
        }}
        
        /* Metrics */
        [data-testid="metric-container"] {{
            color: #000000 !important;
        }}
        
        [data-testid="metric-container"] * {{
            color: #000000 !important;
        }}
        </style>
        """
    else:
        # For other themes (Dark, Ocean, Forest) - keep original styling
        css = f"""
        <style>
        .stApp {{
            background-color: {theme['bg_color']};
            color: {theme['text_color']};
        }}
        
        /* Main text elements */
        .stMarkdown, .stText, p, span, div {{
            color: {theme['text_color']} !important;
        }}
        
        /* Headers */
        h1, h2, h3, h4, h5, h6 {{
            color: {theme['text_color']} !important;
        }}
        
        /* Sidebar */
        .css-1d391kg {{
            background-color: {theme['secondary_color']};
        }}
        
        .css-1d391kg .stMarkdown, .css-1d391kg p {{
            color: {theme['text_color']} !important;
        }}
        
        /* Input fields */
        .stTextInput > div > div > input {{
            background-color: {theme['secondary_color']} !important;
            color: {theme['text_color']} !important;
            border: 1px solid {theme['accent_color']} !important;
        }}
        
        .stSelectbox > div > div {{
            background-color: {theme['secondary_color']} !important;
            color: {theme['text_color']} !important;
        }}
        
        .stSelectbox > div > div > div {{
            color: {theme['text_color']} !important;
        }}
        
        /* Buttons */
        .stButton > button {{
            background-color: {theme['accent_color']} !important;
            color: #ffffff !important;
            border: none !important;
        }}
        
        .stButton > button:hover {{
            background-color: {theme['accent_color']}dd !important;
        }}
        
        /* Radio buttons */
        .stRadio > div {{
            color: {theme['text_color']} !important;
        }}
        
        .stRadio > div > label {{
            color: {theme['text_color']} !important;
        }}
        
        /* Content box */
        .content-box {{
            background: {theme['content_bg']};
            color: {theme['content_text']};
            padding: 20px;
            border-radius: 15px;
            margin: 10px 0;
            font-family: 'Georgia', serif;
            font-style: italic;
            text-align: center;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
            border: 2px solid {theme['accent_color']};
        }}
        
        /* History items */
        .history-item {{
            background-color: {theme['secondary_color']} !important;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            border-left: 4px solid {theme['accent_color']};
            color: {theme['text_color']} !important;
        }}
        
        .history-item h4 {{
            color: {theme['text_color']} !important;
        }}
        
        .history-item p {{
            color: {theme['text_color']} !important;
        }}
        
        .history-item small {{
            color: {theme['text_color']} !important;
        }}
        
        /* Metrics */
        [data-testid="metric-container"] {{
            color: {theme['text_color']} !important;
        }}
        
        [data-testid="metric-container"] > div {{
            color: {theme['text_color']} !important;
        }}
        </style>
        """
    
    st.markdown(css, unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="AI Content Generator Pro", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if 'theme' not in st.session_state:
        st.session_state.theme = "Light"
    if 'page' not in st.session_state:
        st.session_state.page = "Generator"
    
    # Apply theme
    apply_theme(st.session_state.theme)
    
    # Sidebar for navigation and settings
    with st.sidebar:
        st.title("🎨 Content Generator Pro")
        
        # Theme selector
        st.markdown("### 🎨 Theme")
        new_theme = st.selectbox(
            "Choose Theme",
            ["Light", "Dark", "Ocean", "Forest"],
            index=["Light", "Dark", "Ocean", "Forest"].index(st.session_state.theme)
        )
        if new_theme != st.session_state.theme:
            st.session_state.theme = new_theme
            st.rerun()
        
        # Navigation
        st.markdown("### 📍 Navigation")
        page = st.radio(
            "Go to",
            ["Generator", "History", "Analytics"],
            index=["Generator", "History", "Analytics"].index(st.session_state.page)
        )
        if page != st.session_state.page:
            st.session_state.page = page
            st.rerun()
        
        # Quick stats
        history = load_history()
        st.markdown("### 📊 Quick Stats")
        st.metric("Total Generated", len(history))
        st.metric("Favorites", len([h for h in history if h.get("favorite", False)]))
    
    # Main content based on page
    if st.session_state.page == "Generator":
        show_generator_page()
    elif st.session_state.page == "History":
        show_history_page()
    elif st.session_state.page == "Analytics":
        show_analytics_page()


def show_generator_page():
    """Main content generation page."""
    
    # Header
    st.title("✍️ AI Content Generator Pro")
    st.markdown("Create beautiful quotes, poems, and more with advanced AI customization!")
    
    # Main input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📝 Enter Your Inspiration")
        keyword = st.text_input(
            "Keyword / Topic", 
            placeholder="e.g. Love, Success, Adventure, Dreams...",
            help="Enter any word or phrase you want content about"
        )
        
        # Content type selection with more options
        content_type = st.selectbox(
            "Content Type",
            ["Quote", "Poem", "Haiku", "Motivational Saying", "Social Media Caption", "Song Lyrics", "Story Beginning"],
            help="Choose what type of content you want to generate"
        )
    
    with col2:
        st.markdown("### ⚙️ Customization")
        
        # Language selection
        language = st.selectbox(
            "Language",
            ["English", "Hindi", "Marathi", "Spanish", "French", "German"]
        )
        
        # Tone/Style selection
        tone = st.selectbox(
            "Tone/Style",
            ["Inspirational", "Funny", "Serious", "Romantic", "Professional"]
        )
        
        # Target audience
        audience = st.selectbox(
            "Target Audience",
            ["General", "Kids", "Adults", "Professionals"]
        )
        
        # Rhyme scheme (only for poems)
        if content_type == "Poem":
            rhyme_scheme = st.selectbox(
                "Rhyme Scheme",
                ["Free Verse", "ABAB", "AABB", "ABCB"]
            )
        else:
            rhyme_scheme = "Free Verse"
    
    # Length adjustment
    if content_type not in ["Haiku", "Social Media Caption"]:
        length_words = st.slider(
            "Approximate length (words)", 
            min_value=4, 
            max_value=150, 
            value=12 if content_type == "Quote" else 50
        )
    else:
        length_words = 17 if content_type == "Haiku" else 25
    
    # Tags for organization
    tags = st.text_input(
        "Tags (comma-separated)", 
        placeholder="personal, motivation, work...",
        help="Add tags to help organize your content"
    )
    
    # Generate button
    col_gen1, col_gen2, col_gen3 = st.columns([1, 2, 1])
    with col_gen2:
        if st.button("🚀 Generate Content", type="primary", use_container_width=True):
            if not keyword.strip():
                st.warning("⚠️ Please enter a keyword first.")
            else:
                prompt = build_prompt(keyword.strip(), content_type, language, tone, audience, rhyme_scheme, length_words)
                
                with st.spinner("✨ Creating your masterpiece..."):
                    try:
                        output = generate_text(prompt, temperature=0.7, max_tokens=300)
                        
                        if output:
                            # Store in session state
                            st.session_state.current_content = {
                                "keyword": keyword.strip(),
                                "content_type": content_type,
                                "language": language,
                                "tone": tone,
                                "audience": audience,
                                "rhyme_scheme": rhyme_scheme,
                                "length_words": length_words,
                                "tags": tags.strip(),
                                "output": output
                            }
                            
                    except Exception as e:
                        st.error(f"❌ Generation failed: {e}")
                        st.info("💡 Check your API key configuration")
    
    # Display generated content
    if hasattr(st.session_state, 'current_content') and st.session_state.current_content:
        st.markdown("---")
        st.subheader("🎨 Generated Content")
        
        content = st.session_state.current_content
        
        # Display content in styled box
        st.markdown(f"""
        <div class="content-box">
            {content['output']}
        </div>
        """, unsafe_allow_html=True)
        
        # Content details
        with st.expander("📋 Content Details"):
            col_det1, col_det2 = st.columns(2)
            with col_det1:
                st.write(f"**Type:** {content['content_type']}")
                st.write(f"**Language:** {content['language']}")
                st.write(f"**Tone:** {content['tone']}")
            with col_det2:
                st.write(f"**Audience:** {content['audience']}")
                st.write(f"**Keywords:** {content['keyword']}")
                st.write(f"**Tags:** {content.get('tags', 'None')}")
        
        # Action buttons
        col_act1, col_act2, col_act3, col_act4 = st.columns(4)
        
        with col_act1:
            st.download_button(
                "📥 Download",
                data=content['output'],
                file_name=f"{content['keyword']}_{content['content_type']}.txt",
                use_container_width=True
            )
        
        with col_act2:
            if st.button("💾 Save to History", use_container_width=True):
                save_to_history(content)
                st.success("✅ Saved to history!")
                st.balloons()
        
        with col_act3:
            if st.button("🔄 Regenerate", use_container_width=True):
                # Regenerate with the same parameters
                prompt = build_prompt(
                    content['keyword'],
                    content['content_type'],
                    content['language'],
                    content['tone'],
                    content['audience'],
                    content.get('rhyme_scheme', 'Free Verse'),
                    content['length_words']
                )
                
                with st.spinner("✨ Regenerating your masterpiece..."):
                    try:
                        output = generate_text(prompt, temperature=0.7, max_tokens=300)
                        
                        if output:
                            # Update the output in the current content
                            st.session_state.current_content['output'] = output
                            st.rerun()
                            
                    except Exception as e:
                        st.error(f"❌ Generation failed: {e}")
                        st.info("💡 Check your API key configuration")
        
        with col_act4:
            # Copy button
            copy_js = f"""
            <script>
            function copyContent() {{
                navigator.clipboard.writeText(`{content['output']}`).then(function() {{
                    alert('📋 Copied to clipboard!');
                }});
            }}
            </script>
            <button onclick="copyContent()" style="
                background: #28a745;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 5px;
                cursor: pointer;
                width: 100%;
                font-size: 14px;
            ">
                📋 Copy
            </button>
            """
            st.components.v1.html(copy_js, height=40)


def show_history_page():
    """History management page."""
    st.title("📚 Content History")
    
    history = load_history()
    
    if not history:
        st.info("No content generated yet. Go to the Generator page to create some!")
        return
    
    # Header with clear all option
    col_header1, col_header2 = st.columns([3, 1])
    with col_header2:
        if st.button("🗑️ Clear All History", help="Delete all content history"):
            if st.session_state.get('confirm_clear_all', False):
                # Clear all history
                with open("content_history.json", "w", encoding="utf-8") as f:
                    json.dump([], f, ensure_ascii=False, indent=2)
                st.success("✅ All history cleared!")
                st.session_state.confirm_clear_all = False
                st.rerun()
            else:
                st.session_state.confirm_clear_all = True
                st.error("⚠️ This will delete ALL your content! Click again to confirm.")
    
    # Cancel clear all if needed
    if st.session_state.get('confirm_clear_all', False):
        if st.button("❌ Cancel Clear All"):
            st.session_state.confirm_clear_all = False
            st.rerun()
    
    # Search and filter controls
    col_search1, col_search2, col_search3, col_search4 = st.columns([2, 1, 1, 1])
    
    with col_search1:
        search_query = st.text_input("🔍 Search content", placeholder="Search keywords, content, tags...")
    
    with col_search2:
        filter_type = st.selectbox("Filter by Type", ["All"] + list(set([h.get("content_type", "Unknown") for h in history])))
    
    with col_search3:
        filter_favorite = st.checkbox("⭐ Favorites Only")
    
    with col_search4:
        if st.button("📤 Export All"):
            export_data = export_history_as_text()
            st.download_button(
                "Download History",
                data=export_data,
                file_name=f"content_history_{datetime.datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )
    
    # Search and filter results
    if search_query or filter_type != "All" or filter_favorite:
        filtered_history = search_history(search_query, filter_type, filter_favorite)
    else:
        filtered_history = history
    
    # Display results
    st.markdown(f"**Showing {len(filtered_history)} of {len(history)} items**")
    
    if filtered_history:
        # Bulk actions
        col_bulk1, col_bulk2, col_bulk3 = st.columns([1, 1, 4])
        with col_bulk1:
            if st.button("🗑️ Delete All Filtered", help="Delete all currently filtered items"):
                if st.session_state.get('confirm_bulk_delete', False):
                    # Delete all filtered items
                    history = load_history()
                    filtered_ids = [item['id'] for item in filtered_history]
                    updated_history = [item for item in history if item['id'] not in filtered_ids]
                    
                    with open("content_history.json", "w", encoding="utf-8") as f:
                        json.dump(updated_history, f, ensure_ascii=False, indent=2)
                    
                    st.success(f"✅ Deleted {len(filtered_ids)} items!")
                    st.session_state.confirm_bulk_delete = False
                    st.rerun()
                else:
                    st.session_state.confirm_bulk_delete = True
                    st.warning("⚠️ Click again to confirm bulk deletion!")
        
        with col_bulk2:
            if st.session_state.get('confirm_bulk_delete', False):
                if st.button("❌ Cancel"):
                    st.session_state.confirm_bulk_delete = False
                    st.rerun()
    
    for item in reversed(filtered_history):  # Show newest first
        with st.container():
            col_hist1, col_hist2 = st.columns([4, 1])
            
            with col_hist1:
                st.markdown(f"""
                <div class="history-item">
                    <h4>{'⭐ ' if item.get('favorite') else ''}{item.get('content_type', 'Unknown')} - {item.get('keyword', 'No keyword')}</h4>
                    <p><strong>Content:</strong> {item.get('output', 'No content')[:200]}{'...' if len(item.get('output', '')) > 200 else ''}</p>
                    <small>
                        <strong>Language:</strong> {item.get('language', 'N/A')} | 
                        <strong>Tone:</strong> {item.get('tone', 'N/A')} | 
                        <strong>Tags:</strong> {item.get('tags', 'None')} | 
                        <strong>Date:</strong> {item.get('timestamp', 'N/A')[:10]}
                    </small>
                </div>
                """, unsafe_allow_html=True)
            
            with col_hist2:
                # Action buttons in a grid
                col_btn1, col_btn2 = st.columns(2)
                
                with col_btn1:
                    if st.button(f"{'💔' if item.get('favorite') else '❤️'}", key=f"fav_{item['id']}", help="Toggle favorite"):
                        toggle_favorite(item['id'])
                        st.rerun()
                
                with col_btn2:
                    if st.button("�️", key=f"delete_{item['id']}", help="Delete this item"):
                        if st.session_state.get(f'confirm_delete_{item["id"]}', False):
                            if delete_content(item['id']):
                                st.success("✅ Item deleted!")
                                st.session_state[f'confirm_delete_{item["id"]}'] = False
                                st.rerun()
                        else:
                            st.session_state[f'confirm_delete_{item["id"]}'] = True
                            st.warning("⚠️ Click delete again to confirm!")
                
                # Second row of buttons
                col_btn3, col_btn4 = st.columns(2)
                
                with col_btn3:
                    if st.button("�", key=f"copy_{item['id']}", help="View full content"):
                        st.code(item.get('output', ''), language='text')
                
                with col_btn4:
                    st.download_button(
                        "📥",
                        data=item.get('output', ''),
                        file_name=f"{item.get('keyword', 'content')}_{item.get('content_type', 'unknown')}.txt",
                        key=f"dl_{item['id']}",
                        help="Download this item"
                    )
                
                # Show cancel button if delete confirmation is pending
                if st.session_state.get(f'confirm_delete_{item["id"]}', False):
                    if st.button("❌ Cancel", key=f"cancel_{item['id']}"):
                        st.session_state[f'confirm_delete_{item["id"]}'] = False
                        st.rerun()


def show_analytics_page():
    """Analytics and insights page."""
    st.title("📊 Analytics & Insights")
    
    history = load_history()
    
    if not history:
        st.info("No data available yet. Generate some content first!")
        return
    
    # Basic statistics
    col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
    
    with col_stats1:
        st.metric("Total Content", len(history))
    
    with col_stats2:
        favorites = len([h for h in history if h.get("favorite", False)])
        st.metric("Favorites", favorites)
    
    with col_stats3:
        languages = set([h.get("language", "Unknown") for h in history])
        st.metric("Languages Used", len(languages))
    
    with col_stats4:
        types = set([h.get("content_type", "Unknown") for h in history])
        st.metric("Content Types", len(types))
    
    # Charts and analysis
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("Content Types Distribution")
        type_counts = {}
        for item in history:
            content_type = item.get("content_type", "Unknown")
            type_counts[content_type] = type_counts.get(content_type, 0) + 1
        
        if type_counts:
            st.bar_chart(type_counts)
    
    with col_chart2:
        st.subheader("Language Usage")
        lang_counts = {}
        for item in history:
            language = item.get("language", "Unknown")
            lang_counts[language] = lang_counts.get(language, 0) + 1
        
        if lang_counts:
            st.bar_chart(lang_counts)
    
    # Recent activity
    st.subheader("Recent Activity")
    recent_items = sorted(history, key=lambda x: x.get("timestamp", ""), reverse=True)[:5]
    
    for item in recent_items:
        st.write(f"**{item.get('content_type', 'Unknown')}** about *{item.get('keyword', 'Unknown')}* - {item.get('timestamp', 'Unknown')[:10]}")
    
    # Popular keywords
    st.subheader("Popular Keywords")
    keywords = {}
    for item in history:
        keyword = item.get("keyword", "").lower()
        if keyword:
            keywords[keyword] = keywords.get(keyword, 0) + 1
    
    if keywords:
        sorted_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:10]
        for keyword, count in sorted_keywords:
            st.write(f"**{keyword.title()}**: {count} times")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 14px;">
        Made with ❤️ for creative minds | Powered by Google Gemini AI<br>
        Enhanced with Advanced Features & Beautiful UI
        
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()