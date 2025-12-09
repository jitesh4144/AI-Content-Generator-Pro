# 🎭 AI Quote & Poem Generator

A beautiful web application that generates inspiring quotes and poems using Google's Gemini AI. Perfect for SGU students who want to transform their thoughts into artistic expressions!

## ✨ Features

- **Intelligent Generation**: Uses Google Gemini AI to create personalized quotes and poems
- **SGU Student Theme**: Special mode tailored for student life experiences
- **Beautiful Interface**: Modern, user-friendly design with gradients and emojis
- **Download & Save**: Export your creations as text files or save to history
- **Copy to Clipboard**: Easy sharing with one-click copy functionality
- **Customizable Length**: Adjust the word count for your preferred style

## 🚀 Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up your Gemini API Key**:
   - Create a `.env` file in the project directory
   - Add your Gemini API key:
     ```
     # 🎭 AI Content Generator Pro

An advanced, feature-rich web application that generates inspiring quotes, poems, and various content types using Google's Gemini AI. Perfect for anyone who wants to transform their thoughts into artistic expressions with professional customization options!

## ✨ Enhanced Features

### 🎨 **Advanced UI/UX**
- **🌙 Dark/Light Theme Toggle** - Switch between Light, Dark, Ocean, and Forest themes
- **🎨 Custom Color Schemes** - Beautiful gradient designs for each theme
- **📱 Mobile-Responsive Design** - Optimized for all device sizes
- **🖋️ Typography Enhancement** - Professional fonts and styling
- **📊 Multi-Page Layout** - Generator, History, and Analytics pages

### 📝 **Multiple Content Types**
- **📜 Quotes** - Inspiring and memorable quotes
- **🎵 Poems** - Custom length with rhyme scheme options
- **🌸 Haikus** - Traditional 3-line Japanese poems (5-7-5 syllables)
- **💪 Motivational Sayings** - Uplifting motivational content
- **📱 Social Media Captions** - Perfect for Instagram, Twitter, Facebook
- **🎼 Song Lyrics** - Rhythmic and flowing song verses
- **📖 Story Beginnings** - Engaging opening paragraphs

### 🌍 **Advanced Customization**
- **🌐 Multi-Language Support** - English, Hindi, Marathi, Spanish, French, German
- **🎭 Tone/Style Selection** - Funny, Serious, Romantic, Professional, Inspirational
- **👥 Target Audience** - Kids, Adults, Professionals, General
- **🎵 Rhyme Schemes** - Free Verse, ABAB, AABB, ABCB patterns
- **📏 Length Control** - Customizable word count for each content type

### 📚 **Complete History Management**
- **💾 Smart Saving** - Automatic content history with metadata
- **🔍 Advanced Search** - Search by keywords, content, or tags
- **⭐ Favorites System** - Mark and filter favorite content
- **🏷️ Tagging System** - Organize content with custom tags
- **📤 Export Options** - Download individual items or full history
- **📊 Content Analytics** - View usage patterns and statistics

## 🚀 Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up your Gemini API Key**:
   ```env
   GEMINI_API_KEY="your_api_key_here"
   GEMINI_MODEL="gemini-2.0-flash"
   MODEL_PROVIDER="gemini"
   ```

3. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

4. **Access the App**:
   - Navigate to `http://localhost:8501`
   - Explore all the advanced features!

## 🎯 How to Use

### **Generator Page**
1. **Enter Keywords** - Type any topic or emotion
2. **Choose Content Type** - Select from 7 different types
3. **Customize Settings** - Language, tone, audience, rhyme scheme
4. **Set Length** - Adjust word count with slider
5. **Add Tags** - Organize your content
6. **Generate & Save** - Create and save to history

### **History Page**
- **View All Content** - Browse your complete generation history
- **Search & Filter** - Find content by type, keywords, or favorites
- **Manage Favorites** - Mark important content with ⭐
- **Export Data** - Download individual items or complete history

### **Analytics Page**
- **Usage Statistics** - See your content generation patterns
- **Popular Keywords** - Track your most-used topics
- **Type Distribution** - Visualize content type preferences
- **Recent Activity** - View your latest creations

## 💡 Example Outputs

| Input | Type | Language | Tone | Output Example |
|-------|------|----------|------|----------------|
| "Success" | Quote | English | Professional | *"Success is not final, failure is not fatal."* |
| "प्रेम" | Poem | Hindi | Romantic | *Beautiful Hindi verses about love* |
| "यश" | Quote | Marathi | Inspirational | *Marathi motivational quote about success* |
| "Adventure" | Haiku | English | Inspirational | *Mountain peaks call / Adventure awaits the brave / Journey starts within* |
| "Motivation" | Social Caption | English | Funny | *"Monday motivation: Coffee first, conquer world second ☕"* |

## 🛠️ Technical Features

### **Framework & Architecture**
- **Frontend**: Streamlit with custom CSS theming
- **AI Model**: Google Gemini 2.0 Flash
- **Data Storage**: JSON-based history management
- **Responsive Design**: Mobile-first approach

### **Advanced Functionality**
- **Session State Management** - Persistent user preferences
- **Real-time Search** - Instant content filtering
- **Dynamic Theming** - Live theme switching
- **Error Handling** - Graceful API failure recovery
- **Performance Optimization** - Efficient data loading

## 🎨 Theme Options

### **🌞 Light Theme**
- Clean white background
- Professional blue accents
- Perfect for daytime use

### **🌙 Dark Theme**
- Modern dark interface
- Blue accent colors
- Easy on the eyes

### **🌊 Ocean Theme**
- Calming blue tones
- Water-inspired gradients
- Peaceful and serene

### **🌲 Forest Theme**
- Natural green colors
- Earth-inspired design
- Fresh and organic

## � Analytics & Insights

- **Content Statistics** - Total generations, favorites, languages
- **Usage Patterns** - Most popular content types and keywords
- **Trend Analysis** - Your creative preferences over time
- **Export Reports** - Download usage data for analysis

## 🎉 Perfect For

- **🎨 Creative Writers** - Inspiration and idea generation
- **📱 Social Media Managers** - Quick caption creation
- **� Students & Educators** - Creative writing exercises
- **💼 Content Creators** - Blog posts and marketing copy
- **💝 Personal Use** - Gifts, cards, and messages
- **🌍 Multilingual Users** - Content in multiple languages

## 🌟 Advanced Features

### **Smart Content Generation**
- Context-aware prompts based on all selections
- Intelligent length adjustment per content type
- Cultural sensitivity for different languages
- Audience-appropriate language complexity

### **Professional Organization**
- Comprehensive tagging system
- Advanced search capabilities
- Bulk export functionality
- Favorite management

### **User Experience**
- Intuitive navigation between pages
- Quick stats in sidebar
- Responsive button layouts
- Beautiful content display

## 🔧 Configuration

Customize the application in your `.env` file:

```env
# Required
GEMINI_API_KEY="your_api_key_here"

# Optional
GEMINI_MODEL="gemini-2.0-flash"
MODEL_PROVIDER="gemini"
```

## 🌟 Made with ❤️

Built for creative minds who demand professional tools with beautiful design. Whether you're a content creator, writer, marketer, or just someone who loves creative expression, this tool provides everything you need with an interface that's both powerful and delightful to use.

---

