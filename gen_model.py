"""
At the command line, only need to run once to install the package via pip:

$ pip install google-generativeai
"""

import google.generativeai as genai

genai.configure(api_key="AIzaSyDfD87jj4V-3tXK_jEz2sLme-K7aUPPQHw")

# Set up the model
generation_config = {
  "temperature": 0.9,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 2048,
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
]

model = genai.GenerativeModel(model_name="gemini-1.0-pro",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

convo = model.start_chat(history=[
  {
    "role": "user",
    "parts": ["Mood : sad\nAspect : mental\nReason : im upset because i can't work out"]
  },
  {
    "role": "model",
    "parts": ["## Understanding Your Mood\n\nIt seems like you're feeling sad and mentally down because you're unable to work out.  That's completely understandable, exercise is a great way to boost your mood and overall well-being. \n\n## Recommended Activities:\n\nSince you can't work out right now, let's explore some alternative activities that can help lift your spirits:\n\n**Relaxation and Self-Care:**\n\n1. **Mindful Meditation:** Try a guided meditation app or simply focus on your breath for a few minutes. This can help calm your mind and reduce stress.\n2. **Warm Bath with Soothing Music:** Create a spa-like experience at home with a relaxing bath. Add some essential oils or bath salts for an extra touch. \n3. **Gentle Stretching or Yoga:**  Even if you can't do a full workout, some light stretches or yoga poses can help improve your mood and energy levels.\n\n**Engagement and Distraction:**\n\n4. **Creative Expression:**  Engage in a creative activity like painting, drawing, writing, or playing music. Expressing yourself can be a great way to process emotions.\n5. **Connect with Loved Ones:** Spend time with supportive friends and family. Talking about your feelings or simply enjoying their company can make a big difference.\n6. **Explore a New Hobby:**  Is there something you've always wanted to try? Now might be a good time to explore a new hobby that can bring you joy and distract you from your current mood. \n\n**Remember, be kind to yourself and allow yourself to feel your emotions.  These suggested activities are just starting points, feel free to explore and find what works best for you.**"]
  },
])

convo.send_message("YOUR_USER_INPUT")
print(convo.last.text)