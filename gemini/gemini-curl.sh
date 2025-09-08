curl -X POST \
-H "Content-Type: application/json" \
-d '{
  "contents": [
    {
      "parts": [
        {
          "text": "Tell me a short story about a cat and a dog."
        }
      ]
    }
  ]
}' \
"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=<API Key>"
