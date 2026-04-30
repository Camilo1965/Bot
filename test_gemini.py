import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

def test_gemini():
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    
    if not api_key:
        print("❌ ERROR: No se encontró GEMINI_API_KEY en el archivo .env")
        return

    print(f"Testing Gemini API with key: {api_key[:10]}...")
    
    # Probamos con 1.5-flash que es el más estable
    model_name = "gemini-1.5-flash" 
    
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents="Hola, responde con la palabra 'OK' si recibes este mensaje."
        )
        print(f"✅ CONEXIÓN EXITOSA!")
        print(f"Respuesta de Gemini: {response.text}")
    except Exception as e:
        print(f"❌ ERROR DE API:")
        print(f"Tipo: {type(e).__name__}")
        print(f"Detalle: {str(e)}")

if __name__ == "__main__":
    test_gemini()
