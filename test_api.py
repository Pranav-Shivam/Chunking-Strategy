import requests
import json

def test_upload():
    url = "http://localhost:8000/upload"
    
    with open("example.pdf", "rb") as f:
        files = {"file": ("example.pdf", f, "application/pdf")}
        response = requests.post(url, files=files)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Success! Processed {result['total_chunks']} chunks")
        print(f"Document ID: {result['doc_id']}")
        
        with open("api_output.json", "w") as f:
            json.dump(result, f, indent=2)
        print("Results saved to api_output.json")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_upload() 